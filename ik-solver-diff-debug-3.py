import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import sys
import time
import random
import wandb
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_robot_choice, reconstruct_pose_modified, epoch_time, count_parameters

# --- Dataset Loader ---
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        self.q = torch.from_numpy(np.load(filename_Q)).float()
        self.pose = torch.from_numpy(np.load(filename_D)).float()
        assert self.q.shape[0] == self.pose.shape[0]

        # --- Normalization ---
        self.q_mean = self.q.mean(dim=0, keepdim=True)
        self.q_std = self.q.std(dim=0, keepdim=True) + 1e-8
        self.pose_mean = self.pose.mean(dim=0, keepdim=True)
        self.pose_std = self.pose.std(dim=0, keepdim=True) + 1e-8

        self.q = (self.q - self.q_mean) / self.q_std
        self.pose = (self.pose - self.pose_mean) / self.pose_std

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {"q": self.q[idx], "pose": self.pose[idx]}



class DiffIKDataset2(Dataset):
    def __init__(self, D, Q):
        self.q = torch.from_numpy(Q).float()
        self.pose = torch.from_numpy(D).float()
        assert self.q.shape[0] == self.pose.shape[0]

        # --- Normalization ---
        self.q_mean = self.q.mean(dim=0, keepdim=True)
        self.q_std = self.q.std(dim=0, keepdim=True) + 1e-8
        self.pose_mean = self.pose.mean(dim=0, keepdim=True)
        self.pose_std = self.pose.std(dim=0, keepdim=True) + 1e-8

        self.q = (self.q - self.q_mean) / self.q_std
        self.pose = (self.pose - self.pose_mean) / self.pose_std

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {"q": self.q[idx], "pose": self.pose[idx]}


# --- Diffusion MLP Architecture ---
# class DiffIKDenoiser(nn.Module):
#     def __init__(self, dof=7, pose_dim=7, hidden_dim=512, time_embed_dim=64):
#         super().__init__()
#         self.dof = dof
#         self.num_timesteps = 1000
#         self.time_embed = nn.Sequential(
#             nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
#         )
#         input_dim = dof + pose_dim + time_embed_dim
#         self.net = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
#             nn.Linear(hidden_dim, dof)
#         )

#     def forward(self, q_t, pose, t):
#         t_embed = self.time_embed(t.unsqueeze(-1).float())
#         x = torch.cat([q_t, pose, t_embed], dim=-1)
#         return self.net(x)

class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=1024, time_embed_dim=64, pose_embed_dim=64):
        super().__init__()
        self.dof = dof
        self.num_timesteps = 1000
        self.time_embed_dim = time_embed_dim
        self.pose_embed_dim = pose_embed_dim

        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.LeakyReLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )

        input_dim = dof + pose_embed_dim + time_embed_dim

        self.net = nn.Sequential(
            #nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), #nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), #nn.Dropout(0.1),
            nn.Linear(hidden_dim, dof)
        )

    def _sinusoidal_time_embedding(self, t, dim):
        """
        t: Tensor of shape [B] (timesteps)
        dim: int, embedding dimension (must be even)
        Returns: [B, dim]
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb  # [B, dim]

    def forward(self, q_t, pose, t):
        t_embed = self._sinusoidal_time_embedding(t.float(), self.time_embed_dim)
        pose_embed = self.pose_embed(pose)
        x = torch.cat([q_t, pose_embed, t_embed], dim=-1)
        return self.net(x)

# --- Sampling Function ---
@torch.no_grad()
def sample(model, pose, ddim_steps=50, pose_mean=None, pose_std=None, q_mean=None, q_std=None):
    model.eval()
    B = pose.shape[0]

    # Normalize pose input if stats provided
    if pose_mean is not None and pose_std is not None:
        pose = (pose - pose_mean) / pose_std

    q = torch.randn(B, model.dof).to(pose.device)
    pose = pose.repeat_interleave(1, dim=0)
    for t in reversed(range(1, ddim_steps + 1)):
        t_tensor = torch.full((q.size(0),), t, device=q.device, dtype=torch.long)
        beta = 1e-4 + (t_tensor / model.num_timesteps).unsqueeze(-1)
        noise_pred = model(q, pose, t_tensor)
        q = q - beta * noise_pred

    if q_mean is not None and q_std is not None:
        q = q * q_std + q_mean

    return q

# --- Validation ---
def validate(model, val_loader, device, robot_choice, q_stats, pose_stats):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    y_preds = []
    y_desireds = []
    q_mean, q_std = q_stats
    pose_mean, pose_std = pose_stats

    with torch.no_grad():
        for batch in val_loader:
            q_gt = batch["q"].to(device)
            pose_gt = batch["pose"].to(device)
            noise = torch.randn_like(q_gt)
            t = torch.randint(0, model.num_timesteps, (q_gt.size(0),), device=device)
            beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
            q_t = q_gt + beta * noise
            noise_pred = model(q_t, pose_gt, t)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()

            # Reverse normalization
            q_gt_denorm = q_gt * q_std + q_mean
            pose_gt_denorm = pose_gt * pose_std + pose_mean
            q_pred = sample(model, pose_gt_denorm, pose_mean=pose_mean, pose_std=pose_std, q_mean=q_mean, q_std=q_std)
            q_pred = q_pred.detach().cpu().numpy()
            q_gt_denorm = q_gt_denorm.detach().cpu().numpy()
            y_preds.append(q_pred)
            y_desireds.append(q_gt_denorm)

        monitored_total_loss = total_loss / len(val_loader)
        y_preds = np.concatenate(y_preds, axis=0)
        y_desireds = np.concatenate(y_desireds, axis=0)
        X_desireds, X_preds, X_errors = reconstruct_pose_modified(y_desireds, y_preds, robot_choice)
        X_errors_report = np.array([[X_errors.min(axis=0)],
                                    [X_errors.mean(axis=0)],
                                    [X_errors.max(axis=0)],
                                    [X_errors.std(axis=0)]]).squeeze()
        results = {
            "y_preds": y_preds,
            "X_preds": X_preds,
            "y_desireds": y_desireds,
            "X_desireds": X_desireds,
            "X_errors": X_errors,
            "X_errors_report": X_errors_report
        }
        return monitored_total_loss, results

# --- Training Loop ---
def train_loop(model, train_loader, val_loader, max_epochs=10, lr=1e-4, robot_name="panda", save_on_wand=True, print_steps=100):
    save_path = "results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"[Results saved in: {save_path}]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training on device: {device}]")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    robot_choice = get_robot_choice(robot_name)

    if save_on_wand:
        run = wandb.init(
            entity="jacketdembys",
            project="diffik",
            group=f"MLP_{robot_choice}_Data_2.5M",
            name=f"MLP_{robot_choice}_Data_2.5M_Bs_128_Opt_AdamW"
        )

    best_pose_loss = float('inf')
    best_epoch = 0
    start_training_time = time.monotonic()

    sample_batch = next(iter(train_loader))
    q_mean = sample_batch["q"].mean(dim=0, keepdim=True).to(device)
    q_std = sample_batch["q"].std(dim=0, keepdim=True).to(device)
    pose_mean = sample_batch["pose"].mean(dim=0, keepdim=True).to(device)
    pose_std = sample_batch["pose"].std(dim=0, keepdim=True).to(device)
    q_stats = (q_mean, q_std)
    pose_stats = (pose_mean, pose_std)

    wandb.config.update({"q_mean": q_mean.squeeze().cpu().tolist(), "q_std": q_std.squeeze().cpu().tolist(),
                         "pose_mean": pose_mean.squeeze().cpu().tolist(), "pose_std": pose_std.squeeze().cpu().tolist()})

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.monotonic()
        for batch in train_loader:
            q = batch["q"].to(device)
            pose = batch["pose"].to(device)
            noise = torch.randn_like(q)
            t = torch.randint(0, model.num_timesteps, (q.size(0),), device=device)
            beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
            q_t = q + beta * noise
            noise_pred = model(q_t, pose, t)
            loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_results = validate(model, val_loader, device, robot_choice, q_stats, pose_stats)
        X_errors = val_results["X_errors_report"]
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:])
        avg_position_error = X_errors_r[1,:3].mean()
        avg_orientation_error = X_errors_r[1,3:].mean()
        train_metrics = {"train/train_loss": train_loss}
        val_metrics = {
            "val/val_loss": val_loss,
            "val/xyz(mm)": avg_position_error,
            "val/RPY(deg)": avg_orientation_error
        }
        wandb.log({**train_metrics, **val_metrics})
        pose_loss = (avg_position_error + avg_position_error)/2
        if pose_loss < best_pose_loss:
            best_pose_loss = pose_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path+f'/best_epoch_{best_epoch}.pth')
            artifact = wandb.Artifact(name=f"MLP_{robot_choice}_Data_2.5M_Bs_128_Opt_AdamW", type='model')
            artifact.add_file(save_path+f'/best_epoch_{best_epoch}.pth')
            run.log_artifact(artifact)
        if epoch % (max_epochs/print_steps) == 0 or epoch == max_epochs-1:
            print(f"\n[Epoch {epoch+1}/{max_epochs}]")
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | xyz(mm): {avg_position_error:.2f} | RPY(deg): {avg_orientation_error:.2f} | Best Epoch {best_epoch}")
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
            end_training_time = time.monotonic()
            train_mins, train_secs = epoch_time(start_training_time, end_training_time)
            print(f'Been Training for: {train_mins}m {train_secs}s')

    wandb.finish()

# --- Main ---
if __name__ == "__main__":
    batch_size = 512
    max_epochs = 1000
    dof = 7
    pose_dim = 6
    robot_name = "panda"
    seed_choice, seed_number = True, 0
    save_on_wand = True
    if seed_choice:   
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.backends.cudnn.deterministic = True

    """
    #dataset_path = f"/home/miksolver/ik_datasets/{robot_name}"
    dataset_path = f"ik_datasets/{robot_name}"
    train_dataset = DiffIKDataset(
        os.path.join(dataset_path, "endpoints_tr.npy"),
        os.path.join(dataset_path, "samples_tr.npy")
    )
    val_dataset = DiffIKDataset(
        os.path.join(dataset_path, "endpoints_te.npy"),
        os.path.join(dataset_path, "samples_te.npy")
    )
    """

    file_path = "../for_docker/left-out-datasets/7DoF-Combined/review_data_7DoF-7R-PA10_1000000_qlim_scale_10_seq_1.csv"  # <-- Change this to your actual CSV file path
    df = pd.read_csv(file_path)
    data = df.to_numpy(dtype=np.float32)
    train_data, val_data = train_test_split(data, test_size=0.001, random_state=2324)
    train_data, val_data = train_data[:,:pose_dim+dof], val_data[:,:pose_dim+dof]
    train_dataset = DiffIKDataset2(train_data[:,:pose_dim], train_data[:,pose_dim:])
    val_dataset = DiffIKDataset2(val_data[:,:pose_dim], val_data[:,pose_dim:])


    #val_indices = np.random.choice(len(val_dataset), size=2000, replace=False)
    #val_subset = Subset(val_dataset, val_indices)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim)
    print("Model Trainable Parameters: {}".format(count_parameters(model)))
    train_loop(model, train_loader, val_loader, max_epochs=max_epochs, robot_name=robot_name, save_on_wand=save_on_wand)
