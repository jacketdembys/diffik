import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import sys
import time
import random
import wandb
from utils import get_robot_choice, reconstruct_pose_modified, epoch_time, count_parameters


"""
Tacking changes:
1.  Base Model: Increase Architecture Size
"""


# --- Dataset Loader ---
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        self.q = torch.from_numpy(np.load(filename_Q)).float()
        self.pose = torch.from_numpy(np.load(filename_D)).float()
        assert self.q.shape[0] == self.pose.shape[0]

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {"q": self.q[idx], "pose": self.pose[idx]}

# --- Diffusion MLP Architecture ---
class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=1024, time_embed_dim=128):
        super().__init__()
        self.dof = dof
        self.num_timesteps = 1000
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        input_dim = dof + pose_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, dof)
        )

    def forward(self, q_t, pose, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x = torch.cat([q_t, pose, t_embed], dim=-1)
        return self.net(x)

# --- Sampling Function ---
@torch.no_grad()
def sample(model, pose, ddim_steps=100):
    model.eval()
    B = pose.shape[0]
    q = torch.randn(B, model.dof).to(pose.device)
    pose = pose.repeat_interleave(1, dim=0)
    for t in reversed(range(1, ddim_steps + 1)):
        t_tensor = torch.full((q.size(0),), t, device=q.device, dtype=torch.long)
        #beta = 1e-4 + (t_tensor / model.num_timesteps).unsqueeze(-1)
        beta = 1e-4 + (0.02 - 1e-4) * (t_tensor / model.num_timesteps).unsqueeze(-1)
        noise_pred = model(q, pose, t_tensor)
        q = q - beta * noise_pred
    return q

# --- Validation ---
def validate(model, val_loader, device, robot_choice):
    model.eval()
    total_loss = 0.0
    loss_fn = nn.MSELoss()

    y_preds = []
    y_desireds = []

    with torch.no_grad():
        for batch in val_loader:
            q_gt = batch["q"].to(device)
            pose_gt = batch["pose"].to(device)

            # compute denoising loss
            noise = torch.randn_like(q_gt)
            t = torch.randint(0, model.num_timesteps, (q_gt.size(0),), device=device)
            #beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
            beta = 1e-4 + (0.02 - 1e-4) * (t / model.num_timesteps).unsqueeze(-1)
            q_t = q_gt + beta * noise
            noise_pred = model(q_t, pose_gt, t)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()            

            # compute prediction and FK error
            q_pred = sample(model, pose_gt)
            q_pred = q_pred.detach().cpu().numpy()
            q_gt = q_gt.detach().cpu().numpy()

            y_preds.append(q_pred)
            y_desireds.append(q_gt)

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)  # [NEW] Learning rate scheduler
    loss_fn = nn.MSELoss()

    robot_choice = get_robot_choice(robot_name)

    if save_on_wand:
        run = wandb.init(
            entity="jacketdembys",
            project = "diffik",
            group = f"MLP_{robot_choice}_Data_2.5M",
            name = f"MLP_{robot_choice}_Data_2.5M_Bs_128_Opt_AdamW"
        )

    best_pose_loss = float('inf')
    best_epoch = 0
    start_training_time = time.monotonic()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # [NEW] Gradient clipping
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        scheduler.step()  # [NEW] Step the scheduler
        val_loss, val_results = validate(model, val_loader, device, robot_choice)

        X_errors = val_results["X_errors_report"]
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:]) 
        avg_position_error = X_errors_r[1,:3].mean()    
        min_position_error = X_errors_r[1,:3].mean()  
        max_position_error = X_errors_r[1,:3].mean()      
        avg_orientation_error = X_errors_r[1,3:].mean()

        train_metrics = {
            "train/train_loss": train_loss,
        }

        val_metrics = {
            "val/val_loss": val_loss,
            "val/xyz(mm)": avg_position_error,
            "val/RPY(deg)": avg_orientation_error
        }
        wandb.log({**train_metrics, **val_metrics})

        pose_loss = (avg_position_error + avg_orientation_error) / 2  # [FIXED]
        if pose_loss < best_pose_loss:
            best_pose_loss = pose_loss
            best_epoch = epoch

            model_filename = f'best_epoch_{best_epoch}_pose_{pose_loss:.2f}.pth'  # [NEW] Better filename
            torch.save(model.state_dict(), os.path.join(save_path, model_filename))
            artifact = wandb.Artifact(name=f"MLP_{robot_choice}_Data_2.5M_Bs_128_Opt_AdamW", type='model')
            artifact.add_file(os.path.join(save_path, model_filename))
            run.log_artifact(artifact)

        if epoch % (max_epochs / print_steps) == 0 or epoch == max_epochs - 1:
            print(f"\n[Epoch {epoch}/{max_epochs}]")
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | xyz(mm): {avg_position_error:.2f} | RPY(deg): {avg_orientation_error:.2f} | Best Epoch {best_epoch}")

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')

            end_training_time = time.monotonic()
            train_mins, train_secs = epoch_time(start_training_time, end_training_time)
            print(f'Been Training for: {train_mins}m {train_secs}s')

    wandb.run.summary["best_pose_loss"] = best_pose_loss  # [NEW] Final summary
    wandb.run.summary["best_epoch"] = best_epoch  # [NEW] Final summary
    wandb.finish()

# --- Main ---
if __name__ == "__main__":
    batch_size = 128
    max_epochs = 1000
    dof = 7
    pose_dim = 7
    robot_name = "panda"
    seed_choice, seed_number = True, 0
    save_on_wand = True

    if seed_choice:   
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)
        torch.cuda.manual_seed(seed_number)
        torch.backends.cudnn.deterministic = True

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
    val_indices = np.random.choice(len(val_dataset), size=1000, replace=False)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim)
    print("Model Trainable Parameters: {}".format(count_parameters(model)))
    train_loop(model, train_loader, val_loader, max_epochs=max_epochs, robot_name=robot_name, save_on_wand=save_on_wand)
