import os, sys
import math
import time
import random
import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from utils import get_robot_choice, reconstruct_pose_modified, epoch_time, count_parameters


# ---------------------------------------------------------
# --- 1. Dataset with global mean/std built in -------------
# ---------------------------------------------------------
class DiffIKDataset(Dataset):
    def __init__(self, D, Q):
        # D: [N,pose_dim], Q: [N,dof]
        self.pose_raw = torch.from_numpy(D).float()
        self.q_raw    = torch.from_numpy(Q).float()
        assert self.pose_raw.shape[0] == self.q_raw.shape[0]

        # compute dataset‐wide stats once
        self.pose_mean = self.pose_raw.mean(dim=0, keepdim=True)
        self.pose_std  = self.pose_raw.std(dim=0,  keepdim=True) + 1e-8
        self.q_mean    = self.q_raw.mean(dim=0,  keepdim=True)
        self.q_std     = self.q_raw.std(dim=0,   keepdim=True) + 1e-8

        # normalize permanently
        self.pose = (self.pose_raw - self.pose_mean) / self.pose_std
        self.q    = (self.q_raw    - self.q_mean)  / self.q_std

    def __len__(self):
        return self.q.size(0)

    def __getitem__(self, idx):
        return {
            'pose': self.pose[idx],
            'q':    self.q[idx]
        }

# ---------------------------------------------------------
# --- 2.1 MLP-Denoiser with built‐in diffusion schedule ----------
# ---------------------------------------------------------
class MLPDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=6, hidden_dim=1024,
                 time_embed_dim=128, pose_embed_dim=128, T=1000):
        super().__init__()
        self.dof = dof
        self.T   = T

        # 2.1 sinusoidal time embedding
        def sinusoidal_embedding(t, dim):
            half = dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
            )[None, :]  # [1, half]
            args = t.unsqueeze(-1).float() * freqs  # [B, half]
            return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]

        self.time_embed_dim = time_embed_dim
        self.pose_embed_dim = pose_embed_dim

        # 2.2 small MLP to embed pose
        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.SiLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )

        # 2.3 main network
        input_dim = dof + pose_embed_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dof),
        )

        # 2.4 build classical DDPM schedule buffers
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas',      betas)
        self.register_buffer('alphas',     alphas)
        self.register_buffer('alpha_bar',  alpha_bar)

    def forward(self, q_t, pose, t):
        """
        q_t: [B, dof], pose: [B, pose_dim], t: [B] longs in [0, T-1]
        returns predicted noise [B, dof]
        """
        # 1) time embedding
        te = self._sinusoidal_time_embedding(t, self.time_embed_dim).to(q_t.device)
        pe = self.pose_mlp(pose)  # [B, pose_embed_dim]
        x = torch.cat([q_t, pe, te], dim=-1)
        return self.net(x)

    def _sinusoidal_time_embedding(self, t, dim):
        # reuse local function
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )[None, :]
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------
# --- 2.2 ResMLP-Denoiser with built‐in diffusion schedule ----------
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1    = nn.Linear(dim, dim)
        self.act    = nn.SiLU()
        self.fc2    = nn.Linear(dim, dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        # x : [B, dim]
        res = self.fc2(self.act(self.fc1(x)))
        res = self.drop(res)
        return x + res


class ResMLPDenoiser(nn.Module):
    def __init__(
        self,
        dof: int = 7,
        pose_dim: int = 6,
        hidden_dim: int = 1024,
        time_embed_dim: int = 128,
        pose_embed_dim: int = 128,
        num_blocks: int = 6,
        dropout: float = 0.1,
        T: int = 1000
    ):
        super().__init__()
        self.dof = dof
        self.T   = T

        # --- time embedding (sinusoidal) ---
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- pose embedding MLP ---
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.SiLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )

        # --- input projection to hidden size ---
        input_dim = dof + pose_embed_dim + time_embed_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- stack of residual MLP blocks ---
        self.resblocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # --- final output layer back to dof dims ---
        self.output_proj = nn.Linear(hidden_dim, dof)

        # --- diffusion schedule buffers (unchanged) ---
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',     betas)
        self.register_buffer('alphas',    alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    def forward(self, q_t, pose, t):
        """
        q_t:   [B, dof]        noisy joint vectors
        pose:  [B, pose_dim]   (normalized) target pose
        t:     [B]             timestep indices in [0..T-1]
        returns predicted noise [B, dof]
        """
        # 1) sinusoidal time embedding
        te = self._sinusoidal_time_embedding(t, self.time_embed_dim).to(q_t.device)
        te = self.time_mlp(te)

        # 2) pose embedding
        pe = self.pose_embed(pose)  # [B, pose_embed_dim]

        # 3) concat all inputs and project
        x = torch.cat([q_t, pe, te], dim=-1)  # [B, input_dim]
        x = self.input_proj(x)                # [B, hidden_dim]

        # 4) residual blocks
        for block in self.resblocks:
            x = block(x)

        # 5) final projection to noise prediction
        return self.output_proj(x)

    def _sinusoidal_time_embedding(self, t, dim):
        # same as before
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )[None, :]
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------
# --- 2.3 ResMLP-Denoiser with built‐in diffusion schedule ----------
# ---------------------------------------------------------
class MLPUNetDenoiser(nn.Module):
    def __init__(
        self,
        dof: int = 7,
        pose_dim: int = 6,
        hidden_dim: int = 512,
        time_embed_dim: int = 128,
        pose_embed_dim: int = 128,
        dropout: float = 0.1,
        T: int = 1000
    ):
        super().__init__()
        self.dof = dof
        self.T   = T

        # 1) Time embedding
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 2) Pose embedding
        self.pose_embed = nn.Sequential(
            nn.Linear(pose_dim, pose_embed_dim),
            nn.SiLU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )

        # 3) Input projection to hidden_dim
        input_dim = dof + pose_embed_dim + time_embed_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 4) Encoder (contracting path)
        self.enc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.enc2 = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 5) Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 6) Decoder (expanding path) with skips
        self.dec2 = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.dec1 = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # 7) Output projection back to dof
        self.output_proj = nn.Linear(hidden_dim, dof)

        # 8) Diffusion schedule buffers
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',      betas)
        self.register_buffer('alphas',     alphas)
        self.register_buffer('alpha_bar',  alpha_bar)

    def forward(self, q_t, pose, t):
        """
        q_t:   [B, dof]      noisy joint sample
        pose:  [B, pose_dim] normalized target pose
        t:     [B]           timestep indices in [0..T-1]
        """
        # time embedding
        te = self._sinusoidal_time_embedding(t, self.time_embed_dim).to(q_t.device)
        te = self.time_mlp(te)

        # pose embedding
        pe = self.pose_embed(pose)

        # input → hidden
        x = torch.cat([q_t, pe, te], dim=-1)
        x0 = self.input_proj(x)          # [B, hidden_dim]

        # encoder
        e1 = self.enc1(x0)               # [B, hidden_dim/2]
        e2 = self.enc2(e1)               # [B, hidden_dim/4]

        # bottleneck
        b = self.bottleneck(e2)          # [B, hidden_dim/4]

        # decoder with skip from e2 → dec2
        d2 = self.dec2(b) + e1           # up to hidden_dim/2
        d1 = self.dec1(d2) + x0          # up to hidden_dim

        # final output
        return self.output_proj(d1)      # [B, dof]

    def _sinusoidal_time_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )[None, :]
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)



# ---------------------------------------------------------
# --- 3. True DDPM sampling -------------------------------
# ---------------------------------------------------------
@torch.no_grad()
def sample(model, pose, q_mean, q_std, pose_mean, pose_std, device, ddim_steps=50):
    """
    model: trained DiffIKDenoiser
    pose: [B, pose_dim] raw pose (not normalized)
    returns q0 samples in raw joint space
    """
    model.eval()
    # normalize pose once
    pose_n = (pose - pose_mean) / pose_std
    B = pose.size(0)

    q = torch.randn(B, model.dof, device=device)  # start from pure noise

    for t_ in reversed(range(ddim_steps)):
        t = torch.full((B,), t_, device=device, dtype=torch.long)
        beta_t = model.betas[t]
        a_t    = model.alphas[t]
        ab_t   = model.alpha_bar[t]
        #ab_t1  = model.alpha_bar[t-1] if t_ > 0 else torch.ones_like(ab_t)

        # predict noise
        eps_pred = model(q, pose_n, t)

        # DDPM posterior mean
        #coef1 = 1.0 / torch.sqrt(a_t)
        #coef2 = beta_t / torch.sqrt(1.0 - ab_t)
        #mean_t1 = coef1 * (q - coef2.unsqueeze(-1) * eps_pred)

        coef1 = (1.0 / torch.sqrt(a_t)).unsqueeze(-1)               # [B,1]
        coef2 = (beta_t / torch.sqrt(1.0 - ab_t)).unsqueeze(-1)     # [B,1]
        mean_t1 = coef1 * (q - coef2 * eps_pred)

        

        # add noise if not final
        if t_ > 0:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(q)
            q = mean_t1 + sigma_t.unsqueeze(-1) * z
        else:
            q = mean_t1

    # de‐normalize
    return q * q_std + q_mean

# ---------------------------------------------------------
# --- 4. Training & Validation Loops ----------------------
# ---------------------------------------------------------
def validate(model, val_loader, device, robot_choice, q_stats, pose_stats):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    y_preds = []
    y_desireds = []
    q_mean, q_std = q_stats
    pose_mean, pose_std = pose_stats

    with torch.no_grad():
        for batch in val_loader:
            q0 = batch['q'].to(device)
            pose = batch['pose'].to(device)
            # sample a random t
            t = torch.randint(0, model.T, (q0.size(0),), device=device)
            # correct forward noising
            eps   = torch.randn_like(q0)
            ab_t  = model.alpha_bar[t].unsqueeze(-1)        # [B,1]
            sqrt_ab = torch.sqrt(ab_t)
            sqrt_omb = torch.sqrt(1 - ab_t)
            q_t = sqrt_ab * q0 + sqrt_omb * eps

            # predict noise
            eps_pred = model(q_t, pose, t)
            total += loss_fn(eps_pred, eps).item()


            # Reverse normalization
            q_gt_denorm = q0 * q_std + q_mean
            pose_gt_denorm = pose * pose_std + pose_mean
            q_pred = sample(model, pose_gt_denorm, q_mean, q_std, pose_mean, pose_std, device)
            q_pred = q_pred.detach().cpu().numpy()
            q_gt_denorm = q_gt_denorm.detach().cpu().numpy()
            y_preds.append(q_pred)
            y_desireds.append(q_gt_denorm)

        
        monitored_total_loss = total / len(val_loader)
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
        
    #return total / len(val_loader)
    return monitored_total_loss, results

def train_loop(model, train_loader, val_loader, q_stats, pose_stats, device, max_epochs=100, lr=1e-4, robot_name="panda", save_on_wand=True, print_steps=100):
    save_path = "results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"[Results saved in: {save_path}]") 

    
    print(f"[Training on device: {device}]")

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_pose_loss = float('inf')
    best_epoch = 0
    start_training_time = time.monotonic()

    robot_choice = get_robot_choice(robot_name)

    if save_on_wand:
        run = wandb.init(
            entity="jacketdembys",
            project="diffik",
            group=f"MLP_{robot_choice}_Data_2.5M",
            name=f"MLP_{robot_choice}_Data_1M_BS_128_Opt_AdamW_LR_3e_4_1e_8"
        )

    
     # 1) Compute total number of optimizer steps
    #steps_per_epoch = len(train_loader)
    #total_steps     = max_epochs * steps_per_epoch
    warmup_steps    = 5000   # first 5k steps will be warm‑up
    total_steps = 1500000

    # 2) Define the LR Lambda using those two values
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # linearly ramp up
            return float(current_step) / float(max(1, warmup_steps))
        # after warmup, do a cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # 3) Plug it into a LambdaLR scheduler
    #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min', factor=0.5, patience=10, min_lr=1e-8, verbose=True)


    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.monotonic()
        for batch in train_loader:
            global_step += 1
            q0 = batch['q'].to(device)
            pose = batch['pose'].to(device)
            # 1) sample t
            t = torch.randint(0, model.T, (q0.size(0),), device=device)
            # 2) forward noising
            eps   = torch.randn_like(q0)
            ab_t  = model.alpha_bar[t].unsqueeze(-1)
            sqrt_ab = torch.sqrt(ab_t)
            sqrt_omb = torch.sqrt(1 - ab_t)
            q_t = sqrt_ab * q0 + sqrt_omb * eps

            # 3) predict & loss
            eps_pred = model(q_t, pose, t)
            loss = loss_fn(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()
            #scheduler.step()
            epoch_loss += loss.item()

            #current_lr = scheduler.get_last_lr()[0]
            current_lr = opt.param_groups[0]['lr']
            #wandb.log({'train/lr': current_lr}, step=global_step)

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_results = validate(model, val_loader, device, robot_choice, q_stats, pose_stats)


        X_errors = val_results["X_errors_report"]
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:])
        avg_position_error = X_errors_r[1,:3].mean()
        avg_orientation_error = X_errors_r[1,3:].mean()
        train_metrics = {
            "train/train_loss": train_loss,
            "train/lr": current_lr
            }
        val_metrics = {
            "val/val_loss": val_loss,
            "val/xyz(mm)": avg_position_error,
            "val/RPY(deg)": avg_orientation_error
        }
        wandb.log({**train_metrics, **val_metrics})
        pose_loss = (avg_position_error + avg_position_error)/2
        scheduler.step(pose_loss)
        if pose_loss < best_pose_loss:
            best_pose_loss = pose_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path+f'/best_epoch_{best_epoch}.pth')
            artifact = wandb.Artifact(name=f"MLP_{robot_choice}_Data_2.5M_Bs_128_Opt_AdamW", type='model')
            artifact.add_file(save_path+f'/best_epoch_{best_epoch}.pth')
            run.log_artifact(artifact)
        if epoch % (max_epochs/print_steps) == 0 or epoch == max_epochs-1:
            print(f"\n[Epoch {epoch+1}/{max_epochs}]")
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | xyz(mm): {avg_position_error:.2f} | RPY(deg): {avg_orientation_error:.2f} | Best Epoch: {best_epoch}")
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
            end_training_time = time.monotonic()
            train_mins, train_secs = epoch_time(start_training_time, end_training_time)
            print(f'Been Training for: {train_mins}m {train_secs}s')

        """
        print(f"Epoch {epoch+1}/{max_epochs} — TrainLoss: {train_loss/len(train_loader):.6f}, ValLoss: {val_loss:.6f}")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_diffik.pth')
        """


    wandb.finish()


# ---------------------------------------------------------
# --- 5. Main: data prep & run ----------------------------
# ---------------------------------------------------------
if __name__ == "__main__":
    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # load CSV, split pose vs q
    #file_path = "../for_docker/left-out-datasets/7DoF-Combined/review_data_7DoF-7R-Panda_1000000_qlim_scale_10_seq_1.csv"  # <-- Change this to your actual CSV file path
    file_path = "/home/datasets/7DoF-Combined/review_data_7DoF-7R-Panda_1000000_qlim_scale_10_seq_1.csv"  # <-- Change this to your actual CSV file path
    df = pd.read_csv(file_path)
    pose_dim, dof = 6, 7
    data = df.to_numpy(dtype=np.float32)
    train_data, val_data = train_test_split(data, test_size=0.001, random_state=2324)
    train_data, val_data = train_data[:,:pose_dim+dof], val_data[:,:pose_dim+dof]

    #data, labels = df[:,:pose_dim], df[:,pose_dim:]
    train_D, val_D = train_data[:,:pose_dim], val_data[:, :pose_dim]
    train_Q, val_Q = train_data[:,pose_dim:], val_data[:, pose_dim:]

    train_ds = DiffIKDataset(train_D, train_Q)
    val_ds   = DiffIKDataset(val_D,   val_Q)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)

    # build model
    model = ResMLPDenoiser(dof=dof, 
                           pose_dim=pose_dim, 
                           T=1000)
        
    """
    model = ResMLPDenoiser(
        dof=dof,
        pose_dim=pose_dim,
        hidden_dim=512,      # or tweak up/down
        time_embed_dim=64,
        pose_embed_dim=64,
        num_blocks=3,         # depth of residual stacks
        dropout=0.1,
        T=1000
    )
    
    model = MLPUNetDenoiser(
        dof=dof,
        pose_dim=pose_dim,
        hidden_dim=512,        # tweak as needed
        time_embed_dim=128,
        pose_embed_dim=128,
        dropout=0.1,
        T=1000
    )
    """
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_stats = (train_ds.q_mean.to(device), train_ds.q_std.to(device))
    pose_stats = (train_ds.pose_mean.to(device), train_ds.pose_std.to(device))
    train_loop(model, train_loader, val_loader, q_stats, pose_stats, device, max_epochs=1000, lr=3e-4)


    """
    # example sampling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('best_diffik.pth', map_location=device))
    model.eval()

    # pick 16 random poses from val set
    val_pose_raw = val_ds.pose_raw[:16].to(device)
    samples = sample(
        model,
        val_pose_raw,
        train_ds.q_mean.to(device),
        train_ds.q_std.to(device),
        train_ds.pose_mean.to(device),
        train_ds.pose_std.to(device),
        device,
        ddim_steps=200
    )
    print("Sampled joint shapes:", samples.shape)
    """
