import os
import math
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

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
# --- 2. Denoiser with built‐in diffusion schedule ----------
# ---------------------------------------------------------
class DiffIKDenoiser(nn.Module):
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
        ab_t1  = model.alpha_bar[t-1] if t_ > 0 else torch.ones_like(ab_t)

        # predict noise
        eps_pred = model(q, pose_n, t)

        # DDPM posterior mean
        coef1 = 1.0 / torch.sqrt(a_t)
        coef2 = beta_t / torch.sqrt(1.0 - ab_t)
        mean_t1 = coef1 * (q - coef2.unsqueeze(-1) * eps_pred)

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
def validate(model, val_loader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
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
    return total / len(val_loader)

def train_loop(model, train_loader, val_loader, max_epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float('inf')

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
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
            loss = nn.MSELoss()(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{max_epochs} — TrainLoss: {train_loss/len(train_loader):.6f}, ValLoss: {val_loss:.6f}")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_diffik.pth')

# ---------------------------------------------------------
# --- 5. Main: data prep & run ----------------------------
# ---------------------------------------------------------
if __name__ == "__main__":
    # reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # load CSV, split pose vs q
    file_path = "../for_docker/left-out-datasets/7DoF-Combined/review_data_7DoF-7R-Panda_1000000_qlim_scale_10_seq_1.csv"  # <-- Change this to your actual CSV file path
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
    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim, T=1000)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    # train
    train_loop(model, train_loader, val_loader, max_epochs=200, lr=3e-4)


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
