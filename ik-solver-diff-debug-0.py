import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# Dataset loader
class DiffIKDataset(Dataset):
    def __init__(self, filename_D, filename_Q):
        # Load .npy files using NumPy
        self.q = torch.from_numpy(np.load(filename_Q)).float() #.to(device)
        self.pose = torch.from_numpy(np.load(filename_D)).float() #.to(device)
        
        print(f"\nLoaded Dataset:\nq.shape: {self.q.shape}\npose.shape: {self.pose.shape}")
        print(f"Loaded Dataset initially on --> device: {self.q.device}")

        assert self.q.shape[0] == self.pose.shape[0], "Mismatch in sample count"

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, idx):
        return {
            "q": self.q[idx],
            "pose": self.pose[idx],
        }
    

# Diffusion Architecture
class DiffIKDenoiser(nn.Module):
    def __init__(self, dof=7, pose_dim=7, hidden_dim=512, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )
        input_dim = dof + pose_dim + time_embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, dof)
        )

    def forward(self, q_t, pose, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        x = torch.cat([q_t, pose, t_embed], dim=-1)
        return self.net(x)


# Training loop function
def train_loop(model, train_loader, val_loader, robot, max_epochs=1, lr=1e-4):
    print("Setting up training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val_loss = float('inf')

    print("Starting training loop...")

    for epoch in range(max_epochs):
        print(f"\n[Epoch {epoch}]")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            print(f"  Batch {batch_idx} - Loading data")
            try:
                q = batch["q"].to(device)
                pose = batch["pose"].to(device)
                print(f"    q.device: {q.device}, pose.device: {pose.device}")

                noise = torch.randn_like(q)
                print("    Noise generated")

                t = torch.randint(0, model.num_timesteps, (q.size(0),), device=q.device)
                print(f"    t.shape: {t.shape}, t.device: {t.device}")

                beta = 1e-4 + (t / model.num_timesteps).unsqueeze(-1)
                q_t = q + beta * noise

                noise_pred = model(q_t, pose, t)
                loss = loss_fn(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"    ✅ Loss: {loss.item():.6f}")
                #break  # only one batch to debug
            except Exception as e:
                print("🔥 ERROR DURING BATCH PROCESSING:", str(e))
                import traceback
                traceback.print_exc()
                return


if __name__ == "__main__":

    
    # --- Configurable Parameters ---
    batch_size = 128
    max_epochs = 5
    disable_wandb = False  # Set True if using Weights & Biases
    dof = 7
    pose_dim = 7

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = "panda"


    filename_Dtr = '/home/miksolver/ik_datasets/panda/endpoints_tr.npy'
    filename_Qtr = '/home/miksolver/ik_datasets/panda/samples_tr.npy'
    train_dataset = DiffIKDataset(filename_Dtr, filename_Qtr)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


    filename_Dte = '/home/miksolver/ik_datasets/panda/endpoints_te.npy'
    filename_Qte = '/home/miksolver/ik_datasets/panda/samples_te.npy'
    val_dataset = DiffIKDataset(filename_Dte, filename_Qte) 
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size)
    

    model = DiffIKDenoiser(dof=dof, pose_dim=pose_dim) #.to(device)
    model.dof = dof
    model.num_timesteps = 1000
    print(model)

    train_loop(model, 
                train_loader, 
                val_loader, 
                robot, 
                max_epochs=max_epochs
                )