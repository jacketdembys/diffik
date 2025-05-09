import time
import torch
import os
import sys
import numpy as np
import torch.nn as nn
import random
import pandas as pd
import wandb
from tqdm import trange
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import r2_score
from typing import Union, Callable, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import get_robot_choice, reconstruct_pose_modified, epoch_time, count_parameters
from CARD.regression.model import (
    DeterministicFeedForwardNeuralNetwork,
    ConditionalLinear,
)
from CARD.regression.diffusion_utils import (
    make_beta_schedule,
    q_sample,
    p_sample_loop,
    p_sample,
)



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
# --- 2. CARD-based architecutres -------------
# ---------------------------------------------------------
class ConditionalGuidedModel(nn.Module):
    def __init__(
        self,
        n_steps: int,
        cat_x: bool,
        cat_y_pred: bool,
        x_dim: int,
        y_dim: int,
        z_dim: int,
    ):
        super(ConditionalGuidedModel, self).__init__()
        self.cat_x = cat_x
        self.cat_y_pred = cat_y_pred
        data_dim = y_dim
        if self.cat_x:
            data_dim += x_dim
        if self.cat_y_pred:
            data_dim += y_dim
        self.lin1 = ConditionalLinear(data_dim, 1024, n_steps) # 128
        self.lin2 = ConditionalLinear(1024, 1024, n_steps)
        self.lin3 = ConditionalLinear(1024, 1024, n_steps)
        self.lin4 = nn.Linear(1024, 7)

    def forward(self, x, y_t, y_0_hat, t):
        if self.cat_x:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat, x), dim=1)
            else:
                eps_pred = torch.cat((y_t, x), dim=1)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=1)
            else:
                eps_pred = y_t
        eps_pred = F.softplus(self.lin1(eps_pred, t))
        eps_pred = F.softplus(self.lin2(eps_pred, t))
        eps_pred = F.softplus(self.lin3(eps_pred, t))
        return self.lin4(eps_pred)


# ---------------------------------------------------------
# --- 4. Training & Validation Loops ----------------------
# ---------------------------------------------------------
def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


def p_sample(
    x, y, y_0_hat, y_T_mean, t: int, alphas, one_minus_alphas_bar_sqrt, device, guidance_model
    ):
    z = torch.randn_like(y)
    t = torch.tensor([t]).to(device)
    alpha_t = extract(alphas, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, t - 1, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
    # y_t_m_1 posterior mean component coefficients
    gamma_0 = (
        (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_1 = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        * (alpha_t.sqrt())
        / (sqrt_one_minus_alpha_bar_t.square())
    )
    gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
        sqrt_one_minus_alpha_bar_t.square()
    )
    eps_theta = guidance_model(x, y, y_0_hat, t).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y + gamma_2 * y_T_mean

    beta_t_hat = (
        (sqrt_one_minus_alpha_bar_t_m_1.square())
        / (sqrt_one_minus_alpha_bar_t.square())
        * (1 - alpha_t)
    )
    y_t_m_1 = y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device)
    return y_t_m_1



def p_sample_loop(
    x,
    y_0_hat,
    y_T_mean,
    n_steps,
    alphas,
    one_minus_alphas_bar_sqrt,
    only_last_sample,
    device,
    guidance_model,
):
    num_t, y_p_seq = None, None
    z = torch.randn_like(y_T_mean).to(device)
    cur_y = z + y_T_mean  # sampled y_T
    if only_last_sample:
        num_t = 1
    else:
        y_p_seq = [cur_y]
    for t in reversed(range(1, n_steps)):
        y_t = cur_y
        cur_y = p_sample(
            x,
            y_t,
            y_0_hat,
            y_T_mean,
            t,
            alphas,
            one_minus_alphas_bar_sqrt,
            device,
            guidance_model,
        )  # y_{t-1}
        if only_last_sample:
            num_t += 1
        else:
            y_p_seq.append(cur_y)
    if only_last_sample:
        assert num_t == n_steps
        y_0 = p_sample_t_1to0(
            x, cur_y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, device, guidance_model
        )
        return y_0
    else:
        assert len(y_p_seq) == n_steps
        y_0 = p_sample_t_1to0(
            x, y_p_seq[-1], y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, device, guidance_model
        )
        y_p_seq.append(y_0)
        return y_p_seq



def p_sample_t_1to0(x, y, y_0_hat, y_T_mean, one_minus_alphas_bar_sqrt, device, guidance_model):
    # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    t = torch.tensor([0]).to(device)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    eps_theta = guidance_model(x, y, y_0_hat, t).detach()
    # y_0 reparameterization
    y_0_reparam = (
        1
        / sqrt_alpha_bar_t
        * (
            y
            - (1 - sqrt_alpha_bar_t) * y_T_mean
            - eps_theta * sqrt_one_minus_alpha_bar_t
        )
    )
    y_t_m_1 = y_0_reparam.to(device)
    return y_t_m_1



def validate_pretrain(
                    cond_pred_model, 
                    val_loader, 
                    device, 
                    robot_choice, 
                    q_stats, 
                    pose_stats,
                    ):
    
    cond_pred_model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    y_preds = []
    y_desireds = []
    q_mean, q_std = q_stats
    pose_mean, pose_std = pose_stats

    with torch.no_grad():
        for batch in val_loader:
            y_gt = batch['q'].to(device)
            pose = batch['pose'].to(device)
            y_pred = cond_pred_model(pose)
            loss = loss_fn(y_pred,y_gt)
            total_loss += loss.item()

            q_pred = y_pred * q_std + q_mean
            q_gt = y_gt * q_std + q_mean
            y_preds.append(q_pred.detach().cpu().numpy())
            y_desireds.append(q_gt.detach().cpu().numpy())

        val_loss = total_loss / len(val_loader)
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
    return val_loss, results



def validate_train(
                    cond_pred_model, 
                    diff_model,
                    val_loader, 
                    device, 
                    robot_choice, 
                    q_stats, 
                    pose_stats,
                    n_z_samples,
                    alphas,
                    one_minus_alphas_bar_sqrt,
                    n_steps,
                   ):
    cond_pred_model.to(device)
    diff_model.to(device)
    cond_pred_model.eval()
    diff_model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    y_preds = []
    y_desireds = []
    q_mean, q_std = q_stats
    pose_mean, pose_std = pose_stats

    with torch.no_grad():
        for batch in val_loader:
            y_gt = batch['q'].to(device)
            pose = batch['pose'].to(device)
            y_0_hat = cond_pred_model(pose)

            y_0_hat_tile = torch.tile(y_0_hat, (n_z_samples, 1)).to(device)
            test_x_tile = torch.tile(pose, (n_z_samples, 1)).to(device)

            z = torch.randn_like(y_0_hat_tile).to(device)

            x = pose
            y_t = y_0_hat_tile + z

            # generate samples from all time steps for the mini-batch
            y_tile_seq = p_sample_loop(
                test_x_tile,
                y_0_hat_tile,
                y_0_hat_tile,
                n_steps,
                alphas.to(device),
                one_minus_alphas_bar_sqrt.to(device),
                False,
                device,
                diff_model,
            )

            # put in shape [n_z_samples, batch_size, output_dimension]
            y_tile_seq = [
                arr.reshape(n_z_samples, x.shape[0], y_t.shape[-1]) for arr in y_tile_seq
            ]

            y_pred = y_tile_seq[-1]
            y_pred = y_pred.mean(dim=0)
            loss = loss_fn(y_pred,y_gt)
            total_loss += loss.item()

            q_pred = y_pred * q_std + q_mean
            q_gt = y_gt * q_std + q_mean
            y_preds.append(q_pred.detach().cpu().numpy())
            y_desireds.append(q_gt.detach().cpu().numpy())

            #mean_pred = final_recoverd.mean(dim=0).detach().cpu().squeeze()
            #std_pred = final_recoverd.std(dim=0).detach().cpu().squeeze()

            """
            return {
                    "pred": mean_pred,
                    "pred_uct": std_pred,
                    "aleatoric_uct": std_pred,
                    "samples": y_tile_seq,
                }
            """


        val_loss = total_loss / len(val_loader)
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

    return val_loss, results




def train_loop(
        cond_pred_model, 
        diff_model,
        train_loader, 
        val_loader, 
        q_stats, 
        pose_stats, 
        device, 
        n_steps,
        max_pretrain_epochs=1000,
        max_train_epochs=1000,  
        pretrain_lr=1e-2, 
        train_lr=1e-3, 
        robot_name="panda", 
        save_on_wand=True, 
        print_steps=100
    ):
    save_path = "results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f"[Results saved in: {save_path}]") 
    print(f"[Training on device: {device}]")

    robot_choice = get_robot_choice(robot_name)

    if save_on_wand:
        run = wandb.init(
            entity="jacketdembys",
            project="diffik",
            group=f"CARD_MLP_{robot_choice}_Data_1M",
            name=f"CARD_MLP_{robot_choice}_Data_1M_BS_128_Opt_AdamW_LR_3e_4_1e_8"
        )

    # Pretrain conditional
    print(f"==> [Start Pretraining Conditional Model]")
    cond_pred_model.to(device)
    aux_opt = torch.optim.Adam(cond_pred_model.parameters(), lr=pretrain_lr)
    aux_loss_fn = nn.MSELoss()
    best_pose_loss = float('inf')
    best_epoch = 0
    
    scheduler_pre = torch.optim.lr_scheduler.ReduceLROnPlateau(aux_opt,mode='min', factor=0.5, patience=10, min_lr=1e-10, verbose=True)
   
    start_training_time = time.monotonic()
    for epoch in range(max_pretrain_epochs):
        cond_pred_model.train()
        epoch_loss = 0.0
        start_time = time.monotonic()
        for batch in train_loader:
            y_gt = batch['q'].to(device)
            pose = batch['pose'].to(device)
            y_pred = cond_pred_model(pose)
            aux_loss = aux_loss_fn(y_pred,y_gt)

            aux_opt.zero_grad()
            aux_loss.backward()
            aux_opt.step()

            epoch_loss += aux_loss.item()
            #current_lr = pretrain_lr

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_results = validate_pretrain(                    
                                                cond_pred_model, 
                                                val_loader, 
                                                device, 
                                                robot_choice, 
                                                q_stats, 
                                                pose_stats,
                                            )


        
        X_errors = val_results["X_errors_report"]
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:])
        avg_position_error = X_errors_r[1,:3].mean()
        avg_orientation_error = X_errors_r[1,3:].mean()

        pose_loss = (avg_position_error + avg_orientation_error)/2
        scheduler_pre.step(pose_loss)
        current_lr = aux_opt.param_groups[0]['lr']


        train_metrics = {
            "ptrain/train_loss": train_loss,
            "ptrain/lr": current_lr
            }
        val_metrics = {
            "pval/val_loss": val_loss,
            "pval/xyz(mm)": avg_position_error,
            "pval/RPY(deg)": avg_orientation_error
        }
        wandb.log({**train_metrics, **val_metrics})
        

        if pose_loss < best_pose_loss:
            best_pose_loss = pose_loss
            best_epoch = epoch
            torch.save(cond_pred_model.state_dict(), save_path+f'/best_epoch_{best_epoch}_pred_model.pth')
            artifact = wandb.Artifact(name=f"MLP_{robot_choice}_Data_1M_Bs_128_Opt_AdamW_LR_3e_4_1e_8_cond_pred_model", type='model')
            artifact.add_file(save_path+f'/best_epoch_{best_epoch}_pred_model.pth')
            run.log_artifact(artifact)
        if epoch % (max_pretrain_epochs/print_steps) == 0 or epoch == max_pretrain_epochs-1:
            print(f"\n[Epoch {epoch+1}/{max_pretrain_epochs}]")
            print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | xyz(mm): {avg_position_error:.2f} | RPY(deg): {avg_orientation_error:.2f} | Best Epoch: {best_epoch}")
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
            end_training_time = time.monotonic()
            train_mins, train_secs = epoch_time(start_training_time, end_training_time)
            print(f'Been Training for: {train_mins}m {train_secs}s')

    print(f"\n[End Pretraining Conditional Model]")


    
    # Train diffusion model
    beta_schedule = "linear"
    beta_start = 0.0001
    beta_end = 0.02
    n_z_samples = 100
    betas = make_beta_schedule(beta_schedule, n_steps, beta_start, beta_end).to(device)
    betas_sqrt = torch.sqrt(betas)
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)

    print(f"\n\n==> [Start Training Diffusion Model]")
    diff_model.to(device)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=train_lr)
    diff_loss_fn = nn.MSELoss(reduction="mean")
    best_pose_loss = float('inf')
    best_epoch = 0

    scheduler_diff = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-10, verbose=True)

    start_diff_time = time.monotonic()

    for epoch in range(max_train_epochs):
        diff_model.train()
        epoch_loss = 0.0
        epoch_aux_loss = 0.0
        start_time = time.monotonic()
        for batch in train_loader:
            y_gt = batch['q'].to(device)
            pose = batch['pose'].to(device)
            batch_size = y_gt.shape[0]

            # antithetic sampling
            ant_samples_t = torch.randint(
            low=0, high=n_steps, size=(batch_size // 2 + 1,)
            ).to(device)

            ant_samples_t = torch.cat([ant_samples_t, n_steps - 1 - ant_samples_t], dim=0)[:batch_size]

            # noise estimation loss
            y_0_hat = cond_pred_model(pose)
            e = torch.randn_like(y_gt)
            y_t_sample = q_sample(
                                y_gt,
                                y_0_hat,
                                alphas_bar_sqrt,
                                one_minus_alphas_bar_sqrt,
                                ant_samples_t,
                                noise=e,
                            )
            
            y_t_pred = diff_model(pose, y_t_sample, y_0_hat, ant_samples_t)

            # use the same noise sample e during training to compute loss
            loss = diff_loss_fn(e, y_t_pred)

            # Optimize the diffusion mode that predicts the eps_theta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optmize the non-linear guidance model
            aux_loss = aux_loss_fn(cond_pred_model(pose), y_gt)
            aux_opt.zero_grad()
            aux_loss.backward()
            aux_opt.step()

            epoch_loss += loss.item()
            epoch_aux_loss += aux_loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_aux_loss = epoch_aux_loss / len(train_loader)

        # Evaluate the diffusion model
        val_loss, val_results = validate_train(                  
                                                cond_pred_model, 
                                                diff_model,
                                                val_loader, 
                                                device, 
                                                robot_choice, 
                                                q_stats, 
                                                pose_stats,
                                                n_z_samples,
                                                alphas,
                                                one_minus_alphas_bar_sqrt,
                                                n_steps,
                                                )
        



        X_errors = val_results["X_errors_report"]
        X_errors_r = X_errors[:,:6]
        X_errors_r[:,:3] = X_errors_r[:,:3] * 1000
        X_errors_r[:,3:] = np.rad2deg(X_errors_r[:,3:])
        avg_position_error = X_errors_r[1,:3].mean()
        avg_orientation_error = X_errors_r[1,3:].mean()

        pose_loss = (avg_position_error + avg_orientation_error)/2        
        scheduler_diff.step(pose_loss) # the scheduler of lr
        current_lr = optimizer.param_groups[0]['lr']


        train_metrics = {
            "train/train_loss": train_loss,
            "train/train_aux_loss": train_aux_loss,
            "train/lr": current_lr
            }
        val_metrics = {
            "val/val_loss": val_loss,
            "val/xyz(mm)": avg_position_error,
            "val/RPY(deg)": avg_orientation_error
        }
        wandb.log({**train_metrics, **val_metrics})


        if pose_loss < best_pose_loss:
            best_pose_loss = pose_loss
            best_epoch = epoch
            torch.save(diff_model.state_dict(), save_path+f'/best_epoch_{best_epoch}_diff_model.pth')
            artifact = wandb.Artifact(name=f"MLP_{robot_choice}_Data_1M_Bs_128_Opt_AdamW_LR_3e_4_1e_8_diff_model", type='model')
            artifact.add_file(save_path+f'/best_epoch_{best_epoch}_diff_model.pth')
            run.log_artifact(artifact)
        if epoch % (max_train_epochs/print_steps) == 0 or epoch == max_train_epochs-1:
            print(f"\n[Epoch {epoch+1}/{max_train_epochs}]")
            print(f"Train Loss: {train_loss:.6f} | Train Aux-Loss: {train_aux_loss:.6f} | Val Loss: {val_loss:.6f} | xyz(mm): {avg_position_error:.2f} | RPY(deg): {avg_orientation_error:.2f} | Best Epoch: {best_epoch}")
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
            end_diff_time = time.monotonic()
            train_mins, train_secs = epoch_time(start_diff_time, end_diff_time)
            print(f'Been Training for: {train_mins}m {train_secs}s')

    print(f"\n[End Diffusion Model]")
    


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


    # Parameters 
    n_steps = 1000
    cat_x = True
    cat_y_pred = True
    x_dim = 6
    y_dim = 7
    z_dim = None
    hid_layers = [1024,1024,1024]

    # Build the conditional prediction model
    cond_pred_model = DeterministicFeedForwardNeuralNetwork(
        dim_in=x_dim,
        dim_out=y_dim,
        hid_layers=hid_layers
    )
    print(f"Conditional Prediction Model: \n{cond_pred_model}")


    # Build diffusion model
    diff_model = ConditionalGuidedModel(
        n_steps=n_steps,
        cat_x=cat_x,
        cat_y_pred=cat_y_pred,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim
    )
    print(f"Diffusion Model: \n{diff_model}")


    # train the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_stats = (train_ds.q_mean.to(device), train_ds.q_std.to(device))
    pose_stats = (train_ds.pose_mean.to(device), train_ds.pose_std.to(device))
    train_loop(
        cond_pred_model, 
        diff_model,
        train_loader, 
        val_loader, 
        q_stats, 
        pose_stats, 
        device, 
        n_steps,
        max_pretrain_epochs=1000,
        max_train_epochs=5000,  
        pretrain_lr=3e-4, 
        train_lr=3e-4, 
        robot_name="panda", 
        save_on_wand=True, 
        print_steps=100
    ) 



    