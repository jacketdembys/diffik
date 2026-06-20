#!/bin/bash
# PHASE 2 — Model-size sweep at the BEST dataset size from Phase 1 (generate-jobs.sh).
# Vary denoiser capacity (hidden x layers -> #params) at a FIXED dataset.
# Run this ONLY after the dataset sweep is analyzed and you've chosen DS_N below.
#
# Usage:  bash cluster/generate-model-jobs.sh           (defaults to the best dataset)
#         DS_N=<n_trajectories> bash cluster/generate-model-jobs.sh   (override)
set -euo pipefail

export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export CONFIG="configs/panda_lbe_trajectory.yaml"
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"   # rotate if repo public
TEMPLATE="cluster/job-template-gpu.yaml"

# Best dataset from the Phase-1 scaling sweep: n_traj=6400 (204,800 train).
DS_N="${DS_N:-6400}"
SEED=0
MAXEPOCHS=1500
PATIENCE=15
SPECS="128x2 256x3 512x4 768x6 1024x4 1280x6"   # hidden x layers

launch () { export JOBNAME="$1"; export OVERRIDES="$2"; echo "launch ${JOBNAME}"; envsubst < "${TEMPLATE}" | kubectl apply -f -; }

for spec in ${SPECS}; do
  h="${spec%x*}"; l="${spec#*x}"
  launch "diffik-model-n${DS_N}-h${h}-l${l}" \
    "name=lbe_n${DS_N}_h${h}_l${l} seed=${SEED} wandb=true wandb_group=lbe_model_sweep_n${DS_N} \
     data.kind=trajectory data.lbe=true data.n_trajectories=${DS_N} data.steps_per_traj=40 data.v_deg=1.0 \
     model.type=lbe model.hidden_dim=${h} model.n_layers=${l} \
     diffusion.T=1000 diffusion.fk_loss_weight=10.0 diffusion.rot_weight=0.1 diffusion.p_example_dropout=0.2 \
     train.epochs=${MAXEPOCHS} train.patience=${PATIENCE} train.monitor_every=10 \
     eval.n_per_pose=1 eval.seeded=true"
done

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
