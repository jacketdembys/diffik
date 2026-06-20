#!/bin/bash
# PHASE 1 — LBE dataset-size sweep as NRP Kubernetes Jobs (one Job per dataset size).
# Mirrors generate-gpu-jobs-for-ik.sh: loop the grid, envsubst the template, kubectl apply.
# Each Job: git clone -> pip install -e . -> wandb login -> scripts/train.py --override ...
# Results -> wandb. Best-of-K / multimodality are post-hoc evals.
#
# The MODEL-SIZE sweep is PHASE 2 (cluster/generate-model-jobs.sh) and runs on the
# BEST dataset size found here -- do not run it until this sweep is analyzed.
#
# Usage:  bash cluster/generate-jobs.sh
set -euo pipefail

# Existing DiffIK image in your gitlab registry (no rebuild needed).
export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export CONFIG="configs/panda_lbe_trajectory.yaml"   # full-scale base: T=1000, hidden=1024, L4
# wandb key (the Job runs `wandb login ${WANDB_KEY}`). Replace if you rotate it.
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"
TEMPLATE="cluster/job-template-gpu.yaml"

SEED=0
MAXEPOCHS=1500
PATIENCE=15

launch () {  # $1=jobname  $2=overrides
  export JOBNAME="$1"
  export OVERRIDES="$2"
  echo "launch ${JOBNAME}"
  envsubst < "${TEMPLATE}" | kubectl apply -f -
}

# ============================================================================
# (A) LBE dataset-size sweep — doubling n_trajectories (n_train = n_traj*40*0.8)
#     50..25600  ->  n_train 1,600 .. 819,200 (includes the 9th/10th cases)
# ============================================================================
for n in 50 100 200 400 800 1600 3200 6400 12800 25600; do
  launch "diffik-lbe-n${n}" \
    "name=lbe_n${n} seed=${SEED} wandb=true wandb_group=lbe_dataset_sweep \
     data.kind=trajectory data.lbe=true data.n_trajectories=${n} data.steps_per_traj=40 data.v_deg=1.0 \
     model.type=lbe model.hidden_dim=1024 model.n_layers=4 \
     diffusion.T=1000 diffusion.fk_loss_weight=10.0 diffusion.rot_weight=0.1 diffusion.p_example_dropout=0.2 \
     train.epochs=${MAXEPOCHS} train.patience=${PATIENCE} train.monitor_every=10 \
     eval.n_per_pose=1 eval.seeded=true"
done

# Model-size sweep is PHASE 2 -> cluster/generate-model-jobs.sh (run after picking best).

# ============================================================================
# OPTIONAL: seedless-trained MLP baseline (pure DIK-style). Uncomment to run.
# ============================================================================
# for n in 80000 320000 1000000; do
#   launch "diffik-seedless-n${n}" \
#     "name=seedless_n${n} seed=${SEED} wandb=true wandb_group=seedless_sweep \
#      data.kind=random data.lbe=false data.n_samples=${n} \
#      model.type=mlp model.hidden_dim=1024 model.n_layers=4 \
#      diffusion.T=1000 diffusion.fk_loss_weight=10.0 diffusion.rot_weight=0.1 \
#      train.epochs=${MAXEPOCHS} train.patience=${PATIENCE} train.monitor_every=10 \
#      eval.n_per_pose=10"
# done

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
