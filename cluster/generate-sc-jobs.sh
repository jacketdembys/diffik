#!/bin/bash
# PHASE 5 — Self-conditioning A/B on the best backbone (RMLP 768x6) / dataset (n6400),
# fk_weighting=alpha_bar (the FK sweep showed more concentration hurts), eval DDIM eta=0.
# scoff = controlled baseline (fresh retrain, no self-cond); scon = self-conditioning on.
# Both log full-test best-of-K + ranges + min/mean/max-of-K to wandb
# (group lbe_sc_sweep_n6400_768x6).
#
# Usage:  bash cluster/generate-sc-jobs.sh
set -euo pipefail

export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export CONFIG="configs/panda_lbe_trajectory.yaml"
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"   # rotate if repo public
TEMPLATE="cluster/job-template-gpu.yaml"

DS_N="${DS_N:-6400}"; HIDDEN="${HIDDEN:-768}"; LAYERS="${LAYERS:-6}"
SEED=0; MAXEPOCHS=1500; PATIENCE=15

SPECS="scoff:false scon:true"   # tag:self_cond

launch () { export JOBNAME="$1"; export OVERRIDES="$2"; echo "launch ${JOBNAME}"; envsubst < "${TEMPLATE}" | kubectl apply -f -; }

for spec in ${SPECS}; do
  IFS=: read -r tag sc <<< "${spec}"
  launch "diffik-sc-n${DS_N}-${tag}" \
    "name=lbe_n${DS_N}_h${HIDDEN}_l${LAYERS}_rmlp_${tag} seed=${SEED} wandb=true wandb_group=lbe_sc_sweep_n${DS_N}_${HIDDEN}x${LAYERS} \
     data.kind=trajectory data.lbe=true data.n_trajectories=${DS_N} data.steps_per_traj=40 data.v_deg=1.0 \
     model.type=lbe model.backbone=rmlp model.hidden_dim=${HIDDEN} model.n_layers=${LAYERS} model.self_cond=${sc} \
     diffusion.T=1000 diffusion.fk_loss_weight=10.0 diffusion.rot_weight=0.1 diffusion.p_example_dropout=0.2 \
     diffusion.fk_weighting=alpha_bar \
     train.epochs=${MAXEPOCHS} train.patience=${PATIENCE} train.monitor_every=10 \
     eval.n_per_pose=1 eval.seeded=true eval.sampler=ddim eval.eta=0.0"
done

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
