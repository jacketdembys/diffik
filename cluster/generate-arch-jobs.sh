#!/bin/bash
# PHASE 3 — Architecture-type sweep: compare LBE denoiser backbones plain/rmlp/dmlp
# at a fixed size + dataset (the best from earlier phases). IROS found RMLP >> Plain,
# so this is the main remaining lever toward sub-mm. Runs report full-test best/mean/
# worst-of-K (min/mean/max over the K solutions) + ranges to wandb.
#
# Usage:  bash cluster/generate-arch-jobs.sh                                  (all 3 backbones, 768x6, n6400)
#         HIDDEN=512 LAYERS=4 bash cluster/generate-arch-jobs.sh              (other size)
#         DS_N=25600 BACKBONES=rmlp bash cluster/generate-arch-jobs.sh       (best backbone on largest dataset)
set -euo pipefail

export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export CONFIG="configs/panda_lbe_trajectory.yaml"
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"   # rotate if repo public
TEMPLATE="cluster/job-template-gpu.yaml"

DS_N="${DS_N:-6400}"        # best dataset (204,800 train)
HIDDEN="${HIDDEN:-768}"     # best size from the model-size sweep
LAYERS="${LAYERS:-6}"
SEED=0; MAXEPOCHS=1500; PATIENCE=15
BACKBONES="${BACKBONES:-plain rmlp dmlp}"   # e.g. BACKBONES=rmlp to run one

launch () { export JOBNAME="$1"; export OVERRIDES="$2"; echo "launch ${JOBNAME}"; envsubst < "${TEMPLATE}" | kubectl apply -f -; }

for bb in ${BACKBONES}; do
  launch "diffik-arch-n${DS_N}-h${HIDDEN}-l${LAYERS}-${bb}" \
    "name=lbe_n${DS_N}_h${HIDDEN}_l${LAYERS}_${bb} seed=${SEED} wandb=true wandb_group=lbe_arch_sweep_n${DS_N}_${HIDDEN}x${LAYERS} \
     data.kind=trajectory data.lbe=true data.n_trajectories=${DS_N} data.steps_per_traj=40 data.v_deg=1.0 \
     model.type=lbe model.backbone=${bb} model.hidden_dim=${HIDDEN} model.n_layers=${LAYERS} \
     diffusion.T=1000 diffusion.fk_loss_weight=10.0 diffusion.rot_weight=0.1 diffusion.p_example_dropout=0.2 \
     train.epochs=${MAXEPOCHS} train.patience=${PATIENCE} train.monitor_every=10 \
     eval.n_per_pose=1 eval.seeded=true"
done

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
