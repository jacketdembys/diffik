#!/bin/bash
# Sampler A/B on the cluster: DDPM vs DDIM eta=0 vs ZTV on an existing checkpoint,
# FULL test set, K=50, both regimes. One job PER sampler (parallel); each logs
# best-of-K / mean / worst / diversity to wandb (group ab_<RUN>).
#
# Usage:  bash cluster/generate-ab-jobs.sh
#         RUN=lbe_n25600_h768_l6_rmlp bash cluster/generate-ab-jobs.sh
set -euo pipefail

export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"   # rotate if repo public
TEMPLATE="cluster/job-template-eval.yaml"

RUN="${RUN:-lbe_n6400_h768_l6_rmlp}"   # checkpoint to A/B
K="${K:-50}"
ZTV="${ZTV:-50}"
SAMPLERS="${SAMPLERS:-ddpm ddim ztv}"

launch () { export JOBNAME="$1"; export ABARGS="$2"; echo "launch ${JOBNAME}"; envsubst < "${TEMPLATE}" | kubectl apply -f -; }

for s in ${SAMPLERS}; do
  launch "diffik-ab-${RUN//_/-}-${s}" \
    "--run ${RUN} --K ${K} --n_poses 0 --ztv ${ZTV} --samplers ${s} --regimes seedless,seeded --wandb --wandb_group ab_${RUN}"
done

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
