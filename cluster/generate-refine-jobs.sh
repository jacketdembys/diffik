#!/bin/bash
# (1) Full-test refinement headline + (2) baselines, on the cluster.
#   - refine_diffusion_seeded   : diffusion seeded init + 1..5 GN steps (fast single sub-mm)
#   - refine_diffusion_seedless : diffusion seedless init + up to 15 GN steps (multimodal sub-mm)
#   - refine_random_randinit    : random-init Newton BASELINE (same K/steps) -> shows diffusion's value
# Each logs per-step best-of-K / mean / %sub-mm / diversity / distinct sub-mm modes to
# wandb (group refine_eval_<RUN>). Diffusion-alone numbers = step 0 of the diffusion runs.
#
# Usage:  bash cluster/generate-refine-jobs.sh
set -euo pipefail

export IMAGE="gitlab-registry.nrp-nautilus.io/udembys/diffik:latest"
export REPO="https://github.com/jacketdembys/diffik.git"
export WANDB_KEY="d7f81da19c5965b1c5eff37a677caab3ffb5379c"   # rotate if repo public
TEMPLATE="cluster/job-template-refine.yaml"

RUN="${RUN:-lbe_n6400_h768_l6_rmlp_rw01scoff}"
K="${K:-20}"; LAM="${LAM:-1e-3}"
GROUP="refine_eval_${RUN}"

launch () { export JOBNAME="$1"; export ARGS="$2"; echo "launch ${JOBNAME}"; envsubst < "${TEMPLATE}" | kubectl apply -f -; }

launch "diffik-refine-seeded" \
  "--run ${RUN} --init diffusion --regime seeded   --K ${K} --steps 5  --lam ${LAM} --wandb --wandb_group ${GROUP}"
launch "diffik-refine-seedless" \
  "--run ${RUN} --init diffusion --regime seedless --K ${K} --steps 15 --lam ${LAM} --wandb --wandb_group ${GROUP}"
launch "diffik-refine-random" \
  "--run ${RUN} --init random --K ${K} --steps 15 --lam ${LAM} --wandb --wandb_group ${GROUP}"

echo "done. monitor: kubectl get jobs -l k8s-app=diffik-job"
