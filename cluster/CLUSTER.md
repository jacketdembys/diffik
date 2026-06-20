# Running DiffIK on the cluster (NRP Nautilus)

Mirrors the prior miksolver/diffik strategy: a CUDA Docker image + Kubernetes Jobs,
one Job per experiment, results logged to **wandb**. Datasets are generated
on-the-fly in-container (deterministic by seed), so nothing needs to be baked in.

## Prerequisites (one-time)
1. **Push this repo to git** (the Job `git clone`s it):
   ```bash
   git remote add origin https://github.com/jacketdembys/diffik.git   # or your remote
   git push -u origin main
   ```
   (The Job clones `${REPO}` set in `generate-jobs.sh`.)
2. **Image: use your existing `gitlab-registry.nrp-nautilus.io/udembys/diffik:latest` — no rebuild needed.**
   The Job runs `git clone` + `pip install -e .`, which pulls any missing runtime
   deps (likely just `pyyaml`) and does NOT touch torch (pin is `torch>=2.0`, so any
   2.0+ in your image satisfies it). Runtime needs only torch/numpy/pyyaml/wandb;
   roboticstoolbox is test-only. nodeAffinity targets L4/3090/A10.
   *(Optional)* rebuild from `cluster/Dockerfile` only if you want deps baked in for
   faster job startup (skips the runtime `pip install`).
3. (Optional) persistent storage for raw run dirs:
   ```bash
   kubectl apply -f cluster/pvc.yaml      # then uncomment the volume lines in the job template
   ```

## Launch experiments (two phases)
**Phase 1 — dataset-size sweep** (find the best dataset size):
```bash
bash cluster/generate-jobs.sh            # 10 LBE dataset sizes; edit grid/WANDB_KEY inside
kubectl get jobs -l k8s-app=diffik-job   # monitor
kubectl logs -l k8s-app=diffik-job --tail=50
```
Analyze (e.g. `python scripts/pull_wandb.py`) and pick the best `n_trajectories`.

**Phase 2 — model-size sweep** at that best dataset size:
```bash
DS_N=<best n_trajectories> bash cluster/generate-model-jobs.sh
```

Cleanup any time: `bash cluster/delete-jobs.sh`
The wandb key is set in `generate-jobs.sh` (`WANDB_KEY`) and the Job runs
`wandb login ${WANDB_KEY}` in the pod (same as the prior workflow). Rotate the key
if the repo is shared publicly.

## How a Job runs
Each Job: `git clone` repo -> `pip install -e .` -> `wandb login ${WANDB_KEY}` ->
`python scripts/train.py --config <cfg> --override <per-experiment tokens>`.
The `--override` tokens are `section.field=value` (e.g. `data.n_trajectories=6400
train.epochs=1500 wandb=true`). Metrics, per-epoch loss, and the checkpoint
(as a wandb Artifact) are logged to wandb; if a PVC is mounted, `runs/<name>/`
(config, dataset, checkpoint, history, metrics, errors) is also persisted.

## What differs from the local M3/MPS runs
- `get_device()` auto-selects **CUDA** on the cluster (MPS locally).
- Use the full-scale configs (`configs/panda_lbe_trajectory.yaml`: T=1000, hidden=1024,
  1M samples) — far faster on an A100/A10 than locally.
- `wandb=true` so results persist beyond the ephemeral pod.
