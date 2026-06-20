# DiffIK

Refinement-free, sub-millimeter, multimodal **diffusion inverse kinematics** for serial
manipulators. Seedless diffusion gives diverse solutions at ~mm; the **Learning-by-Example
(LBE)** seed — folded in via classifier-free guidance — unlocks sub-mm accuracy, in one model.

## Setup
```bash
conda activate diffik          # Python 3.11, torch 2.x (MPS/CUDA), roboticstoolbox (dev)
pip install -e .
pytest                         # 38 tests: FK, data, diffusion, eval, FK-loss, samplers, LBE
```

## Package layout
- `diffik/kinematics/` — batched differentiable standard-DH FK + Jacobian, pose utils (verified vs roboticstoolbox to 1e-9)
- `diffik/data/` — two **separate** generators (`generate_random`, `generate_trajectory`), LBE example pairs (`add_examples`), normalization, leakage-safe splits
- `diffik/models/` — `MLPDenoiser` (seedless), `LBEDenoiser` (pose + example, null-embedding for CFG)
- `diffik/diffusion/` — DDPM schedule, `GaussianDiffusion` (denoising + differentiable-FK loss), `LBEDiffusion` (CFG), DDPM/DDIM samplers
- `diffik/eval/` — position-mm / orientation-deg metrics, %≤1mm/≤1deg, best-of-N, diversity, timing
- `diffik/{config,checkpoint}.py`, `scripts/train.py` — config-driven training/eval

## Train + evaluate
```bash
python scripts/train.py --config configs/smoke.yaml                 # quick local sanity
python scripts/train.py --config configs/panda_lbe_trajectory.yaml  # full-scale LBE (cluster)
python scripts/train.py --config configs/panda_seedless_random.yaml # seedless multimodal
# override any field:
python scripts/train.py --config configs/smoke.yaml --override train.epochs=200 data.n_trajectories=500
```
Outputs land in `runs/<name>/`: `config.json`, `dataset.npz`, `checkpoint.pth`, `metrics.json`.

## Numerical baselines (MATLAB bridge)
Reuses the validated MATLAB SD/SVF/MX (no port); solved joints are scored with our FK.
```bash
python scripts/export_for_matlab.py             # -> matlab/diffik_testset.csv (joint vectors)
# in MATLAB: run matlab/numerical_ik_diffik.m   -> matlab/diffik_numerical_results.csv
python scripts/score_matlab_results.py          # SD/SVF/MX in the same table as DiffIK
```
External learning baselines (DIK, IKFlow, IKDiffuser): compared via published numbers.
