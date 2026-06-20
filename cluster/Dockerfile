# DiffIK training image for NRP Nautilus (CUDA). Mirrors the prior miksolver/diffik
# strategy: CUDA PyTorch base + deps. Datasets are generated on-the-fly in-container
# (deterministic by seed), so no datasets.zip baking is needed.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y git vim && \
    pip install --no-cache-dir wandb scikit-learn numpy scipy pandas matplotlib tqdm pyyaml optuna

RUN git config --global user.name "Jacket Demby's" && \
    git config --global user.email "jacketdembys@yahoo.fr"

# Option A (default): the Job git-clones the repo at runtime (see job template).
# Option B: bake the code in instead by uncommenting:
# COPY . /home/diffik
# RUN pip install -e /home/diffik
WORKDIR /home
