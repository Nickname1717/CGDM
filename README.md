# CGDM: Cross‑Modality Gradient‑Guided Diffusion for Graph Generation

This repository implements **CGDM**, a graph generation framework that performs diffusion in the **embedding modality** and periodically injects **graph‑modality** structural signals for guidance. The codebase is developed on top of **GDSS** (Score‑based Generative Modeling of Graphs). We keep the GDSS training/evaluation pipeline and add the following components:

- **TUD (Transformer‑U‑Net Diffusion)**: an embedding‑modality denoiser with multi‑scale token downsampling (FPS) and skip‑connected fusion.
- **Cross‑modality mean‑shift guidance**: every \(K\) steps, decode \(Z\!\to\!G\), refine \(G\) with a graph denoiser, re‑encode to \(Z'\) via GEMM, and apply a stop‑gradient mean shift \(Z\leftarrow Z+\eta\,g\) with \(g\propto(Z'-Z)\).
- **GEMM (Graph–Embedding Mapping Module)** with **NDRSA (Node–Distance Relation Self‑Attention)**: encoders/decoders that preserve discrete structural cues in the \(G\!\leftrightarrow\!Z\) mapping.
- **GDM (Graph Denoising Model)** with a **hybrid‑noise family** for training; noisers are **not** used at inference.
- **Unified evaluation protocol** and scripts for all benchmarks.

If you use any part of this repository, please also credit the original GDSS project: <https://github.com/harryjo97/GDSS>.

---

## Datasets

We evaluate on four **general‑graph** datasets (Ego‑small, Community‑small, ENZYMES, Grid) and two **molecular** datasets (QM9, ZINC250k). Data loaders follow GDSS preprocessing and official splits.

### Prepare generic graph datasets
```bash
python data/data_generators.py --dataset {ego_small|community_small|ENZYMES|grid}
```

### Prepare molecular graph datasets
```bash
python data/preprocess.py --dataset {QM9|ZINC250k}
python data/preprocess_for_nspdk.py --dataset {QM9|ZINC250k}
```

### Compile ORCA (for generic‑graph evaluation)
```bash
cd evaluation/orca
g++ -O2 -std=c++11 -o orca orca.cpp
cd -
```

---

## Environment

Install dependencies from `requirements.txt` (PyTorch + common scientific Python stack). For molecule evaluation, install RDKit (e.g., via conda). Example:

```bash
pip install -r requirements.txt
# optional for molecule metrics
conda install -c conda-forge rdkit
```

---

## Configuration

All experiment configurations live in `config/` as YAML files. Each dataset has a file that mirrors the **paper hyperparameters** (training budget, model widths, token retention, guidance schedule, etc.). Examples:

- `config/ego_small.yaml`
- `config/community_small.yaml`
- `config/ENZYMES.yaml`
- `config/grid.yaml`
- `config/QM9.yaml`
- `config/ZINC250k.yaml`

> Tip: keep `timesteps=1000` for diffusion; select guidance steps \(K\) per dataset (e.g., 8 for QM9, 16 for ZINC250k in our runs).

---

## Training

There are three training stages. Each stage is launched with `--type train --config {dataset}`.
Replace `{dataset}` with the name of the dataset (e.g., `QM9`, `ZINC250k`, `ego_small`).

### 1. Train the Graph–Embedding Mapping Module (VAE)
```bash
CUDA_VISIBLE_DEVICES=0 python vae_trainer.py --type train --config QM9
```

### 2. Train the Embedding‑Modality Diffusion (TUD, Latent Diffusion)
```bash
CUDA_VISIBLE_DEVICES=0 python ldm_trainer.py --type train --config QM9
```

### 3. Train the Graph Denoiser (GDM)
```bash
CUDA_VISIBLE_DEVICES=0 python graphdenoiser_trainer.py --type train --config QM9
```

---

## Generation & Evaluation

Inference is performed using the latent diffusion trainer after all three models are trained:

```bash
CUDA_VISIBLE_DEVICES=0 python ldm_trainer.py --type sample --config QM9
```

This will generate samples and evaluate them under the unified protocol (validity, uniqueness, novelty, FCD, NSPDK, etc. for molecules; degree/cluster/orbit for generic graphs).

After training, generate samples and compute metrics under the **unified protocol**:

```bash
# Sampling / evaluation
CUDA_VISIBLE_DEVICES=0 python main.py --type sample --config sample_qm9
CUDA_VISIBLE_DEVICES=0 python main.py --type sample --config sample_zinc250k

# Generic‑graph metrics (Deg./Clus./Orb.) use ORCA; molecular metrics include FCD/NSPDK/Validity.
```

