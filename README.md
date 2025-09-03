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


```

Key options controlled in the config:
- **Guidance steps \(K\)** and placement (even spacing within \(T\)).
- **GDM refinement budget \(m\)** per guided step (use \(m{=}1\) unless you explicitly study cost/quality).
- **Token retention** (full/75%/50%/25%) with **FPS** vs. random at fixed retention.

---

## Reproducibility

- Hyperparameters for each dataset/module are summarized in the paper’s hyperparameter table; the YAML files in `config/` match those settings.
- Timing is reported as **wall‑clock per 10k samples** under a single protocol; we avoid mixing heterogeneous setups.
- We provide scripts to reproduce ablations: guidance \(K\) sweeps, FPS retention sweeps, FPS vs. random, hybrid‑noise variants, and NDRSA ablations.

---


