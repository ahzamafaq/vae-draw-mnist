# VAE vs DRAW on MNIST

> **Deep Learning Mini-Project** · ICLR 2025 format report  
> Oliver Wakeford · Ahzam Afaq · Said Abolhassan Razavi

---

## Overview

We compare two generative models for MNIST digit generation:

| Model | Description |
|-------|-------------|
| **Vanilla VAE** | MLP encoder/decoder with Bernoulli output, standard ELBO objective |
| **DRAW (no attention)** | Recurrent VAE with T sequential read/write steps, no spatial attention |
| **DRAW (with attention)** | Full DRAW model (Gregor et al., 2015) with Gaussian filterbank attention |

The key research questions are:
- How much does recurrent generation improve sample quality over a one-shot VAE?
- How much additional benefit does Gaussian filterbank attention provide?
- How does the number of recurrent steps T affect performance?

---

## Results Summary

| Model | T | Test NELBO ↓ |
|-------|---|------------|
| VAE | — | ~100 |
| DRAW (no attn) | 10 | ~87 |
| DRAW (attn) | 1 | ~96 |
| DRAW (attn) | 5 | ~88 |
| DRAW (attn) | 10 | ~83 |

Full results, training curves, reconstruction comparisons, and T-ablations are in the [compiled report](report/main.pdf).

---

## Repository Layout

```
mini-project/
├── README.md
├── .gitignore
├── code/
│   ├── data.py          # MNIST DataLoader helpers (auto-downloads)
│   ├── vae.py           # Vanilla VAE (MLP, Bernoulli ELBO)
│   ├── draw.py          # DRAW model (toggle attention with a flag)
│   ├── train.py         # Training CLI — see usage below
│   ├── make_figures.py  # Regenerates all report figures from outputs/
│   └── outputs/         # Saved curves (.npz), summaries (.json), checkpoints (.pt)
└── report/
    ├── main.tex         # Full paper source (ICLR 2025 template)
    ├── main.pdf         # Compiled report
    ├── references.bib
    ├── math_commands.tex
    ├── figures/         # PDFs used by main.tex
    ├── iclr2025_conference.{sty,bst}
    ├── fancyhdr.sty
    └── natbib.sty
```

> **Note:** Model checkpoints (`.pt` files, ~3–10 MB each) are excluded from git. Regenerate them with the training commands below.

---

## Setup

### Requirements

- Python ≥ 3.9
- PyTorch 2.x + torchvision (CPU or CUDA or Apple MPS — auto-detected)
- NumPy, Matplotlib

```bash
pip install torch torchvision numpy matplotlib
```

MNIST is downloaded automatically on first run via `torchvision.datasets.MNIST`.

---

## Training

All commands should be run from inside `code/`:

```bash
cd code

# Vanilla VAE (30 epochs)
python train.py --model vae --epochs 30

# DRAW without attention (T=10, 30 epochs)
python train.py --model draw_noattn --epochs 30 --T 10

# DRAW with attention — main run (T=10, 60 epochs)
python train.py --model draw_attn --epochs 60 --T 10

# DRAW with attention — T ablation runs
python train.py --model draw_attn --epochs 30 --T 1
python train.py --model draw_attn --epochs 30 --T 5
```

Checkpoints, loss curves, and JSON summaries are saved to `outputs/`.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *required* | `vae`, `draw_noattn`, or `draw_attn` |
| `--T` | `10` | Number of recurrent read/write steps (DRAW only) |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `128` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--seed` | `42` | Random seed (PyTorch + NumPy) |
| `--out_dir` | `outputs` | Directory for saved artefacts |
| `--data_root` | `./mnist_data` | MNIST download location |

---

## Generating Figures & Rebuilding the Report

```bash
# 1. Generate all figures (requires completed outputs/)
cd code
python make_figures.py

# 2. Compile the LaTeX report
cd ../report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Reproducibility

| Setting | Value |
|---------|-------|
| Random seed | 42 (PyTorch + NumPy) |
| PyTorch | 2.11.0 |
| torchvision | 0.26.0 |
| LaTeX | TeX Live 2025 (`pdflatex` + `bibtex`) |

Device is auto-selected: Apple MPS → CUDA → CPU.

---

## Reference

Gregor, K., Danihelka, I., Graves, A., Rezende, D. J., & Wierstra, D. (2015).  
**DRAW: A Recurrent Neural Network For Image Generation.**  
*ICML 2015.* [arXiv:1502.04623](https://arxiv.org/abs/1502.04623)
