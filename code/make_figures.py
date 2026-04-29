import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import get_loaders, fixed_test_batch
from vae import VAE
from draw import DRAW

OUT = Path("outputs")
FIG = Path("../report/figures")
FIG.mkdir(parents=True, exist_ok=True)


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_vae(ckpt, dev):
    m = VAE(z_dim=20).to(dev)
    m.load_state_dict(torch.load(OUT / ckpt, map_location=dev, weights_only=True))
    m.eval()
    return m


def load_draw(ckpt, attention, dev, T=10):
    m = DRAW(T=T, z_dim=10, h_dim=256, N=5, attention=attention).to(dev)
    m.load_state_dict(torch.load(OUT / ckpt, map_location=dev, weights_only=True))
    m.eval()
    return m


def grid(imgs, n_row):
    # imgs: (n,1,28,28) tensor in [0,1] 
    n = imgs.size(0)
    n_col = (n + n_row - 1) // n_row
    canvas = np.ones((n_row * 28, n_col * 28))
    for i in range(n):
        r, c = i // n_col, i % n_col
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[i, 0].numpy()
    return canvas


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    runs = [
        ("vae_curves.npz", "VAE", "tab:blue"),
        ("draw_noattn_T10_curves.npz", "DRAW, no attention", "tab:orange"),
        ("draw_attn_T10_curves.npz", "DRAW, attention", "tab:green"),
    ]
    for fname, label, c in runs:
        path = OUT / fname
        if not path.exists():
            print(f"missing {fname}, skipping")
            continue
        data = np.load(path)
        ax.plot(np.arange(1, len(data["val"]) + 1), data["val"], label=label, color=c)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test negative ELBO (nats per image)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.pdf")
    plt.close(fig)


def fig_reconstructions(dev):
    _, test_loader = get_loaders(batch_size=128)
    x, _ = fixed_test_batch(test_loader, n=8)
    x = x.to(dev)

    rows = [("Originals", x.cpu())]
    if (OUT / "vae_ckpt.pt").exists():
        m = load_vae("vae_ckpt.pt", dev)
        with torch.no_grad():
            logits, _, _ = m(x)
            recon = torch.sigmoid(logits).view(-1, 1, 28, 28).cpu()
        rows.append(("VAE", recon))
    if (OUT / "draw_noattn_T10_ckpt.pt").exists():
        m = load_draw("draw_noattn_T10_ckpt.pt", False, dev)
        with torch.no_grad():
            c, _, _, _ = m(x)
            recon = torch.sigmoid(c).view(-1, 1, 28, 28).cpu()
        rows.append(("DRAW, no attn", recon))
    if (OUT / "draw_attn_T10_ckpt.pt").exists():
        m = load_draw("draw_attn_T10_ckpt.pt", True, dev)
        with torch.no_grad():
            c, _, _, _ = m(x)
            recon = torch.sigmoid(c).view(-1, 1, 28, 28).cpu()
        rows.append(("DRAW, attn", recon))

    fig, axes = plt.subplots(len(rows), 1, figsize=(5.5, 0.9 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (name, imgs) in zip(axes, rows):
        ax.imshow(grid(imgs, 1), cmap='gray', vmin=0, vmax=1)
        ax.set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "reconstructions.pdf")
    plt.close(fig)


def fig_samples(dev):
    nrow, ncol = 4, 8
    n = nrow * ncol
    panels = []
    if (OUT / "vae_ckpt.pt").exists():
        m = load_vae("vae_ckpt.pt", dev)
        with torch.no_grad():
            s = m.sample(n, dev).cpu()
        panels.append(("VAE", s))
    if (OUT / "draw_noattn_T10_ckpt.pt").exists():
        m = load_draw("draw_noattn_T10_ckpt.pt", False, dev)
        s = m.sample(n, dev).cpu()
        panels.append(("DRAW, no attn", s))
    if (OUT / "draw_attn_T10_ckpt.pt").exists():
        m = load_draw("draw_attn_T10_ckpt.pt", True, dev)
        s = m.sample(n, dev).cpu()
        panels.append(("DRAW, attn", s))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (name, imgs) in zip(axes, panels):
        ax.imshow(grid(imgs, nrow), cmap='gray', vmin=0, vmax=1)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "samples.pdf")
    plt.close(fig)


def fig_draw_steps(dev):
    if not (OUT / "draw_attn_T10_ckpt.pt").exists():
        return
    m = load_draw("draw_attn_T10_ckpt.pt", True, dev)
    final, steps = m.sample(8, dev, return_steps=True)
    fig, axes = plt.subplots(8, len(steps), figsize=(0.55 * len(steps), 4.5))
    for j, s in enumerate(steps):
        for i in range(8):
            axes[i, j].imshow(s[i, 0].numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f"t={j+1}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "draw_steps.pdf")
    plt.close(fig)


def fig_t_ablation():
    if not all((OUT / f"draw_attn_T{T}_curves.npz").exists() for T in (1, 5, 10)):
        print("t-ablation curves not all present; skipping")
        return
    # Cap all curves at 30 epochs so the comparison is on an equal budget.
    # draw_attn_T10 was trained for 60 epochs; we only show the first 30 here.
    ABLATION_EPOCHS = 30
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    for T, c in zip((1, 5, 10), ("tab:red", "tab:orange", "tab:green")):
        d = np.load(OUT / f"draw_attn_T{T}_curves.npz")
        val = d["val"][:ABLATION_EPOCHS]
        ax.plot(np.arange(1, len(val) + 1), val,
                label=f"T={T}", color=c)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test negative ELBO")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "t_ablation.pdf")
    plt.close(fig)


def write_metrics():
    rows = {}
    for tag in ["vae", "draw_noattn_T10", "draw_attn_T10",
                "draw_attn_T1", "draw_attn_T5"]:
        f = OUT / f"{tag}_summary.json"
        if f.exists():
            rows[tag] = json.load(open(f))
    # Extract the epoch-30 test NELBO from the 60-epoch draw_attn_T10 run.
    # This is the fair matched-budget number used in the report table.
    curves_f = OUT / "draw_attn_T10_curves.npz"
    if curves_f.exists():
        val_curve = np.load(curves_f)["val"]
        if len(val_curve) >= 30:
            rows["draw_attn_T10_epoch30_test_nelbo"] = float(val_curve[29])
    with open(OUT / "final_metrics.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(json.dumps(rows, indent=2))


def main():
    dev = device()
    print("device:", dev)
    fig_training_curves()
    fig_reconstructions(dev)
    fig_samples(dev)
    fig_draw_steps(dev)
    fig_t_ablation()
    write_metrics()


if __name__ == '__main__':
    main()
