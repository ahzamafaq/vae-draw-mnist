import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

from data import get_loaders
from vae import VAE, vae_loss
from draw import DRAW


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_loss(model, loader, device, kind):
    model.eval()
    total = 0.0
    n_seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if kind == 'vae':
                logits, mu, logvar = model(x)
                nelbo, _, _ = vae_loss(logits, x, mu, logvar)
            else:
                _, nelbo, _, _ = model(x)
            total += nelbo.item() * x.size(0)
            n_seen += x.size(0)
    model.train()
    return total / n_seen


def train_one(args):
    device = pick_device()
    print(f"device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = get_loaders(batch_size=args.batch_size,
                                            root=args.data_root)

    if args.model == 'vae':
        model = VAE(z_dim=20).to(device)
        kind = 'vae'
    elif args.model == 'draw_noattn':
        model = DRAW(T=args.T, z_dim=10, h_dim=256, N=5, attention=False).to(device)
        kind = 'draw'
    elif args.model == 'draw_attn':
        model = DRAW(T=args.T, z_dim=10, h_dim=256, N=5, attention=True).to(device)
        kind = 'draw'
    else:
        raise ValueError(args.model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}, T={args.T}, params={n_params}")

    opt = optim.Adam(model.parameters(), lr=args.lr)

    train_curve = []
    val_curve = []
    epoch_times = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        running_n = 0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            if kind == 'vae':
                logits, mu, logvar = model(x)
                loss, _, _ = vae_loss(logits, x, mu, logvar)
            else:
                _, loss, _, _ = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * x.size(0)
            running_n += x.size(0)

        train_loss = running / running_n
        val_loss = evaluate_loss(model, test_loader, device, kind)
        dt = time.time() - t0
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        epoch_times.append(dt)
        print(f"epoch {epoch+1:02d}/{args.epochs}  train={train_loss:.2f}  test={val_loss:.2f}  ({dt:.1f}s)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_T{args.T}" if kind == 'draw' else args.model
    np.savez(out / f"{tag}_curves.npz",
             train=np.array(train_curve),
             val=np.array(val_curve),
             epoch_times=np.array(epoch_times))
    torch.save(model.state_dict(), out / f"{tag}_ckpt.pt")
    summary = {
        "model": args.model,
        **( {"T": args.T} if kind == 'draw' else {} ),
        "epochs": args.epochs,
        "params": n_params,
        "final_train_nelbo": train_curve[-1],
        "final_test_nelbo": val_curve[-1],
        "best_test_nelbo": float(min(val_curve)),
        "mean_epoch_seconds": float(np.mean(epoch_times)),
        "device": str(device),
    }
    with open(out / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved:", tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['vae', 'draw_noattn', 'draw_attn'], required=True)
    p.add_argument('--T', type=int, default=10)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--data_root', default='./mnist_data')
    args = p.parse_args()
    train_one(args)


if __name__ == '__main__':
    main()
