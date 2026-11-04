"""VAE condicional (CVAE) no Fashion-MNIST.

Mesma topologia do VAE (MLP 512-256), mas encoder e decoder recebem o rótulo
em one-hot concatenado à entrada: q(z|x,y) e p(x|z,y).

Gera em results/ (d = dimensão latente):
  - cvae_z{d}.pt              : pesos do melhor modelo (early stopping na validação)
  - cvae_z{d}_latent.png      : mu do teste colorido pelos rótulos (PCA 2D se d > 2)
  - cvae_z{d}_generation.png  : uma linha por classe, mesmo z em cada coluna
  - cvae_z{d}_metrics.json    : -ELBO, silhouette e k-NN no latente (e PCA se d > 2)

Uso: .venv/bin/python src/cvae.py [--latent-dim 2]
"""

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.decomposition import PCA

from data import CLASSES, load_splits
from latent import scatter, separation_metrics
from model import INPUT_DIM, mlp, vae_loss

RESULTS = Path(__file__).resolve().parent.parent / "results"
N_CLASSES = 10


class CVAE(nn.Module):
    def __init__(self, latent_dim: int = 2, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = mlp([INPUT_DIM + N_CLASSES, *hidden_dims])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        self.dec = mlp([latent_dim + N_CLASSES, *reversed(hidden_dims)])
        self.out = nn.Linear(hidden_dims[0], INPUT_DIM)

    def encode(self, x, y):
        h = self.enc(torch.cat([x.flatten(1), F.one_hot(y, N_CLASSES).float()], 1))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, y):
        h = self.dec(torch.cat([z, F.one_hot(y, N_CLASSES).float()], 1))
        return torch.sigmoid(self.out(h)).view(-1, 1, 28, 28)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y), mu, logvar


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    totals = torch.zeros(3)
    for xb, yb in zip(x.split(1000), y.split(1000)):
        recon, mu, logvar = model(xb, yb)
        totals += torch.stack(vae_loss(recon, xb, mu, logvar)) * len(xb)
    loss, bce, kl = (totals / len(x)).tolist()
    return {"loss": loss, "bce": bce, "kl": kl}


@torch.no_grad()
def encode_mu(model, x, y):
    return torch.cat([model.encode(xb, yb)[0] for xb, yb in zip(x.split(1000), y.split(1000))]).numpy()


def main(latent_dim: int = 2, epochs: int = 50, patience: int = 5, batch_size: int = 128):
    torch.manual_seed(0)
    tag = f"cvae_z{latent_dim}"
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_splits()
    model = CVAE(latent_dim=latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best, best_state, bad, history = float("inf"), None, 0, {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        running = 0.0
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), batch_size):
            idx = perm[i : i + batch_size]
            recon, mu, logvar = model(x_tr[idx], y_tr[idx])
            loss, _, _ = vae_loss(recon, x_tr[idx], mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        val = evaluate(model, x_val, y_val)["loss"]
        history["train_loss"].append(running / len(x_tr))
        history["val_loss"].append(val)
        print(f"ép {ep + 1}: treino {running / len(x_tr):.2f} val {val:.2f}", flush=True)
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": best_state}, RESULTS / f"{tag}.pt")

    metrics = {
        "val": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_te, y_te),
        "epochs": len(history["val_loss"]),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "params": sum(p.numel() for p in model.parameters()),
        "history": history,
    }
    z_te, z_val = encode_mu(model, x_te, y_te), encode_mu(model, x_val, y_val)
    metrics["latent"] = separation_metrics(z_te, y_te.numpy(), z_val, y_val.numpy())

    fig, ax = plt.subplots(figsize=(9, 7))
    if latent_dim == 2:
        z_plot, (xl, yl) = z_te, ("z1", "z2")
        title = "Espaço latente do CVAE (mu, conjunto de teste)"
    else:
        # Mesmo protocolo do VAE z=8: PCA ajustado na validação, aplicado no teste.
        pca = PCA(n_components=latent_dim).fit(z_val)
        ratio = pca.explained_variance_ratio_
        z_plot = pca.transform(z_te)[:, :2]
        metrics["pca_explained_variance_ratio"] = ratio.tolist()
        metrics["pca_2d_total_variance"] = float(ratio[:2].sum())
        metrics["latent_pca2d"] = separation_metrics(z_plot, y_te.numpy(), pca.transform(z_val)[:, :2], y_val.numpy())
        xl, yl = f"PC1 ({ratio[0]:.1%})", f"PC2 ({ratio[1]:.1%})"
        title = f"PCA do latente do CVAE z={latent_dim} ({ratio[:2].sum():.1%} da variância)"
    scatter(ax, z_plot, y_te.numpy())
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    fig.savefig(RESULTS / f"{tag}_latent.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Cada coluna usa o mesmo z e cada linha fixa a classe. Com z=2 percorre z1 nos quantis de
    # N(0,1) (z2 = 0); com mais dimensões usa z sorteados do prior.
    n = 10
    if latent_dim == 2:
        zs = torch.tensor([[v, 0.0] for v in norm.ppf(np.linspace(0.05, 0.95, n))], dtype=torch.float32)
    else:
        zs = torch.randn(n, latent_dim, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        rows = [model.decode(zs, torch.full((n,), c)).squeeze(1).numpy() for c in range(N_CLASSES)]
    canvas = np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas, cmap="gray")
    ax.set_yticks([14 + 28 * c for c in range(N_CLASSES)], CLASSES)
    if latent_dim == 2:
        ax.set_xticks([14 + 28 * i for i in range(n)], [f"{v:.1f}" for v in zs[:, 0].tolist()])
        ax.set_xlabel("z1 (com z2 = 0)")
    else:
        ax.set_xticks([])
        ax.set_xlabel("cada coluna: um z ~ N(0, I) compartilhado pelas classes")
    fig.savefig(RESULTS / f"{tag}_generation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, default=2)
    main(latent_dim=parser.parse_args().latent_dim)
