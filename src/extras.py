"""Figuras complementares para o relatório (usa os modelos já treinados).

Gera em results/:
  - loss_curves.png       : -ELBO de treino e validação por época (z=2 e z=8)
  - reconstructions.png   : imagens de teste e reconstruções dos dois VAEs
  - samples.png           : amostras geradas a partir de z ~ N(0, I)
  - kl_per_dim.png        : KL médio por dimensão latente (detecta dimensões inativas)
  - extras_metrics.json   : métricas de teste, PCA do z=2 e KL por dimensão

Uso: .venv/bin/python src/extras.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from data import load_splits
from latent import encode, load_model
from train import evaluate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def grid(imgs: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return imgs.reshape(rows, cols, 28, 28).transpose(0, 2, 1, 3).reshape(rows * 28, cols * 28)


@torch.no_grad()
def kl_per_dim(model, x: torch.Tensor) -> np.ndarray:
    mus, logvars = zip(*(model.encoder(xb) for xb in x.split(1000)))
    mu, logvar = torch.cat(mus), torch.cat(logvars)
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0).cpu().numpy()


def main():
    torch.manual_seed(0)
    device = "cpu"
    _, (x_val, _), (x_test, _) = load_splits(device=device)
    models = {z: load_model(z, device) for z in (2, 8)}
    metrics = {}

    # Curvas de perda
    runs = json.loads((RESULTS / "search.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, run in zip(axes, runs):
        h = run["history"]
        ep = range(1, len(h["val_loss"]) + 1)
        ax.plot(ep, h["train_loss"], label="treino")
        ax.plot(ep, h["val_loss"], label="validação")
        best = int(np.argmin(h["val_loss"]))
        ax.axvline(best + 1, color="gray", ls="--", lw=0.8, label=f"melhor época ({best + 1})")
        ax.set_title(f"z = {run['latent_dim']}")
        ax.set_xlabel("Época")
        ax.set_ylim(230, 275)
        ax.legend()
    axes[0].set_ylabel("-ELBO por amostra")
    fig.savefig(RESULTS / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Métricas no teste
    for z, m in models.items():
        metrics[f"test_z{z}"] = evaluate(m, x_test)
        metrics[f"kl_per_dim_z{z}"] = kl_per_dim(m, x_test).tolist()

    # PCA do z=2 (só uma rotação dos eixos, mas mostra se o latente é anisotrópico)
    z2_val = encode(models[2], x_val)
    metrics["pca_z2_explained_variance_ratio"] = PCA(2).fit(z2_val).explained_variance_ratio_.tolist()
    metrics["z2_std_per_dim"] = z2_val.std(0).tolist()
    metrics["z8_std_per_dim"] = encode(models[8], x_val).std(0).tolist()

    # Reconstruções: 12 imagens de teste
    idx = torch.arange(0, 10_000, 10_000 // 12)[:12]
    xs = x_test[idx]
    with torch.no_grad():
        rec = {z: m.decoder(m.encoder(xs)[0]) for z, m in models.items()}
    rows = torch.cat([xs, rec[2], rec[8]]).squeeze(1).numpy()
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.imshow(grid(rows, 3, 12), cmap="gray")
    ax.set_yticks([14, 42, 70], ["original", "z = 2", "z = 8"])
    ax.set_xticks([])
    fig.savefig(RESULTS / "reconstructions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Amostras do prior
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (z, m) in zip(axes, models.items()):
        with torch.no_grad():
            s = m.decoder(torch.randn(64, z)).squeeze(1).numpy()
        ax.imshow(grid(s, 8, 8), cmap="gray")
        ax.set_title(f"z = {z}")
        ax.axis("off")
    fig.savefig(RESULTS / "samples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # KL por dimensão
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"width_ratios": [1, 3]}, sharey=True)
    for ax, z in zip(axes, (2, 8)):
        kl = metrics[f"kl_per_dim_z{z}"]
        ax.bar(range(1, z + 1), kl)
        ax.set_xticks(range(1, z + 1))
        ax.set_xlabel("Dimensão latente")
        ax.set_title(f"z = {z}")
    axes[0].set_ylabel("KL médio (nats)")
    fig.savefig(RESULTS / "kl_per_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / "extras_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
