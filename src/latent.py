"""Exploração do espaço latente dos melhores VAEs (z=2 e z=8) no conjunto de teste.

Gera em results/:
  - latent_z2.png        : mu (2D) colorido pelos rótulos
  - latent_z8_pca.png    : projeção PCA 2D de mu (8D) com a variância explicada
  - manifold_z2.png      : imagens decodificadas de uma grade no espaço latente 2D
  - pca_z8_variance.png  : variância explicada por componente (8 PCs)
  - latent_z8_tsne.png   : projeção t-SNE 2D de mu (8D), preserva vizinhanças locais
  - latent_metrics.json  : silhouette (clusters/separação) e variância da PCA

Uso: uv run python src/latent.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

from data import CLASSES, load_splits
from model import VAE

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_model(latent_dim: int, device: str) -> VAE:
    ckpt = torch.load(RESULTS / f"vae_z{latent_dim}.pt", map_location=device)
    cfg = ckpt["config"]
    model = VAE(latent_dim=cfg["latent_dim"], hidden_dims=tuple(cfg["hidden_dims"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def encode(model: VAE, x: torch.Tensor) -> np.ndarray:
    """Usa mu (média de q(z|x)) como representação latente determinística."""
    return torch.cat([model.encoder(xb)[0] for xb in x.split(1000)]).cpu().numpy()


def scatter(ax, z2d: np.ndarray, y: np.ndarray):
    sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=y, cmap="tab10", s=2, alpha=0.6, vmin=-0.5, vmax=9.5)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels(CLASSES)


def separation_metrics(z: np.ndarray, y: np.ndarray, z_fit: np.ndarray, y_fit: np.ndarray) -> dict:
    """Silhouette com os rótulos como clusters e acurácia de um k-NN treinado no latente da validação."""
    knn = KNeighborsClassifier(n_neighbors=10).fit(z_fit, y_fit)
    return {
        "silhouette_labels": float(silhouette_score(z, y, sample_size=5000, random_state=0)),
        "knn_accuracy": float(knn.score(z, y)),
    }


@torch.no_grad()
def plot_manifold(model: VAE, device: str, n: int = 20):
    """Grade nos quantis de N(0,1): mostra o que o decoder gera em cada região do latente 2D."""
    grid = norm.ppf(np.linspace(0.05, 0.95, n))
    zz = torch.tensor([[xi, yi] for yi in grid[::-1] for xi in grid], dtype=torch.float32, device=device)
    imgs = model.decoder(zz).cpu().numpy().reshape(n, n, 28, 28)
    canvas = imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas, cmap="gray", extent=[grid[0], grid[-1], grid[0], grid[-1]])
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Manifold do decoder (z=2)")
    fig.savefig(RESULTS / "manifold_z2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, (x_val, y_val), (x_test, y_test) = load_splits(device=device)
    y_val, y_test = y_val.cpu().numpy(), y_test.cpu().numpy()
    metrics = {}

    # --- z = 2: o próprio latente já é 2D
    model2 = load_model(2, device)
    z2, z2_val = encode(model2, x_test), encode(model2, x_val)
    metrics["z2"] = separation_metrics(z2, y_test, z2_val, y_val)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter(ax, z2, y_test)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Espaço latente 2D (mu, conjunto de teste)")
    fig.savefig(RESULTS / "latent_z2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    plot_manifold(model2, device)

    # --- z = 8: projeção PCA em 2D (PCA ajustado na validação, aplicado no teste)
    model8 = load_model(8, device)
    z8, z8_val = encode(model8, x_test), encode(model8, x_val)
    pca = PCA(n_components=8).fit(z8_val)
    ratio = pca.explained_variance_ratio_
    z8_2d = pca.transform(z8)[:, :2]

    metrics["z8"] = separation_metrics(z8, y_test, z8_val, y_val)
    metrics["z8_pca2d"] = separation_metrics(z8_2d, y_test, pca.transform(z8_val)[:, :2], y_val)
    metrics["pca_explained_variance_ratio"] = ratio.tolist()
    metrics["pca_2d_total_variance"] = float(ratio[:2].sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter(ax, z8_2d, y_test)
    ax.set_xlabel(f"PC1 ({ratio[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.1%})")
    ax.set_title(f"PCA do espaço latente 8D — {ratio[:2].sum():.1%} da variância")
    fig.savefig(RESULTS / "latent_z8_pca.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, 9), ratio, label="por componente")
    ax.plot(range(1, 9), np.cumsum(ratio), "o-", color="black", label="acumulada")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Variância explicada")
    ax.legend()
    fig.savefig(RESULTS / "pca_z8_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # t-SNE: projeção não linear que preserva vizinhanças (onde o k-NN em 8D mostra a separação)
    z8_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z8)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter(ax, z8_tsne, y_test)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE do espaço latente 8D (mu, conjunto de teste)")
    fig.savefig(RESULTS / "latent_z8_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (RESULTS / "latent_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
