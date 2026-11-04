"""Projeção t-SNE do latente do CVAE com z=8, no mesmo protocolo do VAE (latent.py).

Usa as médias mu das 10 000 imagens de teste, perplexidade 30 e inicialização pela PCA.
Gera results/cvae_z8_tsne.png.
Uso: .venv/bin/python src/cvae_tsne.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from cvae import CVAE, encode_mu
from data import load_splits
from latent import RESULTS, scatter


def main():
    _, _, (x_te, y_te) = load_splits()
    model = CVAE(latent_dim=8)
    model.load_state_dict(torch.load(RESULTS / "cvae_z8.pt", map_location="cpu")["state_dict"])
    model.eval()

    z = encode_mu(model, x_te, y_te)
    z_tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter(ax, z_tsne, y_te.numpy())
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE do latente do CVAE z=8 (mu, conjunto de teste)")
    fig.savefig(RESULTS / "cvae_z8_tsne.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
