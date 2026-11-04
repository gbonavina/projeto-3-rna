import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 28 * 28


def mlp(dims: list[int]) -> nn.Sequential:
    """Camadas densas com ReLU entre todas (inclusive após a última)."""
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """MLP sobre a imagem achatada (784) e projeção em mu e logvar."""

    def __init__(self, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.net = mlp([INPUT_DIM, *hidden_dims])
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x.flatten(start_dim=1))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Espelho do encoder: reconstrói 28x28 a partir de z. Sigmoide só na última camada."""

    def __init__(self, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.net = mlp([latent_dim, *reversed(hidden_dims)])
        self.out = nn.Linear(hidden_dims[0], INPUT_DIM)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.out(self.net(z)))
        return x.view(-1, 1, 28, 28)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dims: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, hidden_dims)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(
    recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-ELBO médio por amostra: BCE (soma sobre pixels) + KL(q(z|x) || N(0, I))."""
    n = x.size(0)
    bce = F.binary_cross_entropy(recon, x, reduction="sum") / n
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
    return bce + kl, bce, kl
