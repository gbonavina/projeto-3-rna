"""Métricas de fidelidade de imagem para os VAEs e CVAEs (z=2 e z=8).

- PSNR e SSIM: reconstrução (a partir de mu) de cada imagem de teste comparada com a original.
- FID: 10 000 amostras geradas a partir de z ~ N(0, I) comparadas com as 10 000 imagens de teste,
  usando as features de 2048 dimensões da Inception-v3. Nos CVAEs as amostras são condicionadas em
  1 000 rótulos de cada classe, a mesma proporção do conjunto de teste. Como referência, calcula-se
  também o FID entre 10 000 imagens reais de treino e o teste (o melhor valor alcançável na prática).

As imagens 28x28 em cinza são replicadas em 3 canais; o torchmetrics redimensiona para 299x299.

Gera results/fidelity_metrics.json (PSNR/SSIM) e results/fid_metrics.json (FID).
Uso: .venv/bin/python src/fidelity.py [--no-fid] [--n-fid 10000]
A extração de features da Inception é lenta em CPU (~0,1 s/imagem); prefira GPU para o FID.
"""

import argparse
import json
from pathlib import Path

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from cvae import CVAE
from data import load_splits
from latent import load_model

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHUNK = 250


def load_cvae(latent_dim: int, device: str) -> CVAE:
    model = CVAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(RESULTS / f"cvae_z{latent_dim}.pt", map_location=device)["state_dict"])
    return model.to(device).eval()


def rgb(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1)


@torch.no_grad()
def reconstruction_metrics(reconstruct, x: torch.Tensor, y: torch.Tensor) -> dict:
    psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3)).to(x.device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(x.device)
    psnrs = []
    for xb, yb in zip(x.split(1000), y.split(1000)):
        rec = reconstruct(xb, yb)
        psnrs.append(psnr(rec, xb))
        ssim.update(rec, xb)
    return {"psnr": torch.cat(psnrs).mean().item(), "ssim": ssim.compute().item()}


@torch.no_grad()
def fid_score(fid: FrechetInceptionDistance, fake: torch.Tensor) -> float:
    fid.reset()  # mantém as features reais (reset_real_features=False)
    for xb in fake.split(CHUNK):
        fid.update(rgb(xb), real=False)
    return fid.compute().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fid", action="store_true", help="calcula só PSNR e SSIM")
    parser.add_argument("--n-fid", type=int, default=10_000, help="amostras por conjunto no FID")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = args.n_fid
    (x_tr, _), _, (x_te, y_te) = load_splits(device=device)

    recon, fids = {}, {"n_samples": n}
    if not args.no_fid:
        fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
        for xb in x_te[:n].split(CHUNK):
            fid.update(rgb(xb), real=True)
        fids["reference_train_vs_test"] = fid_score(fid, x_tr[:n])
        print(f"referência (treino x teste): FID {fids['reference_train_vs_test']:.2f}", flush=True)

    # Rótulos das amostras condicionadas: mesma proporção de classes do teste (balanceado).
    labels = torch.arange(10, device=device).repeat_interleave(n // 10)
    for z in (2, 8):
        vae, cvae = load_model(z, device), load_cvae(z, device)
        models = {
            f"vae_z{z}": (lambda xb, yb, m=vae: m.decoder(m.encoder(xb)[0]),
                          lambda e, l, m=vae: m.decoder(e)),
            f"cvae_z{z}": (lambda xb, yb, m=cvae: m.decode(m.encode(xb, yb)[0], yb),
                           lambda e, l, m=cvae: m.decode(e, l)),
        }
        for name, (reconstruct, generate) in models.items():
            recon[name] = reconstruction_metrics(reconstruct, x_te, y_te)
            print(name, recon[name], flush=True)
            if not args.no_fid:
                with torch.no_grad():
                    eps = torch.randn(len(labels), z, device=device)
                    samples = torch.cat([generate(e, l) for e, l in zip(eps.split(1000), labels.split(1000))])
                fids[name] = fid_score(fid, samples)
                print(f"{name}: FID {fids[name]:.2f}", flush=True)

    (RESULTS / "fidelity_metrics.json").write_text(json.dumps(recon, indent=2))
    if not args.no_fid:
        (RESULTS / "fid_metrics.json").write_text(json.dumps(fids, indent=2))


if __name__ == "__main__":
    main()
