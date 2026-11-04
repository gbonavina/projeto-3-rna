"""Busca de topologia do VAE para espaços latentes de 2 e 8 dimensões.

Cada configuração é treinada com early stopping na loss de validação (-ELBO);
o melhor modelo de cada dimensão latente é salvo em results/.

Uso: uv run python src/train.py [--epochs 50] [--patience 5]
"""

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import torch

from data import load_splits
from model import VAE, vae_loss

RESULTS = Path(__file__).resolve().parent.parent / "results"

LATENT_DIMS = [2, 8]
# Camadas ocultas do MLP do encoder (o decoder usa a ordem inversa).
SEARCH_SPACE = {
    "hidden_dims": [(512, 256)],
    "lr": [1e-3],
}


def batches(x: torch.Tensor, batch_size: int, shuffle: bool):
    idx = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for i in range(0, len(x), batch_size):
        yield x[idx[i : i + batch_size]]


@torch.no_grad()
def evaluate(model: VAE, x: torch.Tensor, batch_size: int = 1000) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, device=x.device)
    for xb in batches(x, batch_size, shuffle=False):
        recon, mu, logvar = model(xb)
        totals += torch.stack(vae_loss(recon, xb, mu, logvar)) * len(xb)
    loss, bce, kl = (totals / len(x)).tolist()
    return {"loss": loss, "bce": bce, "kl": kl}


def train_one(config: dict, x_train, x_val, epochs: int, patience: int, batch_size: int, device: str):
    model = VAE(latent_dim=config["latent_dim"], hidden_dims=config["hidden_dims"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb in batches(x_train, batch_size, shuffle=True):
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)

        val = evaluate(model, x_val)
        history["train_loss"].append(running / len(x_train))
        history["val_loss"].append(val["loss"])

        if val["loss"] < best_val:
            best_val, best_state, bad_epochs = val["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, evaluate(model, x_val), history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    (x_train, _), (x_val, _), _ = load_splits(device=device)
    RESULTS.mkdir(exist_ok=True)

    keys = list(SEARCH_SPACE)
    all_runs = []
    for latent_dim in LATENT_DIMS:
        best = None
        for values in itertools.product(*SEARCH_SPACE.values()):
            config = {"latent_dim": latent_dim, **dict(zip(keys, values))}
            torch.manual_seed(args.seed)
            start = time.time()
            model, val, history = train_one(config, x_train, x_val, args.epochs, args.patience, args.batch_size, device)
            run = {**config, **{f"val_{k}": v for k, v in val.items()}, "epochs": len(history["val_loss"]),
                   "params": sum(p.numel() for p in model.parameters()), "history": history}
            all_runs.append(run)
            print(f"z={latent_dim} hidden={config['hidden_dims']} lr={config['lr']:.0e} "
                  f"-> val {val['loss']:.2f} (bce {val['bce']:.2f} + kl {val['kl']:.2f}) "
                  f"[{run['epochs']} ép, {time.time() - start:.0f}s]", flush=True)

            if best is None or val["loss"] < best[1]["val_loss"]:
                best = (model, run)

        model, run = best
        torch.save({"config": {k: run[k] for k in ["latent_dim", *keys]}, "state_dict": model.state_dict()},
                   RESULTS / f"vae_z{latent_dim}.pt")
        print(f"==> melhor z={latent_dim}: hidden={run['hidden_dims']} lr={run['lr']} "
              f"val {run['val_loss']:.2f}\n", flush=True)

    (RESULTS / "search.json").write_text(json.dumps(all_runs, indent=2))


if __name__ == "__main__":
    main()
