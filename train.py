import os
from pathlib import Path
import time

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import DSpritesDataset
from model import BetaVAE, loss_fn


def train(beta, seed, max_steps, save_path):
    torch.manual_seed(seed); np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using " + str(device))

    dataloader = DataLoader(DSpritesDataset(), batch_size=64, shuffle=True)
    
    model = BetaVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter(log_dir=save_path)
    
    steps = 0
    max_steps = max_steps
    while True:
        for X, _ in dataloader:
            X = X.to(device, non_blocking=True)
            X_hat_logits, mu, logvar = model(X)
            loss, recon, kl = loss_fn(X_hat_logits, X, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % 1000 == 0:
                writer.add_scalar("loss/recon", recon.item(), global_step=steps)
                writer.add_scalar("loss/kl", kl.item(), global_step=steps)
                writer.add_scalar("loss/total", loss.item(), global_step=steps)
            
            steps += 1
            if steps == max_steps:
                break
        if steps == max_steps:
            break

    writer.close()

    # Save model checkpoints
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "beta": beta,
            "seed": seed,
            "max_steps": max_steps
        }
    }, str(Path(save_path) / "model.ckpt"))
    

if __name__ == "__main__":
    max_steps = 300000
    t0 = time.perf_counter()
    train(beta=1, seed=42, max_steps=max_steps, save_path="runs/first_test")
    duration = time.perf_counter() - t0
    print(f"Speed is {duration / max_steps:0.5f} s/steps")

