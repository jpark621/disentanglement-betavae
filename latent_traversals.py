from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset

from data import DSpritesDataset
from model import BetaVAE

if __name__ == "__main__":
    beta = 10
    seed = 12345
    save_path = f"runs/sweep/beta{beta}_seed{seed}"

    model = BetaVAE()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(Path(save_path) / "model.ckpt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    dataset = DSpritesDataset()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset), size=10000, replace=False)
    dataloader = DataLoader(Subset(dataset, idx), batch_size=512, shuffle=True)
    
    mus = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x = x.unsqueeze(1)
            h = model.encoder(x)
            mu, _ = h.chunk(2, dim=1)

            mus.append(mu.cpu().numpy())
    mus = np.vstack(mus)

    p01, p99 = np.percentile(mus, [1, 99], axis=0)

    i = rng.choice(len(dataset))
    x, _ = dataset[i]
    sample_mu = None
    with torch.no_grad():
        x = x.to(device)
        x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        h = model.encoder(x)
        mu, _ = h.chunk(2, dim=1)

        mu = mu.squeeze(0)
        sample_mu = mu.cpu().numpy()

    factors_and_latents = [('scale', 8), ('posX', 6), ('posY', 4)]
    for factor, latent in factors_and_latents:
        num_points = 6
        sweep = []
        dx = (p99[latent] - p01[latent]) / (num_points - 1)
        for i in range(num_points):
            point = p01[latent] + i * dx
            sweep.append(point)
    
        imgs = []
        for point in sweep:
            z = sample_mu.copy()
            z[latent] = point
            with torch.no_grad():
                z = torch.tensor(z).unsqueeze(0)
                h = model.decoder_fc(z).view(-1, 64, 4, 4)
                logits = model.decoder_conv(h)
                logits = logits.squeeze(1)
                img = torch.sigmoid(logits)
                img = img.squeeze(0)
                
                img = img.cpu().numpy()
                imgs.append(img)
        imgs = np.stack(imgs)

        fig, axes = plt.subplots(1, len(imgs), figsize=(2 * len(imgs), 2))
        for ax, img, v in zip(axes, imgs, sweep):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{v:.1f}")
            ax.axis("off")
        fig.savefig(Path(save_path) / f"traversal_factor={factor}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
