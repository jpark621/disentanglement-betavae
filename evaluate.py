from pathlib import Path

import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader, Subset

from data import DSpritesDataset
from model import BetaVAE

def generate_mus_and_factors(save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(str(Path(save_path) / "model.ckpt"), map_location=device)
    model = BetaVAE()
    model.load_state_dict(ckpt["model"])
    
    dataset = DSpritesDataset()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(dataset), size=10000, replace=False)
    loader = DataLoader(Subset(dataset, idx), batch_size=512, shuffle=False)
    
    mus, factors = [], []
    with torch.no_grad():
        for x, f in loader:
            x = x.to(device)
            x = x.unsqueeze(1) 
            h = model.encoder(x) 
            mu, _ = h.chunk(2, dim=1) 
     
            mus.append(mu.cpu().numpy())
            factors.append(f.numpy())
    mus = np.vstack(mus)
    factors = np.vstack(factors)
    return mus, factors
    #np.save(str(Path(save_path) / "mus.npy"), mus)
    #np.save(str(Path(save_path) / "factors.npy"), factors)

def discretize(z, n_bins=20):
    # bin each latent dim separately on its own range
    return np.stack([np.digitize(z[:, j], np.histogram(z[:, j], n_bins)[1][:-1])
                     for j in range(z.shape[1])], axis=1)

def mig(mu, factors, n_bins=20):
    z = discretize(mu, n_bins)
    K, J = factors.shape[1], z.shape[1]
    mi = np.array([[mutual_info_score(factors[:, k], z[:, j])
                    for j in range(J)] for k in range(K)])      # K x J heatmap
    H = np.array([mutual_info_score(factors[:, k], factors[:, k])
                  for k in range(K)])                           # H(v_k) = I(v_k; v_k)
    top2 = np.sort(mi, axis=1)[:, ::-1][:, :2]
    gaps = (top2[:, 0] - top2[:, 1]) / H                        # one per factor (row)
    normalized_mi = (mi.T / H).T
    return gaps.mean(), gaps, mi, normalized_mi

def plot_mi_heatmap(mi, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    row_names = ['shape', 'scale', 'orientation', 'posX', 'posY']
    sns.heatmap(
        mi,
        yticklabels=row_names,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,  # Anchor at 0 (statistical independence)
        cbar_kws={"label": "Mutual Information"},
        ax=ax,
    )

    # Keep y-axis latent index numbers upright
    ax.tick_params(axis="y", rotation=0)

    # Set axis labels
    ax.set_xlabel("Latents")
    ax.set_ylabel("Factors")
    ax.set_title("Latent–Factor Mutual Information (normalized)")

    plt.tight_layout()
    plt.savefig(str(Path(save_path) / "latent_factor_mi.png"), dpi=300)
    plt.close()
    
def plot_mig_table(results, seeds, betas):
    # 2. Format labels and cell text
    col_labels = [f"β = {b}" for b in betas]
    row_labels = [f"Seed {s}" for s in seeds]
    cell_text = [[f"{val:.3f}" for val in row] for row in results]

    # 3. Set up the figure
    fig, ax = plt.subplots(figsize=(len(betas) * 1.5, len(seeds) * 0.7))
    ax.axis("off")

    # 4. Build the table with both row and column headers
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )

    # 5. Style headers and cells
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Bold the header cells
    for (row, col), cell in table.get_celld().items():
      if row == 0 or col == -1:  # row 0 is colLabels; col -1 is rowLabels
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#f0f0f0")

    # 6. Save image
    plt.savefig("runs/sweep/beta_seed_table.png", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    betas = [1, 3, 5, 7, 10]
    seeds = [1, 42, 12345]
    results = []
    for seed in seeds:
        results_per_seed = []
        for beta in betas:
            save_path = f"runs/sweep/beta{beta}_seed{seed}"
            mus, factors = generate_mus_and_factors(save_path)

            score, _, mi, normalized_mi = mig(mus, factors)
            plot_mi_heatmap(normalized_mi, save_path)
            results_per_seed.append(score)
        results.append(results_per_seed)

    plot_mig_table(results, seeds, betas)

