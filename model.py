import time
import torch
from torch import nn, optim
import torch.nn.functional as F


class BetaVAE(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), nn.ReLU(), 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(4*4*64, 256), nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4), nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # logits
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)

        # reparameterize
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        h = self.decoder_fc(z).view(-1, 64, 4, 4)
        logits = self.decoder_conv(h)
        logits = logits.squeeze(1)
        return logits, mu, logvar
        
def loss_fn(X_hat_logits, X, mu, logvar, beta):
    B = X.shape[0]
    recon = F.binary_cross_entropy_with_logits(X_hat_logits, X, reduction="sum") / B
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum() / B
    return recon + beta * kl, recon, kl


if __name__ == "__main__":
    x = torch.randn(32, 64, 64)
    model = BetaVAE()
    logits, mu, logvar = model(x)
    print(logits.shape)
    print(mu.shape)
    print(logvar.shape)

    t0 = time.perf_counter()
    model = BetaVAE()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(1000):
        x = torch.randn(32, 64, 64)
        logits, mu, logvar = model(x)
        loss, _, _ = loss_fn(logits, x, mu, logvar, 1)
        loss.backward()
        optimizer.step()
    duration = time.perf_counter() - t0
    print(f"Processing is {duration / 1000} s")

