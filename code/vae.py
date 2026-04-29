import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Vanilla VAE for MNIST. MLP encoder/decoder, Bernoulli output.
    Loss returned per image, summed over pixels (standard ELBO scale).
    """
    def __init__(self, x_dim=784, h_dim=400, z_dim=20):
        super().__init__()
        # encoder
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        # decoder
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)

        self.x_dim = x_dim
        self.z_dim = z_dim

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)  # logits, sigmoid done in BCE_with_logits

    def forward(self, x):
        x_flat = x.view(-1, self.x_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparam(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.z_dim, device=device)
        logits = self.decode(z)
        return torch.sigmoid(logits).view(-1, 1, 28, 28)


def vae_loss(logits, x, mu, logvar):
    # negative ELBO, summed over pixels, mean over batch
    x_flat = x.view(x.size(0), -1)
    bce = F.binary_cross_entropy_with_logits(logits, x_flat, reduction='none').sum(dim=1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    nelbo = bce + kl
    return nelbo.mean(), bce.mean(), kl.mean()
