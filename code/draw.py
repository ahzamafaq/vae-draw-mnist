import torch
from torch import nn
import torch.nn.functional as F


def _filterbank(g, log_sigma_sq, log_delta, log_gamma, A, B, N, device):
    """
    Build the two Gaussian filterbanks F_x (N,A) and F_y (N,B) used by DRAW.

    """
    gx_t, gy_t = g[:, 0:1], g[:, 1:2]
    gx = 0.5 * (A + 1) * (gx_t + 1.0)
    gy = 0.5 * (B + 1) * (gy_t + 1.0)
    delta = (max(A, B) - 1) / (N - 1) * torch.exp(log_delta)
    sigma2 = torch.exp(log_sigma_sq)
    gamma = torch.exp(log_gamma)

    # grid of patch indices, shifted so the centre is at gx, gy
    rng = torch.arange(N, device=device, dtype=torch.float32) - 0.5 * N - 0.5
    mu_x = gx + (rng + 1) * delta  # (batch, N)
    mu_y = gy + (rng + 1) * delta

    a = torch.arange(A, device=device, dtype=torch.float32).view(1, 1, A)
    b = torch.arange(B, device=device, dtype=torch.float32).view(1, 1, B)

    Fx = torch.exp(-((a - mu_x.unsqueeze(2)) ** 2) / (2 * sigma2.unsqueeze(2)))
    Fy = torch.exp(-((b - mu_y.unsqueeze(2)) ** 2) / (2 * sigma2.unsqueeze(2)))
    # normalise each row
    Fx = Fx / (Fx.sum(dim=2, keepdim=True) + 1e-8)
    Fy = Fy / (Fy.sum(dim=2, keepdim=True) + 1e-8)
    return Fx, Fy, gamma  # Fx: (batch,N,A), Fy: (batch,N,B), gamma: (batch,1)


class DRAW(nn.Module):
    """
    DRAW: A Recurrent Neural Network For Image Generation 
    Two flavours toggled by `attention`:
      - False: read = (x, x_err), write = full-image linear projection
      - True:  Gaussian filterbank read/write with N x N patch
    Image is 28x28 MNIST.
    """
    def __init__(self, T=10, z_dim=10, h_dim=256, N=5, attention=True,
                 H=28, W=28):
        super().__init__()
        self.T = T
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.N = N
        self.attention = attention
        self.H, self.W = H, W
        self.x_dim = H * W

        if attention:
            read_dim = 2 * N * N
            write_dim = N * N
            # 5 attention params for read, 5 for write (gx, gy, log_sigma2, log_delta, log_gamma)
            self.fc_read_attn = nn.Linear(h_dim, 5)
            self.fc_write_attn = nn.Linear(h_dim, 5)
            self.fc_write_patch = nn.Linear(h_dim, write_dim)
        else:
            read_dim = 2 * self.x_dim
            write_dim = self.x_dim
            self.fc_write_patch = nn.Linear(h_dim, write_dim)

        self.enc_rnn = nn.LSTMCell(read_dim + h_dim, h_dim)
        self.dec_rnn = nn.LSTMCell(z_dim, h_dim)

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    # attention ops 
    def _attn_params(self, h, kind):
        layer = self.fc_read_attn if kind == 'read' else self.fc_write_attn
        params = layer(h)
        gx_t, gy_t, log_sigma_sq, log_delta, log_gamma = params.split(1, dim=1)
        g = torch.cat([torch.tanh(gx_t), torch.tanh(gy_t)], dim=1)
        return g, log_sigma_sq, log_delta, log_gamma

    def _read(self, x, x_err, h_dec):
        if not self.attention:
            return torch.cat([x, x_err], dim=1)
        g, lss, ld, lg = self._attn_params(h_dec, 'read')
        Fx, Fy, gamma = _filterbank(g, lss, ld, lg, self.W, self.H, self.N, x.device)
        # reshape to (batch, H, W)
        x_img = x.view(-1, self.H, self.W)
        e_img = x_err.view(-1, self.H, self.W)
        glimpse_x = gamma.unsqueeze(2) * (Fy @ x_img @ Fx.transpose(1, 2))
        glimpse_e = gamma.unsqueeze(2) * (Fy @ e_img @ Fx.transpose(1, 2))
        return torch.cat([glimpse_x.view(-1, self.N * self.N),
                          glimpse_e.view(-1, self.N * self.N)], dim=1)

    def _write(self, h_dec):
        if not self.attention:
            return self.fc_write_patch(h_dec)
        patch = self.fc_write_patch(h_dec).view(-1, self.N, self.N)
        g, lss, ld, lg = self._attn_params(h_dec, 'write')
        Fx, Fy, gamma = _filterbank(g, lss, ld, lg, self.W, self.H, self.N, h_dec.device)
        out = (1.0 / (gamma.unsqueeze(2) + 1e-8)) * (Fy.transpose(1, 2) @ patch @ Fx)
        return out.view(-1, self.x_dim)

    # forward 
    def forward(self, x):
        batch = x.size(0)
        device = x.device
        x_flat = x.view(batch, -1)

        c = torch.zeros(batch, self.x_dim, device=device)
        h_enc = torch.zeros(batch, self.h_dim, device=device)
        c_enc = torch.zeros(batch, self.h_dim, device=device)
        h_dec = torch.zeros(batch, self.h_dim, device=device)
        c_dec = torch.zeros(batch, self.h_dim, device=device)

        kl_total = torch.zeros(batch, device=device)
        for _ in range(self.T):
            x_hat = torch.sigmoid(c)
            x_err = x_flat - x_hat
            r = self._read(x_flat, x_err, h_dec)
            h_enc, c_enc = self.enc_rnn(torch.cat([r, h_dec], dim=1), (h_enc, c_enc))
            mu = self.fc_mu(h_enc)
            logvar = self.fc_logvar(h_enc)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl_total = kl_total - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
            h_dec, c_dec = self.dec_rnn(z, (h_dec, c_dec))
            c = c + self._write(h_dec)

        # final loss: BCE on canvas + sum of step KLs
        bce = F.binary_cross_entropy_with_logits(c, x_flat, reduction='none').sum(dim=1)
        nelbo = bce + kl_total
        return c, nelbo.mean(), bce.mean(), kl_total.mean()

    @torch.no_grad()
    def sample(self, n, device, return_steps=False):
        c = torch.zeros(n, self.x_dim, device=device)
        h_dec = torch.zeros(n, self.h_dim, device=device)
        c_dec = torch.zeros(n, self.h_dim, device=device)
        steps = []
        for _ in range(self.T):
            z = torch.randn(n, self.z_dim, device=device)
            h_dec, c_dec = self.dec_rnn(z, (h_dec, c_dec))
            c = c + self._write(h_dec)
            if return_steps:
                steps.append(torch.sigmoid(c).view(n, 1, self.H, self.W).cpu())
        if return_steps:
            return torch.sigmoid(c).view(n, 1, self.H, self.W), steps
        return torch.sigmoid(c).view(n, 1, self.H, self.W)
