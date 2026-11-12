"""PVNet-style spatio-temporal fusion network for Romanian solar / wind quantile forecasting."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class SatelliteAdvection3DEncoder(nn.Module):
    """(B, C_sat, T_sat, H, W) -> (B, d_model)."""

    def __init__(self, in_channels: int = 5, d_model: int = 32, base: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base, kernel_size=3, padding=1),
            nn.BatchNorm3d(base),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(base, 2 * base, kernel_size=3, padding=1),
            nn.BatchNorm3d(2 * base),
            nn.SiLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Linear(2 * base, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.proj(self.net(x).flatten(1)))


class Earth2EnsembleSpatialEncoder(nn.Module):
    """(B, T, C_e2, H, W) ensemble mean/spread maps -> (B, T, d_model), folding time into the batch."""

    def __init__(self, in_channels: int = 16, d_model: int = 32, base: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(),
            nn.Conv2d(base, 2 * base, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 2 * base),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(2 * base * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = self.net(x.reshape(b * t, c, h, w)).flatten(1)
        return F.silu(self.proj(z)).reshape(b, t, -1)


class PVNetRomaniaFusion(nn.Module):
    """Fuses satellite, telemetry, Earth-2 and exogenous branches into monotone quantiles (B, T_future, 5)."""

    def __init__(
        self,
        f_hist: int,
        f_exo: int,
        c_sat: int = 5,
        c_e2: int = 16,
        d_model: int = 32,
        n_quantiles: int = len(QUANTILES),
        capacity_ceiling_mw: float = 3550.0,
        chronos_dim: int = 0,
    ) -> None:
        super().__init__()
        self.capacity_ceiling_mw = float(capacity_ceiling_mw)
        self.n_quantiles = n_quantiles
        self.chronos_dim = chronos_dim
        self.sat_enc = SatelliteAdvection3DEncoder(c_sat, d_model)
        self.e2_enc = Earth2EnsembleSpatialEncoder(c_e2, d_model)
        self.hist_gru = nn.GRU(f_hist, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.exo_proj = nn.Sequential(nn.Linear(f_exo + chronos_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        fused = 4 * d_model
        self.temporal = nn.Conv1d(fused, fused, kernel_size=5, padding=2)
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, 2 * fused),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * fused, fused),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        # Residual head anchored on the NWP baseline: Q monotone spread increments + 1 median correction.
        # Zero-init weights => prediction starts exactly at the NWP quantile fan, so the model can only
        # improve on NWP, never collapse to zero when uncertain.
        self.out = nn.Linear(fused, n_quantiles + 1)
        nn.init.zeros_(self.out.weight)
        bias = torch.full((n_quantiles + 1,), -2.5)  # softplus(-2.5) ~ 0.08 of capacity per quantile step
        bias[-1] = 0.0  # median correction starts at 0 (i.e. q50 = NWP)
        with torch.no_grad():
            self.out.bias.copy_(bias)

    def forward(
        self,
        sat: torch.Tensor,
        hist: torch.Tensor,
        e2: torch.Tensor,
        exo: torch.Tensor,
        solar_elevation: torch.Tensor | None = None,
        chronos_prior: torch.Tensor | None = None,
        nwp_frac: torch.Tensor | None = None,
        lead_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_future = exo.shape[1]
        sat_ctx = self.sat_enc(sat)  # (B, d)
        _, h_n = self.hist_gru(hist)
        hist_ctx = h_n[-1]  # (B, d)
        e2_seq = self.e2_enc(e2)  # (B, T, d)
        if self.chronos_dim:
            if chronos_prior is None:
                chronos_prior = exo.new_zeros(exo.shape[0], t_future, self.chronos_dim)
            exo = torch.cat([exo, chronos_prior], dim=-1)
        exo_seq = self.exo_proj(exo)  # (B, T, d)
        fused = torch.cat(
            [sat_ctx[:, None].expand(-1, t_future, -1), hist_ctx[:, None].expand(-1, t_future, -1), e2_seq, exo_seq],
            dim=-1,
        )
        fused = fused + F.silu(self.temporal(fused.transpose(1, 2)).transpose(1, 2))
        z = self.out(fused + self.head(fused))  # (B, T, Q+1): Q spread increments + median correction
        inc = F.softplus(z[..., :-1])
        spread = torch.cumsum(inc, dim=-1)  # monotone increasing, (B, T, Q)
        mid = self.n_quantiles // 2
        med_spread = spread[..., mid : mid + 1]
        if nwp_frac is None:
            nwp_frac = torch.zeros_like(med_spread)
        elif nwp_frac.dim() == 2:
            nwp_frac = nwp_frac.unsqueeze(-1)
        if lead_weight is None:
            lead_weight = torch.ones_like(z[..., -1:])
        elif lead_weight.dim() == 1:
            lead_weight = lead_weight.view(1, -1, 1)
        # Gate the learned median correction by the nowcast horizon: beyond it the satellite signal is
        # exhausted, so q50 falls back to the NWP baseline instead of an overfit correction.
        q = (nwp_frac + lead_weight * z[..., -1:] + (spread - med_spread)) * self.capacity_ceiling_mw
        q = torch.clamp(q, 0.0, self.capacity_ceiling_mw)
        if solar_elevation is not None:
            q = q * (solar_elevation > 0.0).to(q.dtype).unsqueeze(-1)
        return q


class CalibratedPinballLoss(nn.Module):
    def __init__(self, quantiles: tuple[float, ...] = QUANTILES) -> None:
        super().__init__()
        self.register_buffer("q", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """y_hat: (B, T, Q), y: (B, T)."""
        err = y.unsqueeze(-1) - y_hat
        loss = torch.maximum(self.q * err, (self.q - 1.0) * err)
        return loss.mean()


def chronos_bolt_prior(
    history: np.ndarray,
    horizon: int,
    quantiles: tuple[float, ...] = QUANTILES,
    model_id: str = "amazon/chronos-bolt-tiny",
) -> np.ndarray:
    """Optional zero-shot Chronos-Bolt quantile prior (T_future, Q) from a 1-D history; requires chronos-forecasting."""
    from chronos import BaseChronosPipeline

    pipe = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)
    q, _ = pipe.predict_quantiles(
        context=torch.tensor(history, dtype=torch.float32), prediction_length=horizon, quantile_levels=list(quantiles)
    )
    return q[0].numpy()
