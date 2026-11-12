"""Deterministic clear-sky physics (pvlib) and satellite cloud advection (pySTEPS)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pvlib.location import Location
from scipy import ndimage

from rocast.config import HubCoordinate, RoCastSettings, get_settings


def hub_pixel(hub: HubCoordinate, shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Map a hub to (row, col) on a grid whose row 0 is the northern edge."""
    lon_min, lat_min, lon_max, lat_max = bbox
    h, w = shape
    col = int(np.clip(np.floor((hub.lon - lon_min) / (lon_max - lon_min) * w), 0, w - 1))
    row = int(np.clip(np.floor((lat_max - hub.lat) / (lat_max - lat_min) * h), 0, h - 1))
    return row, col


class ClearSkyPhysicsCalculator:
    """Solar geometry and Ineichen clear-sky irradiance for each Romanian hub."""

    def __init__(self, settings: RoCastSettings | None = None) -> None:
        self.cfg = settings or get_settings()
        self.locations = {h.name: Location(h.lat, h.lon, tz="UTC", altitude=h.alt, name=h.name) for h in self.cfg.hubs}

    @staticmethod
    def _as_utc_index(timestamps) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(timestamps)
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    def hub_clearsky(self, hub_name: str, timestamps) -> pd.DataFrame:
        idx = self._as_utc_index(timestamps)
        loc = self.locations[hub_name]
        sol = loc.get_solarposition(idx)
        cs = loc.get_clearsky(idx, model="ineichen")
        out = pd.DataFrame(
            {
                "apparent_elevation": sol["apparent_elevation"].to_numpy(),
                "apparent_zenith": sol["apparent_zenith"].to_numpy(),
                "ghi": cs["ghi"].to_numpy(),
                "dni": cs["dni"].to_numpy(),
                "dhi": cs["dhi"].to_numpy(),
            },
            index=idx,
        )
        return out

    def all_hubs_clearsky(self, timestamps) -> dict[str, pd.DataFrame]:
        return {h.name: self.hub_clearsky(h.name, timestamps) for h in self.cfg.hubs}

    @staticmethod
    def heliosat_cloud_index(vis_reflectance, rho_ground_min, rho_cloud_max: float = 0.85) -> np.ndarray:
        """Heliosat cloud albedo index n in [0, 1.2]."""
        rho = np.asarray(vis_reflectance, dtype=np.float64)
        rmin = np.asarray(rho_ground_min, dtype=np.float64)
        denom = np.maximum(rho_cloud_max - rmin, 1e-4)
        return np.clip((rho - rmin) / denom, 0.0, 1.2)

    @staticmethod
    def clearsky_index(cloud_index) -> np.ndarray:
        """Clear-sky index K_c = clip(1 - 0.85 n, 0.05, 1.05)."""
        return np.clip(1.0 - 0.85 * np.asarray(cloud_index, dtype=np.float64), 0.05, 1.05)


def _dense_lucas_kanade(frames: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Dense Lucas-Kanade optical flow (pixels/step), accumulated over consecutive frame pairs."""
    f = np.nan_to_num(frames.astype(np.float64))
    t, h, w = f.shape
    a11 = np.zeros((h, w))
    a12 = np.zeros((h, w))
    a22 = np.zeros((h, w))
    b1 = np.zeros((h, w))
    b2 = np.zeros((h, w))
    for i in range(t - 1):
        i0 = ndimage.gaussian_filter(f[i], 1.0)
        i1 = ndimage.gaussian_filter(f[i + 1], 1.0)
        avg = 0.5 * (i0 + i1)
        ix = np.gradient(avg, axis=1)
        iy = np.gradient(avg, axis=0)
        it = i1 - i0
        a11 += ix * ix
        a12 += ix * iy
        a22 += iy * iy
        b1 += -ix * it
        b2 += -iy * it
    a11, a12, a22, b1, b2 = (ndimage.gaussian_filter(x, sigma) for x in (a11, a12, a22, b1, b2))
    reg = 1e-6 + 1e-3 * (a11 + a22).mean()
    a11 = a11 + reg
    a22 = a22 + reg
    det = a11 * a22 - a12 * a12
    u = (a22 * b1 - a12 * b2) / det
    v = (-a12 * b1 + a11 * b2) / det
    return np.clip(np.stack([u, v]), -5.0, 5.0)


def _semilagrangian(frame: np.ndarray, velocity: np.ndarray, n_steps: int) -> np.ndarray:
    """Backward semi-Lagrangian advection of one frame for n_steps; edge values are held."""
    h, w = frame.shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    out = np.empty((n_steps, h, w), dtype=np.float64)
    cur = frame.astype(np.float64)
    for k in range(n_steps):
        # one backward step from the arrival point, with a midpoint velocity estimate
        mx = xx - 0.5 * velocity[0]
        my = yy - 0.5 * velocity[1]
        vu = ndimage.map_coordinates(velocity[0], [my, mx], order=1, mode="nearest")
        vv = ndimage.map_coordinates(velocity[1], [my, mx], order=1, mode="nearest")
        cur = ndimage.map_coordinates(cur, [yy - vv, xx - vu], order=1, mode="nearest")
        out[k] = cur
    return out


class PyStepsCloudNowcaster:
    """Lagrangian cloud-index extrapolation with pySTEPS (dense NumPy fallback)."""

    def __init__(self, settings: RoCastSettings | None = None, use_pysteps: bool = True) -> None:
        self.cfg = settings or get_settings()
        self.physics = ClearSkyPhysicsCalculator(self.cfg)
        self.use_pysteps = use_pysteps
        self.backend = "numpy-fallback"

    def validate_satellite_stack(self, sat_stack: np.ndarray) -> np.ndarray:
        arr = np.asarray(sat_stack, dtype=np.float64)
        nan_ratio = float(np.isnan(arr).mean())
        if nan_ratio > self.cfg.max_satellite_nan_ratio:
            raise ValueError(
                f"Satellite stack NaN ratio {nan_ratio:.3f} exceeds limit {self.cfg.max_satellite_nan_ratio:.3f}"
            )
        if nan_ratio > 0:
            arr = arr.copy()
            for i in range(arr.shape[0]):
                frame = arr[i]
                mask = np.isnan(frame)
                if mask.any():
                    frame[mask] = np.nanmedian(frame) if not mask.all() else 0.0
                    arr[i] = frame
        return arr

    def estimate_motion(self, frames: np.ndarray) -> np.ndarray:
        if self.use_pysteps:
            try:
                from pysteps import motion

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vel = np.asarray(motion.get_method("LK")(frames), dtype=np.float64)
                if vel.shape == (2, *frames.shape[1:]) and np.isfinite(vel).all():
                    self.backend = "pysteps-LK"
                    return vel
            except Exception:
                pass
        self.backend = "numpy-fallback"
        return _dense_lucas_kanade(frames)

    def advect(self, frames: np.ndarray, velocity: np.ndarray, n_lead_steps: int) -> np.ndarray:
        if self.use_pysteps and self.backend == "pysteps-LK":
            try:
                from pysteps import extrapolation

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = np.asarray(
                        extrapolation.get_method("semilagrangian")(frames[-1], velocity, n_lead_steps),
                        dtype=np.float64,
                    )
                if out.shape == (n_lead_steps, *frames.shape[1:]) and np.isfinite(out).all():
                    return out
            except Exception:
                pass
        return _semilagrangian(frames[-1], velocity, n_lead_steps)

    def nowcast(self, cloud_index_stack: np.ndarray, n_lead_steps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return advected_cloud_index (n, H, W) and velocity_field (2, H, W)."""
        n = n_lead_steps or self.cfg.t_nowcast
        frames = self.validate_satellite_stack(cloud_index_stack)
        if frames.ndim != 3 or frames.shape[0] < 3:
            raise ValueError("Need a (T_past>=3, H, W) cloud-index stack")
        vel = self.estimate_motion(frames)
        adv = self.advect(frames, vel, n)
        return np.clip(adv, 0.0, 1.2), vel

    def hub_ghi_nowcast(self, advected_cloud_index: np.ndarray, t0) -> pd.DataFrame:
        """Deterministic 0-6h GHI nowcast per hub: K_c(3x3 mean) x Ineichen GHI."""
        n, h, w = advected_cloud_index.shape
        t0 = pd.Timestamp(t0)
        t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
        times = pd.date_range(
            t0 + pd.Timedelta(minutes=self.cfg.mtu_minutes), periods=n, freq=f"{self.cfg.mtu_minutes}min"
        )
        cols = {}
        for hub in self.cfg.hubs:
            r, c = hub_pixel(hub, (h, w), self.cfg.bbox)
            r0, r1 = max(r - 1, 0), min(r + 2, h)
            c0, c1 = max(c - 1, 0), min(c + 2, w)
            n_hub = advected_cloud_index[:, r0:r1, c0:c1].mean(axis=(1, 2))
            kc = self.physics.clearsky_index(n_hub)
            ghi_cs = self.physics.hub_clearsky(hub.name, times)["ghi"].to_numpy()
            cols[hub.name] = kc * ghi_cs
        return pd.DataFrame(cols, index=times)
