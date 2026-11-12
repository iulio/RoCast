"""NVIDIA Earth-2 engine: earth2studio ensembles, NIM REST client, and a synthetic offline atmosphere."""

from __future__ import annotations

import io
import re
import tarfile
from typing import Any

import numpy as np
import pandas as pd

from rocast.config import RoCastSettings, get_settings
from rocast.physics.clearsky_advection import hub_pixel

# Romania crop on the 0.25 degree global grid (lat 90 -> -90, lon 0 -> 360)
LAT_IDX = slice(166, 186)
LON_IDX = slice(80, 120)


def _hours(times) -> np.ndarray:
    idx = pd.DatetimeIndex(times)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(hours=1)).to_numpy(
        dtype=np.float64
    )  # hours since epoch


def synthetic_hub_wind_truth(times, settings: RoCastSettings | None = None) -> pd.DataFrame:
    """Deterministic 100 m wind speed (m/s) per hub: synoptic waves + diurnal cycle."""
    cfg = settings or get_settings()
    hrs = _hours(times)
    out = {}
    for i, hub in enumerate(cfg.hubs):
        base = {"dobrogea_wind": 8.5, "moldova_wind": 7.0}.get(hub.name, 5.0)
        ph = 1.7 * i
        syn = (
            2.8 * np.sin(2 * np.pi * hrs / 61.0 + ph)
            + 1.8 * np.sin(2 * np.pi * hrs / 27.0 + 2 * ph)
            + 1.0 * np.sin(2 * np.pi * hrs / 14.0 + 0.5 * ph)
        )
        diurnal = 0.8 * np.sin(2 * np.pi * (hrs - 15.0) / 24.0)
        out[hub.name] = np.clip(base + syn + diurnal, 0.5, 25.0)
    return pd.DataFrame(out, index=pd.DatetimeIndex(times))


def _wind_direction_rad(hrs: np.ndarray) -> np.ndarray:
    return np.deg2rad(300.0 + 45.0 * np.sin(2 * np.pi * hrs / 70.0))  # meteorological "from" direction


class NvidiaEarth2Engine:
    def __init__(self, settings: RoCastSettings | None = None) -> None:
        self.cfg = settings or get_settings()

    # ------------------------------------------------------------------ live
    def run_earth2studio_fcn3_ensemble(
        self,
        init_times: list[str] | list[pd.Timestamp],
        n_steps: int = 8,
        n_ensemble: int = 8,
        noise_amplitude: float = 0.05,
    ):
        """FCN3 ensemble from GFS initial conditions, cropped to Romania. Needs GPU + earth2studio."""
        import xarray as xr
        from earth2studio.data import GFS
        from earth2studio.io import ZarrBackend
        from earth2studio.models.px import FCN3
        from earth2studio.perturbation import SphericalGaussian
        from earth2studio.run import ensemble

        model = FCN3.load_model(FCN3.load_default_package())
        data = GFS()
        perturbation = SphericalGaussian(noise_amplitude=noise_amplitude)
        backend = ZarrBackend()
        times = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in init_times]
        io_backend = ensemble(times, n_steps, n_ensemble, model, perturbation, data, backend)

        root = io_backend.root
        lat = np.asarray(root["lat"][:])
        lon = np.asarray(root["lon"][:])
        coords = {
            "ensemble": np.asarray(root["ensemble"][:]),
            "time": np.asarray(root["time"][:]),
            "lead_time": np.asarray(root["lead_time"][:]),
            "lat": lat,
            "lon": lon,
        }
        dims = ("ensemble", "time", "lead_time", "lat", "lon")
        ds = xr.Dataset(
            {v: (dims, np.asarray(root[v][:])) for v in self.cfg.earth2_variables if v in root}, coords=coords
        )
        lon_min, lat_min, lon_max, lat_max = self.cfg.bbox
        ds = ds.sel(
            lat=slice(lat_max, lat_min) if lat[0] > lat[-1] else slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        if "u100m" in ds and "v100m" in ds:
            ds["ws100m"] = np.hypot(ds["u100m"], ds["v100m"])
        if "u10m" in ds and "v10m" in ds:
            ds["ws10m"] = np.hypot(ds["u10m"], ds["v10m"])
        mean = ds.mean("ensemble")
        spread = ds.std("ensemble").rename({v: f"{v}_spread" for v in ds.data_vars})
        return xr.merge([mean, spread])

    def call_fourcastnet_nim(self, initial_state: np.ndarray, simulation_length: int = 8) -> dict[int, np.ndarray]:
        """POST an initial state (1,1,72,721,1440 float32) to the Earth-2 NIM and parse the tar reply."""
        import httpx

        arr = np.ascontiguousarray(initial_state, dtype=np.float32)
        if arr.shape != (1, 1, 72, 721, 1440):
            raise ValueError(f"initial_state must have shape (1, 1, 72, 721, 1440), got {arr.shape}")
        buf = io.BytesIO()
        np.save(buf, arr)
        headers = {"Authorization": f"Bearer {self.cfg.ngc_api_key}"}
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(
                self.cfg.earth2_nim_url,
                headers=headers,
                files={"input_array": ("input_array", buf.getvalue(), "application/octet-stream")},
                data={"simulation_length": str(simulation_length)},
            )
        resp.raise_for_status()
        return self._parse_nim_tar(resp.content)

    @staticmethod
    def _parse_nim_tar(content: bytes) -> dict[int, np.ndarray]:
        results: dict[int, np.ndarray] = {}
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
            for member in tar.getmembers():
                m = re.match(r"(\d+)_(\d+)\.npy$", member.name.split("/")[-1])
                if not member.isfile() or m is None:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = np.load(io.BytesIO(f.read()))
                results[int(m.group(1))] = data[..., LAT_IDX, LON_IDX]
        return dict(sorted(results.items()))

    # ------------------------------------------------------------- synthetic
    def dataset_to_tensor(self, ds, init_time, target_timestamps) -> tuple[np.ndarray, pd.DataFrame]:
        """Convert a cropped ensemble Dataset (from run_earth2studio_fcn3_ensemble) to (T, 16, H, W) + hub features."""
        h, w = self.cfg.spatial_shape
        lon_min, lat_min, lon_max, lat_max = self.cfg.bbox
        lons = lon_min + (np.arange(w) + 0.5) * (lon_max - lon_min) / w
        lats = lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h
        times = pd.DatetimeIndex(target_timestamps)
        lead_h = ((times - pd.Timestamp(init_time).tz_localize("UTC")) / pd.Timedelta(hours=1)).to_numpy()
        ds = ds.isel(time=0)
        lead_axis = ds["lead_time"].values.astype("timedelta64[h]").astype(float)
        ds = ds.assign_coords(lead_time=lead_axis)
        chans = []
        for suffix in ("", "_spread"):
            for v in self.cfg.earth2_variables:
                da = ds[f"{v}{suffix}"].interp(lead_time=np.clip(lead_h, lead_axis.min(), lead_axis.max()))
                da = da.interp(lat=lats, lon=lons)
                chans.append(da.values)
        tensor = np.stack(chans, axis=1).astype(np.float32)
        cols: dict[str, np.ndarray] = {}
        for hub in self.cfg.hubs:
            r, c = hub_pixel(hub, (h, w), self.cfg.bbox)
            cols[f"{hub.name}_ws100m"] = np.hypot(tensor[:, 2, r, c], tensor[:, 3, r, c])
            cols[f"{hub.name}_ws100m_spread"] = tensor[:, 10, r, c]
            cols[f"{hub.name}_t2m"] = tensor[:, 4, r, c]
            cols[f"{hub.name}_tcwv"] = tensor[:, 6, r, c]
        return tensor, pd.DataFrame(cols, index=times)

    def generate_synthetic_earth2_ensemble(self, target_timestamps) -> tuple[np.ndarray, pd.DataFrame]:
        """Return tensor (T, 16, H, W) [8 means, 8 spreads] and hub-level covariates DataFrame."""
        times = pd.DatetimeIndex(target_timestamps)
        t = len(times)
        h, w = self.cfg.spatial_shape
        lon_min, lat_min, lon_max, lat_max = self.cfg.bbox
        lons = lon_min + (np.arange(w) + 0.5) * (lon_max - lon_min) / w
        lats = lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h
        lon_g, lat_g = np.meshgrid(lons, lats)
        hrs = _hours(times)
        truth = synthetic_hub_wind_truth(times, self.cfg)
        # Synthetic Earth-2 stands in for an NWP ensemble: 2 h lag + smoothing + deterministic wobble,
        # mirroring the series NWP relation. Using the raw truth here would leak the future wind.
        fcst = pd.DataFrame(index=times)
        wobble = 0.3 * np.sin(2 * np.pi * hrs / 13.0)
        for col in truth.columns:
            fcst[col] = np.clip(0.95 * np.roll(truth[col].to_numpy(), 8) + 0.4 + wobble, 0.5, 25.0)

        # blend hub winds onto the grid with Gaussian kernels; add Black Sea / Focsani jet enhancement
        num = np.zeros((t, h, w))
        den = np.zeros((1, h, w))
        for hub in self.cfg.hubs:
            k = np.exp(-(((lon_g - hub.lon) / 1.8) ** 2 + ((lat_g - hub.lat) / 1.2) ** 2))[None]
            num += fcst[hub.name].to_numpy()[:, None, None] * k
            den += k
        ws = num / np.maximum(den, 1e-3)
        weight = np.clip(den, 0, 1)
        ws = ws * weight + 4.0 * (1 - weight)
        sea = 1.0 + 0.25 * np.exp(-(((lon_g - 29.5) / 1.0) ** 2))[None]
        focsani = 1.0 + 0.2 * np.exp(-(((lon_g - 27.3) / 0.7) ** 2 + ((lat_g - 45.7) / 0.6) ** 2))[None]
        ws = ws * sea * focsani

        theta = _wind_direction_rad(hrs)[:, None, None]
        u100 = -ws * np.sin(theta)
        v100 = -ws * np.cos(theta)
        u10, v10 = 0.7 * u100, 0.7 * v100
        diurnal = np.sin(2 * np.pi * (hrs - 9.0) / 24.0)[:, None, None]
        elev = np.zeros((1, h, w)) + 0.0
        t2m = 288.0 + 5.0 * diurnal - 0.55 * (lat_g[None] - 46.0) - 0.002 * elev
        orog = 400.0 * np.exp(-(((lon_g - 25.0) / 1.5) ** 2 + ((lat_g - 46.0) / 1.0) ** 2))[None]
        sp = 101000.0 - 11.0 * orog + 0.0 * ws
        tcwv = 18.0 + 4.0 * np.sin(2 * np.pi * hrs / 90.0)[:, None, None] + 0.0 * ws
        z500 = 55500.0 + 300.0 * np.sin(2 * np.pi * hrs / 120.0)[:, None, None] + 20.0 * (lat_g[None] - 46.0) + 0.0 * ws

        means = np.stack([u10, v10, u100, v100, t2m, sp, tcwv, z500], axis=1)
        lead = (np.arange(t) / max(t - 1, 1))[:, None, None]
        scale = np.array([0.4, 0.4, 0.6, 0.6, 0.6, 60.0, 0.8, 12.0]).reshape(1, 8, 1, 1)
        spreads = (
            scale * (0.4 + 1.2 * lead[:, None]) * (1.0 + 0.3 * np.cos(2 * np.pi * hrs / 24.0)[:, None, None, None])
        )
        spreads = np.broadcast_to(spreads, means.shape)
        tensor = np.concatenate([means, spreads], axis=1).astype(np.float32)

        cols: dict[str, np.ndarray] = {}
        for hub in self.cfg.hubs:
            r, c = hub_pixel(hub, (h, w), self.cfg.bbox)
            ws_hub = np.hypot(means[:, 2, r, c], means[:, 3, r, c])
            cols[f"{hub.name}_ws100m"] = ws_hub
            cols[f"{hub.name}_ws100m_spread"] = spreads[:, 2, r, c]
            cols[f"{hub.name}_t2m"] = means[:, 4, r, c]
            cols[f"{hub.name}_tcwv"] = means[:, 6, r, c]
        return tensor, pd.DataFrame(cols, index=times)


def feature_columns(settings: RoCastSettings | None = None) -> list[str]:
    cfg = settings or get_settings()
    return [f"{h.name}_{s}" for h in cfg.hubs for s in ("ws100m", "ws100m_spread", "t2m", "tcwv")]


__all__: list[Any] = ["NvidiaEarth2Engine", "synthetic_hub_wind_truth", "feature_columns"]
