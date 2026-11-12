"""Live (ENTSO-E, Open-Meteo) and synthetic data sources for the Romanian SEN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rocast.config import RoCastSettings, get_settings
from rocast.earth2.nvidia_weather_engine import synthetic_hub_wind_truth
from rocast.physics.clearsky_advection import ClearSkyPhysicsCalculator, hub_pixel

OPENMETEO_VARS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
    "surface_pressure",
]

CLOUD_VELOCITY_PX_PER_HOUR = (1.6, 0.4)  # (x, y) eastward / southward drift in grid pixels
RHO_GROUND_MIN = 0.08


def _utc_index(start, periods: int, freq: str = "15min") -> pd.DatetimeIndex:
    t = pd.Timestamp(start)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return pd.date_range(t, periods=periods, freq=freq)


def _hours(times) -> np.ndarray:
    idx = pd.DatetimeIndex(times)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return ((idx - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float64)


def synthetic_cloud_index(hrs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Advecting Heliosat cloud albedo index n(x, y, t) in [0, 1]; hrs shape (T,), x/y broadcastable to (H, W)."""
    ux, uy = CLOUD_VELOCITY_PX_PER_HOUR
    t = np.asarray(hrs, dtype=np.float64).reshape((-1,) + (1,) * np.ndim(x))
    xs = x[None] - ux * t
    ys = y[None] - uy * t
    f = (
        0.5
        + 0.22 * np.sin(2 * np.pi * (xs / 17.0 + ys / 11.0) + 0.7)
        + 0.20 * np.sin(2 * np.pi * (xs / 9.0 - ys / 23.0) + 2.1)
        + 0.12 * np.sin(2 * np.pi * (t / 37.0 + x[None] / 40.0))
    )
    regime = np.clip(0.65 + 0.6 * np.sin(2 * np.pi * t / 84.0 + 1.0), 0.0, 1.2)
    return np.clip(f * regime, 0.0, 1.0)


def synthetic_satellite_frames(times, settings: RoCastSettings | None = None) -> np.ndarray:
    """EUMETSAT-like frames (T, 4, H, W): VIS006, VIS008 reflectance, IR_108 / WV_062 brightness temperature."""
    cfg = settings or get_settings()
    h, w = cfg.spatial_shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    n = synthetic_cloud_index(_hours(times), xx, yy)
    rho = RHO_GROUND_MIN + n * (0.85 - RHO_GROUND_MIN)
    vis08 = 0.95 * rho + 0.01
    ir = 292.0 - 55.0 * n
    wv = 238.0 + 6.0 * n
    return np.stack([rho, vis08, ir, wv], axis=1).astype(np.float32)


def wind_power_curve(ws: np.ndarray, capacity_mw: float) -> np.ndarray:
    ws = np.asarray(ws, dtype=np.float64)
    frac = np.clip((ws - 3.0) / 9.0, 0.0, 1.0) ** 3
    return capacity_mw * np.where(ws > 25.0, 0.0, frac)


@dataclass
class SyntheticBundle:
    history: pd.DataFrame
    future_truth: pd.DataFrame
    nwp_future: pd.DataFrame
    satellite: np.ndarray  # (T_sat, C_sat, H, W)
    satellite_times: pd.DatetimeIndex


class RomanianGridWeatherSampler:
    def __init__(self, settings: RoCastSettings | None = None) -> None:
        self.cfg = settings or get_settings()
        self.physics = ClearSkyPhysicsCalculator(self.cfg)

    # ------------------------------------------------------------------ live
    def fetch_entsoe_sen_history(self, start_utc, end_utc) -> pd.DataFrame:
        """ENTSO-E A75 (solar, wind onshore) and A65 (load) at 15-minute MTU, gaps <= 4 MTU interpolated."""
        from entsoe import EntsoePandasClient

        if not self.cfg.entsoe_api_key:
            raise RuntimeError("ROCAST_ENTSOE_API_KEY is required for live ENTSO-E ingestion")
        client = EntsoePandasClient(api_key=self.cfg.entsoe_api_key)
        tz = self.cfg.timezone
        start = (
            pd.Timestamp(start_utc).tz_convert(tz)
            if pd.Timestamp(start_utc).tzinfo
            else pd.Timestamp(start_utc, tz="UTC").tz_convert(tz)
        )
        end = (
            pd.Timestamp(end_utc).tz_convert(tz)
            if pd.Timestamp(end_utc).tzinfo
            else pd.Timestamp(end_utc, tz="UTC").tz_convert(tz)
        )
        area = "RO"  # entsoe-py alias of bidding zone 10YRO-TEL------P

        gen = client.query_generation(area, start=start, end=end, psr_type=None)  # A75
        if isinstance(gen.columns, pd.MultiIndex):
            gen = (
                gen.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
                if "Actual Aggregated" in gen.columns.get_level_values(1)
                else gen.droplevel(1, axis=1)
            )
        load = client.query_load(area, start=start, end=end)  # A65
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]

        df = pd.DataFrame(
            {
                "solar_mw": gen.get("Solar"),
                "wind_onshore_mw": gen.get("Wind Onshore"),
                "total_load_mw": load,
            }
        )
        df.index = df.index.tz_convert("UTC")
        df = df.resample(f"{self.cfg.mtu_minutes}min").mean()
        df = df.interpolate(limit=self.cfg.max_interp_gap_mtu, limit_area="inside")
        df["net_load_ex_renewables_mw"] = df["total_load_mw"] - df["solar_mw"] - df["wind_onshore_mw"]
        return df

    def fetch_openmeteo_hub_nwp(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Open-Meteo NWP per hub, upsampled to 15 min. Columns are '{hub}_{variable}'."""
        import requests

        frames = []
        for hub in self.cfg.hubs:
            resp = requests.get(
                self.cfg.openmeteo_url,
                params={
                    "latitude": hub.lat,
                    "longitude": hub.lon,
                    "hourly": ",".join(OPENMETEO_VARS),
                    "start_date": start_date,
                    "end_date": end_date,
                    "timezone": "UTC",
                    "wind_speed_unit": "ms",
                },
                timeout=60,
            )
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
            df = pd.DataFrame({k: hourly[k] for k in OPENMETEO_VARS}, index=pd.to_datetime(hourly["time"], utc=True))
            df["wind_power_proxy"] = np.clip(df["wind_speed_100m"], 0.0, 25.0) ** 3
            df = df.resample(f"{self.cfg.mtu_minutes}min").interpolate("linear")
            df.columns = [f"{hub.name}_{c}" for c in df.columns]
            frames.append(df)
        return pd.concat(frames, axis=1)

    def fetch_eumetsat_frames(self, native_files: list[str], channels: list[str] | None = None) -> np.ndarray:
        """Load SEVIRI HRIT/native files with satpy and resample to the Romanian (H, W) grid -> (T, C, H, W)."""
        import xarray as xr
        from pyresample.geometry import AreaDefinition
        from satpy import Scene

        chans = channels or self.cfg.eumetsat_channels
        lon_min, lat_min, lon_max, lat_max = self.cfg.bbox
        h, w = self.cfg.spatial_shape
        area = AreaDefinition("ro", "Romania", "ro", "EPSG:4326", w, h, (lon_min, lat_min, lon_max, lat_max))
        frames = []
        for f in native_files:
            scn = Scene(reader="seviri_l1b_native", filenames=[f])
            scn.load(chans)
            res = scn.resample(area, resampler="nearest")
            arrs = [np.asarray(xr.DataArray(res[c]).values, dtype=np.float32) for c in chans]
            frames.append(np.stack(arrs))
        return np.stack(frames)

    def nwp_power_baseline(self, nwp: pd.DataFrame) -> pd.DataFrame:
        """Physical power conversion of raw NWP (hub GHI / 100 m wind) used as the benchmark forecast."""
        solar = np.zeros(len(nwp))
        wind = np.zeros(len(nwp))
        for hub in self.cfg.hubs:
            if hub.kind == "solar":
                solar += hub.capacity_mw * 0.85 * nwp[f"{hub.name}_shortwave_radiation"].to_numpy() / 1000.0
            else:
                wind += wind_power_curve(nwp[f"{hub.name}_wind_speed_100m"].to_numpy(), hub.capacity_mw)
        return pd.DataFrame({"nwp_solar_mw": solar, "nwp_wind_onshore_mw": wind}, index=nwp.index)

    def build_live_series(self, start_utc, end_utc) -> pd.DataFrame:
        """Real ENTSO-E telemetry joined with Open-Meteo NWP on the common series schema."""
        sen = self.fetch_entsoe_sen_history(start_utc, end_utc)
        nwp = self.fetch_openmeteo_hub_nwp(
            pd.Timestamp(start_utc).strftime("%Y-%m-%d"), pd.Timestamp(end_utc).strftime("%Y-%m-%d")
        )
        df = sen.join(nwp, how="inner").join(self.nwp_power_baseline(nwp), how="inner")
        return df.dropna()

    # ------------------------------------------------------------- synthetic
    def generate_synthetic_series(self, start_utc, periods: int, seed: int = 0) -> pd.DataFrame:
        """Continuous deterministic SEN truth (solar/wind/load) and NWP covariates on a 15-min grid."""
        cfg = self.cfg
        times = _utc_index(start_utc, periods, f"{cfg.mtu_minutes}min")
        hrs = _hours(times)
        rng = np.random.default_rng(seed)
        h, w = cfg.spatial_shape
        wind_truth = synthetic_hub_wind_truth(times, cfg)
        out: dict[str, np.ndarray] = {}

        def ar1(sd: float, phi: float = 0.97) -> np.ndarray:
            e = rng.normal(0.0, sd * np.sqrt(1 - phi**2), periods)
            x = np.zeros(periods)
            for i in range(1, periods):
                x[i] = phi * x[i - 1] + e[i]
            return x

        solar = np.zeros(periods)
        wind = np.zeros(periods)
        nwp_solar = np.zeros(periods)
        nwp_wind = np.zeros(periods)
        for hub in cfg.hubs:
            r, c = hub_pixel(hub, (h, w), cfg.bbox)
            cs = self.physics.hub_clearsky(hub.name, times)
            n_true = synthetic_cloud_index(hrs, np.array([c + 0.5]), np.array([r + 0.5]))[:, 0]
            n_shift = synthetic_cloud_index(hrs + 1.5, np.array([c + 0.5]), np.array([r + 0.5]))[:, 0]
            regime_mean = 0.5 * n_true.mean() * np.ones(periods)
            n_nwp = np.clip(0.55 * n_shift + 0.45 * regime_mean + 0.05 * ar1(1.0), 0.0, 1.0)
            kc_true = ClearSkyPhysicsCalculator.clearsky_index(n_true)
            kc_nwp = ClearSkyPhysicsCalculator.clearsky_index(n_nwp)
            ghi_true = kc_true * cs["ghi"].to_numpy()
            ghi_nwp = kc_nwp * cs["ghi"].to_numpy()
            ws_true = np.clip(wind_truth[hub.name].to_numpy() + ar1(0.7), 0.0, 30.0)
            ws_nwp = np.clip(0.95 * np.roll(wind_truth[hub.name].to_numpy(), 8) + 0.4 + ar1(0.4), 0.0, 30.0)
            wdir = (300.0 + 45.0 * np.sin(2 * np.pi * hrs / 70.0)) % 360.0

            if hub.kind == "solar":
                solar += hub.capacity_mw * 0.85 * ghi_true / 1000.0
                nwp_solar += hub.capacity_mw * 0.85 * ghi_nwp / 1000.0
            else:
                wind += wind_power_curve(ws_true, hub.capacity_mw)
                nwp_wind += wind_power_curve(ws_nwp, hub.capacity_mw)

            hub_nwp = {
                "shortwave_radiation": ghi_nwp,
                "direct_normal_irradiance": cs["dni"].to_numpy() * (1.0 - 0.9 * n_nwp),
                "diffuse_radiation": cs["dhi"].to_numpy() * (1.0 + 0.8 * n_nwp),
                "cloud_cover": 100.0 * np.clip(n_nwp, 0.0, 1.0),
                "cloud_cover_low": 70.0 * np.clip(n_nwp, 0.0, 1.0),
                "wind_speed_100m": ws_nwp,
                "wind_direction_100m": wdir,
                "temperature_2m": 285.0 - 273.15 + 5.0 * np.sin(2 * np.pi * (hrs - 9.0) / 24.0),
                "surface_pressure": 1000.0 - 0.11 * hub.alt + 0.0 * hrs,
                "wind_power_proxy": np.clip(ws_nwp, 0.0, 25.0) ** 3,
            }
            for k, v in hub_nwp.items():
                out[f"{hub.name}_{k}"] = v

        hour_local = (hrs + 2.0) % 24.0
        load = (
            6600.0
            + 900.0 * np.sin(2 * np.pi * (hour_local - 9.0) / 24.0)
            + 500.0 * np.sin(2 * np.pi * (hour_local - 19.0) / 12.0)
            - 300.0 * (((hrs // 24) % 7) >= 5)
            + ar1(120.0)
        )
        df = pd.DataFrame(out, index=times)
        df.insert(0, "net_load_ex_renewables_mw", load - solar - wind)
        df.insert(0, "total_load_mw", load)
        df.insert(0, "wind_onshore_mw", np.clip(wind, 0.0, cfg.wind_capacity_mw))
        df.insert(0, "solar_mw", np.clip(solar, 0.0, cfg.solar_capacity_mw))
        df["net_load_ex_renewables_mw"] = df["total_load_mw"] - df["solar_mw"] - df["wind_onshore_mw"]
        df["nwp_solar_mw"] = nwp_solar
        df["nwp_wind_onshore_mw"] = np.clip(nwp_wind, 0.0, cfg.wind_capacity_mw)
        return df

    def generate_synthetic_sen_and_nwp(
        self, start_utc, periods_past: int = 96, periods_future: int = 192, seed: int = 0
    ) -> SyntheticBundle:
        """History up to t0, future truth/NWP, and the last T_sat satellite frames (t <= t0 - 15 min)."""
        series = self.generate_synthetic_series(start_utc, periods_past + periods_future, seed)
        hist = series.iloc[:periods_past]
        fut = series.iloc[periods_past:]
        t0 = fut.index[0]
        sat_times = pd.date_range(
            t0 - pd.Timedelta(minutes=self.cfg.satellite_lag_minutes + self.cfg.mtu_minutes * (self.cfg.t_sat - 1)),
            periods=self.cfg.t_sat,
            freq=f"{self.cfg.mtu_minutes}min",
        )
        sat = synthetic_satellite_frames(sat_times, self.cfg)
        nwp_cols = [c for c in series.columns if any(c.endswith(v) for v in OPENMETEO_VARS + ["wind_power_proxy"])]
        return SyntheticBundle(
            history=hist[["solar_mw", "wind_onshore_mw", "total_load_mw", "net_load_ex_renewables_mw"]],
            future_truth=fut[["solar_mw", "wind_onshore_mw", "total_load_mw", "net_load_ex_renewables_mw"]],
            nwp_future=fut[nwp_cols],
            satellite=sat,
            satellite_times=sat_times,
        )
