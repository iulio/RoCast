"""RoCast command line interface and end-to-end backtest pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rocast import __version__
from rocast.backtest.predico_evaluator import PredicoEvaluator
from rocast.config import RoCastSettings, get_settings
from rocast.earth2.nvidia_weather_engine import NvidiaEarth2Engine, feature_columns
from rocast.geometry.spatial_polygon import RomaniaPolygonEngine
from rocast.ingestion.entsoe_capacity import DynamicEntsoeCapacityManager
from rocast.ingestion.grid_weather_sampler import (
    RHO_GROUND_MIN,
    RomanianGridWeatherSampler,
    synthetic_satellite_frames,
    wind_power_curve,
)
from rocast.models.polygon_forecaster import PolygonForecaster
from rocast.models.pvnet_romania import CalibratedPinballLoss, PVNetRomaniaFusion
from rocast.physics.clearsky_advection import ClearSkyPhysicsCalculator, PyStepsCloudNowcaster

TARGETS = {"solar_mw": "nwp_solar_mw", "wind_onshore_mw": "nwp_wind_onshore_mw"}
E2_MEAN = np.array([0, 0, 0, 0, 288.0, 100800.0, 18.0, 55500.0] + [0.0] * 8, dtype=np.float32).reshape(1, 16, 1, 1)
E2_STD = np.array(
    [5, 5, 7, 7, 5.0, 300.0, 4.0, 300.0, 0.4, 0.4, 0.6, 0.6, 0.6, 50.0, 0.8, 12.0], dtype=np.float32
).reshape(1, 16, 1, 1)
SAT_SCALE = np.array([1.0, 1.0, 300.0, 250.0], dtype=np.float32).reshape(1, 4, 1, 1)
LEAD_FRAMES = (5, 11, 17, 23)

SatProvider = Callable[[pd.DatetimeIndex], np.ndarray]
E2Provider = Callable[[pd.DatetimeIndex], tuple[np.ndarray, pd.DataFrame]]

log = logging.getLogger("rocast.cli")


@dataclass
class Sources:
    satellite: SatProvider
    earth2: E2Provider


class WindowBuilder:
    """Builds leakage-safe model inputs for a forecast cutoff t0 = series.index[i0]."""

    def __init__(self, cfg: RoCastSettings, series: pd.DataFrame, target: str, sources: Sources) -> None:
        self.cfg, self.series, self.target, self.sources = cfg, series, target, sources
        # Per-timestamp caches: window builds repeat every epoch, but the underlying satellite /
        # Earth-2 / nowcast fields are pure functions of time, so fetch each timestamp only once.
        self._sat_cache: dict[pd.Timestamp, np.ndarray] = {}
        self._e2_cache: dict[pd.Timestamp, np.ndarray] = {}
        self._e2_row_cache: dict[pd.Timestamp, dict[str, float]] = {}
        self._e2_cols: list[str] = []
        self._nowcast_cache: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}
        self.physics = ClearSkyPhysicsCalculator(cfg)
        self.nowcaster = PyStepsCloudNowcaster(cfg)
        self.e2_cols = feature_columns(cfg)
        cs = self.physics.all_hubs_clearsky(series.index)
        self.ghi_cs = np.stack([cs[h.name]["ghi"].to_numpy() for h in cfg.hubs], axis=1)
        self.elev = np.stack([cs[h.name]["apparent_elevation"].to_numpy() for h in cfg.hubs], axis=1)
        self.solar_idx = [i for i, h in enumerate(cfg.hubs) if h.kind == "solar"]
        self.capacity = cfg.solar_capacity_mw if target == "solar_mw" else cfg.wind_capacity_mw
        h = series.index.hour.to_numpy() + series.index.minute.to_numpy() / 60.0
        self.hist_all = np.stack(
            [
                series["solar_mw"] / cfg.solar_capacity_mw,
                series["wind_onshore_mw"] / cfg.wind_capacity_mw,
                series["total_load_mw"] / 10000.0,
                series["net_load_ex_renewables_mw"] / 10000.0,
                np.sin(2 * np.pi * h / 24.0),
                np.cos(2 * np.pi * h / 24.0),
            ],
            axis=1,
        ).astype(np.float32)
        self.hour_feats = np.stack([np.sin(2 * np.pi * h / 24.0), np.cos(2 * np.pi * h / 24.0)], axis=1)

    @property
    def f_hist(self) -> int:
        return self.hist_all.shape[1]

    def _satellite(self, times: pd.DatetimeIndex) -> np.ndarray:
        missing = [t for t in times if t not in self._sat_cache]
        if missing:
            arr = np.asarray(self.sources.satellite(pd.DatetimeIndex(missing)), dtype=np.float32)
            for k, t in enumerate(missing):
                self._sat_cache[t] = arr[k]
        return np.stack([self._sat_cache[t] for t in times])

    def _earth2(self, times: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DataFrame]:
        missing = [t for t in times if t not in self._e2_cache]
        if missing:
            arr, df = self.sources.earth2(pd.DatetimeIndex(missing))
            self._e2_cols = list(df.columns)
            for k, t in enumerate(missing):
                self._e2_cache[t] = np.asarray(arr[k], dtype=np.float32)
                self._e2_row_cache[t] = df.iloc[k].to_dict()
        e2 = np.stack([self._e2_cache[t] for t in times])
        e2_df = pd.DataFrame({c: [self._e2_row_cache[t][c] for t in times] for c in self._e2_cols}, index=times)
        return e2, e2_df

    def _nowcast(self, frames: np.ndarray, last_sat_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
        if last_sat_time not in self._nowcast_cache:
            n_obs = ClearSkyPhysicsCalculator.heliosat_cloud_index(frames[:, 0], RHO_GROUND_MIN)
            adv, _ = self.nowcaster.nowcast(n_obs, self.cfg.t_nowcast)
            ghi_now = self.nowcaster.hub_ghi_nowcast(adv, last_sat_time).to_numpy()
            self._nowcast_cache[last_sat_time] = (adv, ghi_now)
        return self._nowcast_cache[last_sat_time]

    def build(self, i0: int) -> dict[str, np.ndarray]:
        cfg, s = self.cfg, self.series
        tp, tf = cfg.t_past, cfg.t_future
        t0 = s.index[i0]
        fut_idx = s.index[i0 : i0 + tf]
        step = pd.Timedelta(minutes=cfg.mtu_minutes)

        # satellite branch: frames strictly at or before t0 - 15 min
        sat_times = pd.date_range(t0 - step * cfg.t_sat, periods=cfg.t_sat, freq=f"{cfg.mtu_minutes}min")
        assert sat_times[-1] <= t0 - pd.Timedelta(minutes=cfg.satellite_lag_minutes)
        frames = self._satellite(sat_times)  # (T, 4, H, W)
        n_obs = ClearSkyPhysicsCalculator.heliosat_cloud_index(frames[:, 0], RHO_GROUND_MIN)
        adv, ghi_now = self._nowcast(frames, sat_times[-1])  # ghi_now: (24, hubs), starts at t0
        past = np.concatenate([frames / SAT_SCALE, n_obs[:, None].astype(np.float32)], axis=1)
        lead = np.concatenate(
            [
                np.repeat(frames[-1:], len(LEAD_FRAMES), 0) / SAT_SCALE,
                adv[list(LEAD_FRAMES)][:, None].astype(np.float32),
            ],
            axis=1,
        )
        sat = np.concatenate([past, lead], axis=0).transpose(1, 0, 2, 3)  # (5, 8, H, W)

        # Earth-2 branch
        e2, e2_df = self._earth2(fut_idx)
        e2_n = ((e2 - E2_MEAN) / E2_STD).astype(np.float32)
        e2_hub = e2_df[self.e2_cols].to_numpy().reshape(tf, len(cfg.hubs), 4)
        e2_hub = (e2_hub - np.array([0.0, 0.0, 288.0, 18.0])) / np.array([10.0, 2.0, 10.0, 5.0])

        e2_ws = e2_df[[f"{h.name}_ws100m" for h in cfg.hubs]].to_numpy()
        nwp_base = s[TARGETS[self.target]].to_numpy()[i0 : i0 + tf] / self.capacity
        exo_hubs = []
        for j, hub in enumerate(cfg.hubs):
            nowcast = np.zeros(tf, dtype=np.float64)
            nowcast[: ghi_now.shape[0]] = ghi_now[:, j]
            exo_hubs.append(
                np.stack(
                    [
                        self.ghi_cs[i0 : i0 + tf, j] / 1000.0,
                        np.clip(self.elev[i0 : i0 + tf, j], 0, 90) / 90.0,
                        s[f"{hub.name}_shortwave_radiation"].to_numpy()[i0 : i0 + tf] / 1000.0,
                        s[f"{hub.name}_cloud_cover"].to_numpy()[i0 : i0 + tf] / 100.0,
                        wind_power_curve(s[f"{hub.name}_wind_speed_100m"].to_numpy()[i0 : i0 + tf], 1.0),
                        wind_power_curve(e2_ws[:, j], 1.0),
                        nowcast / 1000.0,
                    ],
                    axis=1,
                )
            )
        exo = np.concatenate(
            exo_hubs + [e2_hub.reshape(tf, -1), self.hour_feats[i0 : i0 + tf], nwp_base[:, None]], axis=1
        )

        return {
            "sat": sat.astype(np.float32),
            "hist": self.hist_all[i0 - tp : i0],
            "e2": e2_n,
            "exo": exo.astype(np.float32),
            "elev": np.clip(self.elev[i0 : i0 + tf][:, self.solar_idx].mean(axis=1), -90, 90).astype(np.float32),
            "y": s[self.target].to_numpy()[i0 : i0 + tf].astype(np.float32),
            "nwp": s[TARGETS[self.target]].to_numpy()[i0 : i0 + tf].astype(np.float32),
        }


class WindowDataset(Dataset):
    def __init__(self, builder: WindowBuilder, indices: list[int]) -> None:
        self.builder, self.indices = builder, indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in self.builder.build(self.indices[i]).items()}


def _forward(
    model: PVNetRomaniaFusion,
    batch: dict[str, torch.Tensor],
    solar: bool,
    capacity: float,
    t_nowcast: int | None = None,
) -> torch.Tensor:
    lead_weight = None
    if t_nowcast is not None:
        lead_weight = (torch.arange(batch["exo"].shape[1]) < t_nowcast).to(batch["exo"].dtype)
    return model(
        batch["sat"],
        batch["hist"],
        batch["e2"],
        batch["exo"],
        solar_elevation=batch["elev"] if solar else None,
        nwp_frac=batch["nwp"] / capacity,
        lead_weight=lead_weight,
    )


def train_model(
    builder: WindowBuilder,
    train_idx: list[int],
    epochs: int,
    seed: int,
    batch_size: int = 8,
    lr: float = 3e-3,
    verbose: bool = True,
    t_gate: int | None = None,
) -> PVNetRomaniaFusion:
    torch.manual_seed(seed)
    probe = builder.build(train_idx[0])
    model = PVNetRomaniaFusion(
        f_hist=builder.f_hist,
        f_exo=probe["exo"].shape[1],
        c_sat=probe["sat"].shape[0],
        c_e2=probe["e2"].shape[1],
        d_model=builder.cfg.d_model,
        capacity_ceiling_mw=builder.capacity,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    loss_fn = CalibratedPinballLoss(builder.cfg.quantiles)
    loader = DataLoader(
        WindowDataset(builder, train_idx),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    solar = builder.target == "solar_mw"
    if t_gate is None:
        t_gate = builder.cfg.t_nowcast
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            opt.zero_grad()
            pred = _forward(model, batch, solar, builder.capacity, t_gate)
            loss = loss_fn(pred / builder.capacity, batch["y"] / builder.capacity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch["y"])
            n += len(batch["y"])
        sched.step()
        if verbose:
            log.info("[%s] epoch %d/%d pinball(norm)=%.5f", builder.target, ep + 1, epochs, total / n)
    return model


@torch.no_grad()
def predict(
    model: PVNetRomaniaFusion, builder: WindowBuilder, indices: list[int], t_gate: int | None = None
) -> dict[str, np.ndarray]:
    model.eval()
    batches = list(DataLoader(WindowDataset(builder, indices), batch_size=len(indices)))
    b = batches[0]
    if t_gate is None:
        t_gate = builder.cfg.t_nowcast
    return {
        "q": _forward(model, b, builder.target == "solar_mw", builder.capacity, t_gate).numpy(),
        "y": b["y"].numpy(),
        "nwp": b["nwp"].numpy(),
    }


def synthetic_sources(cfg: RoCastSettings) -> Sources:
    engine = NvidiaEarth2Engine(cfg)
    return Sources(
        satellite=lambda t: synthetic_satellite_frames(t, cfg),
        earth2=lambda t: engine.generate_synthetic_earth2_ensemble(t),
    )


def live_sources(
    cfg: RoCastSettings, sampler: RomanianGridWeatherSampler, seviri_dir: str | None, earth2: str
) -> Sources:
    h, w = cfg.spatial_shape
    files: dict[pd.Timestamp, str] = {}
    if seviri_dir:
        for p in sorted(Path(seviri_dir).glob("*")):
            m = re.search(r"(\d{14})", p.name)
            if m:
                files[pd.Timestamp(m.group(1), tz="UTC").floor("15min")] = str(p)
    else:
        warnings.warn("No --seviri-dir given: satellite branch receives clear-sky (zero cloud) frames.", stacklevel=2)
    engine = NvidiaEarth2Engine(cfg)
    if earth2 == "none":
        warnings.warn("Earth-2 disabled: ensemble branch receives climatological zeros.", stacklevel=2)

    def sat(times: pd.DatetimeIndex) -> np.ndarray:
        if not files:
            f = np.zeros((len(times), 4, h, w), dtype=np.float32)
            f[:, 0] = RHO_GROUND_MIN
            f[:, 1] = RHO_GROUND_MIN
            f[:, 2] = 290.0
            f[:, 3] = 240.0
            return f
        missing = [t for t in times if t not in files]
        if missing:
            raise RuntimeError(f"Missing SEVIRI scans for {missing[:2]}")
        return sampler.fetch_eumetsat_frames([files[t] for t in times])

    def e2(times: pd.DatetimeIndex) -> tuple[np.ndarray, pd.DataFrame]:
        if earth2 == "none":
            t = np.broadcast_to(E2_MEAN, (len(times), 16, h, w)).astype(np.float32).copy()
            cols = {c: np.zeros(len(times)) for c in feature_columns(cfg)}
            for k in cols:
                if k.endswith("_t2m"):
                    cols[k] += 288.0
                if k.endswith("_tcwv"):
                    cols[k] += 18.0
            return t, pd.DataFrame(cols, index=times)
        init = times[0].floor("6h")
        ds = engine.run_earth2studio_fcn3_ensemble(
            [init], n_steps=max(int((times[-1] - init) / pd.Timedelta(hours=6)) + 1, 2)
        )
        return engine.dataset_to_tensor(ds, init, times)

    return Sources(satellite=sat, earth2=e2)


def split_indices(n: int, cfg: RoCastSettings, n_test: int, train_stride: int) -> tuple[list[int], list[int]]:
    last = n - cfg.t_future
    test = sorted(last - 96 * k for k in range(n_test))
    train = list(range(cfg.t_past, test[0] - cfg.t_future + 1, train_stride))
    if not train or test[0] < cfg.t_past:
        raise ValueError("Series too short for the requested walk-forward split")
    return train, test


def run_backtest(
    mode: str = "synthetic",
    epochs: int = 45,
    output_dir: str | Path = "outputs",
    seed: int = 0,
    days: int = 21,
    n_test_windows: int = 5,
    train_stride: int = 4,
    primary_target: str = "solar_mw",
    seviri_dir: str | None = None,
    earth2: str = "none",
    verbose: bool = True,
) -> dict:
    t_start = time.time()
    cfg = get_settings()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = RomanianGridWeatherSampler(cfg)

    if mode == "synthetic":
        start = pd.Timestamp("2025-06-01 00:00", tz="UTC")
        series = sampler.generate_synthetic_series(start, periods=96 * days + 36, seed=seed)
        sources = synthetic_sources(cfg)
    elif mode == "live":
        end = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
        series = sampler.build_live_series(end - pd.Timedelta(days=days), end)
        series = series.asfreq(f"{cfg.mtu_minutes}min").interpolate(limit=cfg.max_interp_gap_mtu).dropna()
        sources = live_sources(cfg, sampler, seviri_dir, earth2)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    train_idx, test_idx = split_indices(len(series), cfg, n_test_windows, train_stride)
    evaluator = PredicoEvaluator(cfg.quantiles)
    report: dict = {
        "mode": mode,
        "epochs": epochs,
        "bidding_zone": cfg.bidding_zone,
        "n_train_windows": len(train_idx),
        "n_test_windows": len(test_idx),
        "horizon_steps": cfg.t_future,
        "targets": {},
    }
    results: dict[str, dict] = {}
    for target in TARGETS:
        if verbose:
            log.info("Training %s on %d windows ...", target, len(train_idx))
        builder = WindowBuilder(cfg, series, target, sources)
        # The satellite nowcast advects clouds (GHI), so the correction gate is the full nowcast horizon
        # for solar; wind carries no nowcast signal, so its gate is short to avoid overfit corrections.
        t_gate = cfg.t_nowcast if target == "solar_mw" else max(cfg.t_nowcast // 3, 1)
        model = train_model(builder, train_idx, epochs, seed, verbose=verbose, t_gate=t_gate)
        pred = predict(model, builder, test_idx, t_gate=t_gate)
        metrics = evaluator.compute_metrics(pred["y"], pred["q"], pred["nwp"])
        report["targets"][target] = metrics
        report["nowcast_backend"] = builder.nowcaster.backend
        last = len(test_idx) - 1
        times = series.index[test_idx[last] : test_idx[last] + cfg.t_future]
        evaluator.export_predico_submission(times, pred["q"][last], out / f"predico_submission_{target}.csv")
        pd.DataFrame(
            {
                "datetime": times,
                "actual_mw": pred["y"][last],
                "nwp_mw": pred["nwp"][last],
                **{f"q{int(q * 100)}": pred["q"][last][:, k] for k, q in enumerate(cfg.quantiles)},
            }
        ).to_csv(out / f"backtest_forecast_{target}.csv", index=False)
        results[target] = {"times": times, "q": pred["q"][last]}

    primary = results[primary_target]
    evaluator.export_predico_submission(primary["times"], primary["q"], out / "predico_submission.csv")
    report["primary_target"] = primary_target
    report["runtime_s"] = round(time.time() - t_start, 1)
    evaluator.save_report(report, out / "backtest_metrics.json")
    return report


def format_table(report: dict) -> str:
    cols = [
        "pinball_loss_mw",
        "crps_mw",
        "mae_p50_mw",
        "rmse_p50_mw",
        "mae_nwp_mw",
        "skill_score_vs_nwp_pct",
        "p10_p90_coverage_pct",
    ]
    heads = ["pinball", "crps", "mae_p50", "rmse_p50", "mae_nwp", "skill_%", "cov_p10p90_%"]
    lines = [
        f"RoCast backtest | zone {report['bidding_zone']} | mode={report['mode']} | "
        f"{report['n_train_windows']} train / {report['n_test_windows']} test windows | "
        f"nowcast={report.get('nowcast_backend')}",
        "-" * 112,
        f"{'target (MW)':<20}" + "".join(f"{c:>13}" for c in heads),
        "-" * 112,
    ]
    for tgt, m in report["targets"].items():
        lines.append(f"{tgt:<20}" + "".join(f"{m[c]:>13.2f}" for c in cols))
    lines.append("-" * 112)
    return "\n".join(lines)


def _run_polygon_forecast(args) -> None:
    """Run a polygon-scoped quantile forecast and write CSV output."""
    cfg = get_settings()

    # --- parse geometry input ---
    if args.geojson:
        geometry_input = json.loads(Path(args.geojson).read_text(encoding="utf-8"))
    elif args.coords:
        geometry_input = json.loads(args.coords)
    else:
        raise ValueError("Either --geojson or --coords must be provided")

    # --- set up components ---
    engine = RomaniaPolygonEngine()
    capacity_manager = DynamicEntsoeCapacityManager(api_key=cfg.entsoe_api_key)
    forecaster = PolygonForecaster(
        polygon_engine=engine,
        capacity_manager=capacity_manager,
    )

    # --- weather grid over Romania bbox ---
    lon_min, lat_min, lon_max, lat_max = cfg.bbox
    h, w = cfg.spatial_shape
    lons = lon_min + (np.arange(w) + 0.5) * (lon_max - lon_min) / w
    lats = lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h

    n_intervals = args.hours * 4
    now = pd.Timestamp.now(tz="UTC").floor("h")
    times = pd.date_range(now, periods=n_intervals, freq="15min")

    # --- generate weather tensors ---
    # Clear-sky GHI (W/m^2) from first hub, tiled across grid
    physics = ClearSkyPhysicsCalculator(cfg)
    cs = physics.all_hubs_clearsky(times)
    ghi_hub = cs[cfg.hubs[0].name]["ghi"].to_numpy(dtype=np.float32)  # (T,)
    ghi = np.broadcast_to(ghi_hub[:, None, None], (n_intervals, h, w)).copy().astype(np.float32)

    # Cloud index from satellite frames (clear-sky fallback = zero cloud)
    sat_t = pd.date_range(now - pd.Timedelta(hours=1), periods=4, freq="15min")
    frames = synthetic_satellite_frames(sat_t, cfg)  # (4, 4, H, W)
    n_obs = ClearSkyPhysicsCalculator.heliosat_cloud_index(frames[:, 0], RHO_GROUND_MIN)  # (4, H, W)
    if n_obs.shape[0] < n_intervals:
        pad = np.full((n_intervals - n_obs.shape[0], h, w), n_obs[-1], dtype=np.float32)
        cloud_index = np.concatenate([n_obs, pad], axis=0).astype(np.float32)
    else:
        cloud_index = n_obs[:n_intervals].astype(np.float32)

    # Synthetic wind speed (m/s) - mild diurnal pattern
    hour = times.hour.to_numpy(dtype=np.float32)
    wind_speed = (5.0 + 2.0 * np.sin(2 * np.pi * (hour - 6) / 24.0))[:, None, None]
    wind_speed = np.broadcast_to(wind_speed, (n_intervals, h, w)).copy().astype(np.float32)

    # --- run forecast ---
    result = forecaster.forecast(
        geometry_input=geometry_input,
        resource=args.resource,
        hours=args.hours,
        lons_1d=lons,
        lats_1d=lats,
        ghi=ghi,
        wind_speed=wind_speed,
        cloud_index=cloud_index,
        capacity_mw=args.capacity_mw,
        forecast_year=args.forecast_year,
    )

    # --- write output ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = forecaster.to_csv(result)
    out_path.write_text(csv_text, encoding="utf-8", newline="")

    print(
        f"Polygon forecast written to {out_path} | "
        f"resource={args.resource} | hours={args.hours} | "
        f"cells={result.grid_cells_inside} | capacity={result.capacity_mw:.1f} MW"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="rocast",
        description="RoCast — probabilistic solar/wind forecasting for the Romanian SEN (10YRO-TEL------P)",
        epilog=(
            "examples:\n"
            "  rocast backtest --mode synthetic --output-dir outputs\n"
            "  rocast backtest --mode live --days 30 --seviri-dir data/seviri --earth2 earth2studio\n"
            "  rocast --version\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"rocast {__version__}")
    p.add_argument("--quiet", action="store_true", help="only warnings and errors on stderr")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backtest", help="walk-forward backtest and Predico export")
    b.add_argument("--mode", choices=["synthetic", "live"], default="synthetic")
    b.add_argument("--epochs", type=int, default=45)
    b.add_argument("--output-dir", default="outputs")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--days", type=int, default=21)
    b.add_argument("--n-test-windows", type=int, default=5)
    b.add_argument("--train-stride", type=int, default=4)
    b.add_argument("--target", choices=list(TARGETS), default="solar_mw", help="target for predico_submission.csv")
    b.add_argument("--seviri-dir", default=None, help="live: directory of SEVIRI native files")
    b.add_argument("--earth2", choices=["none", "earth2studio"], default="none", help="live: Earth-2 backend")

    pf = sub.add_parser(
        "polygon-forecast",
        help="polygon-scoped quantile forecast (dynamic geometry + ENTSO-E capacity)",
    )
    pf.add_argument("--geojson", default=None, help="path to a GeoJSON polygon file")
    pf.add_argument(
        "--coords",
        default=None,
        help='JSON list of [lon, lat] pairs, e.g. "[[27.5,44.2],[28.8,44.2],[28.8,45.2],[27.5,45.2],[27.5,44.2]]"',
    )
    pf.add_argument("--resource", choices=["solar", "wind"], default="solar", help="resource type")
    pf.add_argument("--hours", type=int, default=48, help="forecast horizon in hours")
    pf.add_argument("--output", default="outputs/polygon_forecast.csv", help="output CSV path")
    pf.add_argument(
        "--capacity-mw",
        type=float,
        default=None,
        help="user-specified installed capacity (MW; overrides ENTSO-E lookup)",
    )
    pf.add_argument(
        "--forecast-year",
        type=int,
        default=None,
        help="year for ENTSO-E installed capacity lookup",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        if args.command == "polygon-forecast":
            _run_polygon_forecast(args)
        else:
            report = run_backtest(
                mode=args.mode,
                epochs=args.epochs,
                output_dir=args.output_dir,
                seed=args.seed,
                days=args.days,
                n_test_windows=args.n_test_windows,
                train_stride=args.train_stride,
                primary_target=args.target,
                seviri_dir=args.seviri_dir,
                earth2=args.earth2,
            )
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        log.error("%s", exc)
        return 1
    if args.command != "polygon-forecast":
        print(format_table(report))
        log.info("Artifacts written to %s", Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
