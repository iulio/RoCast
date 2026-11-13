"""Archive inputs and chronological forecast calibration for the LinkedIn chart."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from pvlib import irradiance
from scipy.integrate import quad_vec
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls
from scipy.special import ndtr

from rocast.ingestion.grid_weather_sampler import wind_power_curve
from rocast.physics.clearsky_advection import ClearSkyPhysicsCalculator

START = pd.Timestamp("2025-06-15", tz="UTC")
TEST_START = pd.Timestamp("2025-06-20 08:00", tz="UTC")
TEST_END = pd.Timestamp("2025-06-22 08:00", tz="UTC")
ARCHIVE_END = pd.Timestamp("2025-06-23", tz="UTC")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_FILE = "romania_era5_archive_20250615_20250622.json"
ENTSOE_FILE = "entsoe_ro_20250615_20250622.csv"
VARIABLES = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "wind_speed_100m",
    "wind_direction_100m",
    "temperature_2m",
)
TARGETS = ("solar_mw", "wind_onshore_mw")
QUANTILE_COLUMNS = ["q10", "q25", "q50", "q75", "q90"]
NORMAL_OFFSETS = np.array([-1.2816, -0.6745, 0.0, 0.6745, 1.2816])
INTRADAY_LEAD = pd.Timedelta(hours=2)
SPLINE_LOOKAHEAD = pd.Timedelta(hours=2)
SOLAR_FEATURES = (*VARIABLES[:3], "cloud_cover")


def _archive_frames(payload, points) -> list[pd.DataFrame]:
    expected = pd.date_range(START, ARCHIVE_END, freq="h", inclusive="left")
    if not isinstance(payload, list) or len(payload) != len(points):
        raise ValueError(f"Expected {len(points)} locations in the ERA5 archive")
    frames = []
    for index, (location, (lat, lon)) in enumerate(zip(payload, points, strict=True)):
        if (
            location.get("location_id", 0) != index
            or abs(location["latitude"] - lat) > 0.13
            or abs(location["longitude"] - lon) > 0.13
        ):
            raise ValueError(f"Unexpected ERA5 grid cell for requested location {index}")
        units = location["hourly_units"]
        if location["utc_offset_seconds"] != 0 or units["wind_speed_100m"] != "m/s":
            raise ValueError("ERA5 archive must use UTC and m/s")
        hourly = location["hourly"]
        frame = pd.DataFrame({var: hourly[var] for var in VARIABLES}, index=pd.to_datetime(hourly["time"], utc=True))
        if not frame.index.equals(expected) or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Incomplete ERA5 hourly data")
        frames.append(frame)
    return frames


def load_weather(data_dir: Path, cfg):
    """Return hub series and a north-to-south 1-degree grid, preserving the raw API bytes."""
    west, south, east, north = cfg.bbox
    lons = np.linspace(west, east, 11)
    lats = np.linspace(north, south, 6)
    points = [(hub.lat, hub.lon) for hub in cfg.hubs]
    points += [(lat, lon) for lat in lats for lon in lons]
    path = data_dir / ARCHIVE_FILE
    cached = path.exists()
    if cached:
        raw = path.read_bytes()
    else:
        query = urlencode(
            {
                "latitude": ",".join(str(lat) for lat, _ in points),
                "longitude": ",".join(str(lon) for _, lon in points),
                "start_date": START.date(),
                "end_date": (ARCHIVE_END - pd.Timedelta(days=1)).date(),
                "hourly": ",".join(VARIABLES),
                "models": "era5",
                "timezone": "GMT",
                "wind_speed_unit": "ms",
                "cell_selection": "nearest",
            }
        )
        with urlopen(f"{ARCHIVE_URL}?{query}", timeout=120) as response:
            raw = response.read()
    payload = json.loads(raw)
    frames = _archive_frames(payload, points)
    if not cached:
        _write_cache(path, raw)
    hubs = {hub.name: frame for hub, frame in zip(cfg.hubs, frames[: len(cfg.hubs)], strict=True)}
    return hubs, (lons, lats, frames[len(cfg.hubs) :])


def _write_cache(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def load_entsoe(data_dir: Path, cfg) -> pd.DataFrame | None:
    """Use cached A75 totals, or fetch them without printing credential-bearing errors."""
    path = data_dir / ENTSOE_FILE
    if path.exists():
        frame = pd.read_csv(path, index_col="datetime", parse_dates=["datetime"])
    elif cfg.entsoe_api_key.strip():
        from entsoe import EntsoePandasClient
        from requests import HTTPError, RequestException

        client = EntsoePandasClient(api_key=cfg.entsoe_api_key, timeout=25, retry_count=1)
        try:
            generation = client.query_generation("RO", start=START, end=ARCHIVE_END)
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 401:
                print("ENTSO-E rejected the configured key (HTTP 401). Using the ERA5-derived benchmark.")
                return None
            raise RuntimeError(f"ENTSO-E A75 request failed (HTTP {status})") from None
        except RequestException as exc:
            raise RuntimeError(f"ENTSO-E A75 request failed ({type(exc).__name__})") from None
        if isinstance(generation.columns, pd.MultiIndex):
            generation = generation.xs("Actual Aggregated", level=1, axis=1)
        frame = generation[["Solar", "Wind Onshore"]].copy()
        frame.columns = list(TARGETS)
    else:
        return None

    frame = frame.loc[:, list(TARGETS)].tz_convert("UTC").sort_index()
    frame = frame.loc[(frame.index >= START) & (frame.index < ARCHIVE_END)]
    if not path.exists():
        _mtu(frame, START, TEST_START)
        _mtu(frame, TEST_START, TEST_END)
        _write_cache(path, frame.to_csv(index_label="datetime").encode("utf-8"))
    return frame


def _mtu(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Interpolate within one partition, holding its last observation for at most 45 min."""
    times = pd.date_range(start, end, freq="15min", inclusive="left")
    observed = frame.loc[(frame.index >= start) & (frame.index < end)]
    result = observed.resample("15min").mean().reindex(times).interpolate(limit=3)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"Incomplete archive data in [{start}, {end})")
    return result


def _era5_generation(cfg, hubs, start, end):
    physics = ClearSkyPhysicsCalculator(cfg)
    times = pd.date_range(start, end, freq="15min", inclusive="left")
    reference = pd.DataFrame(0.0, index=times, columns=list(TARGETS))
    clear_sky = pd.Series(0.0, index=times)
    for hub in cfg.hubs:
        weather = _mtu(hubs[hub.name], start, end)
        if hub.kind == "wind":
            reference["wind_onshore_mw"] += wind_power_curve(weather.wind_speed_100m, hub.capacity_mw)
            continue
        cs = physics.hub_clearsky(hub.name, times)
        position = physics.locations[hub.name].get_solarposition(times)
        daylight = cs.apparent_elevation > 0
        for ghi, dni, dhi, archived in (
            (weather.shortwave_radiation, weather.direct_normal_irradiance, weather.diffuse_radiation, True),
            (cs.ghi, cs.dni, cs.dhi, False),
        ):
            plane = irradiance.get_total_irradiance(30, 180, position.apparent_zenith, position.azimuth, dni, ghi, dhi)[
                "poa_global"
            ]
            power = hub.capacity_mw * (0.85 * plane / 1000).clip(0, 1).where(daylight, 0)
            if archived:
                reference["solar_mw"] += power
            else:
                clear_sky += power
    return reference, clear_sky


def _lagged_weather(hourly: pd.Series | pd.DataFrame, times: pd.DatetimeIndex) -> np.ndarray:
    # PCHIP can use the two hourly knots after the sampled time. Lag the sample
    # accordingly so all influencing knots precede or equal the forecast issue time.
    sample = times - INTRADAY_LEAD - SPLINE_LOOKAHEAD
    knots = (hourly.index - START).total_seconds().to_numpy() / 3600
    coordinates = (sample - START).total_seconds().to_numpy() / 3600
    return PchipInterpolator(knots, hourly.to_numpy(), extrapolate=False)(coordinates)


def _persistence(weather_power: pd.DataFrame, clear_sky: pd.Series) -> pd.DataFrame:
    """Persist recent ERA5 power at the same issue time as the calibrated forecast."""
    times = weather_power.index
    hourly = weather_power.loc[times.minute == 0]
    hourly_clear = clear_sky.reindex(hourly.index)
    clearness = hourly["solar_mw"].div(hourly_clear.where(hourly_clear > 50)).fillna(0.75)
    return pd.DataFrame(
        {
            "solar_mw": _lagged_weather(clearness, times) * clear_sky,
            "wind_onshore_mw": _lagged_weather(hourly["wind_onshore_mw"], times),
        },
        index=times,
    )


def _weather_features(cfg, hubs, times: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    sample = times - INTRADAY_LEAD - SPLINE_LOOKAHEAD
    physics = ClearSkyPhysicsCalculator(cfg)
    features = {target: [] for target in TARGETS}
    for hub in cfg.hubs:
        solar = hub.kind == "solar"
        columns = list(SOLAR_FEATURES) if solar else ["wind_speed_100m"]
        hourly = hubs[hub.name][columns]
        values = np.maximum(_lagged_weather(hourly, times), 0.0)
        if solar:
            sun = physics.hub_clearsky(hub.name, sample).apparent_elevation.to_numpy() > 0
            values[~sun, :3] = 0.0
            values[:, 3] = np.minimum(values[:, 3], 100.0)
        features["solar_mw" if solar else "wind_onshore_mw"].append(values)

    phase = 2 * np.pi * (times.hour.to_numpy() + times.minute.to_numpy() / 60) / 24
    clock = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)])
    predictors = {}
    for target, hub_values in features.items():
        weather = np.column_stack(hub_values)
        predictors[target] = np.column_stack([weather, weather**2, clock])
    return predictors


def _ridge_predict(features, target, envelope, fit, regularization):
    mean = features[fit].mean(axis=0)
    scale = np.maximum(features[fit].std(axis=0), 1e-8)
    design = np.column_stack([np.ones(len(features)), (features - mean) / scale]) * envelope[:, None]
    x = design[fit]
    penalty = np.eye(x.shape[1]) * regularization * fit.sum()
    penalty[0, 0] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ target[fit])
    return design @ weights


def _validation_folds(times, training):
    # Choose regularization and estimate spread from chronological validation days.
    folds = []
    for days_before_test in (3, 2, 1):
        start = TEST_START - pd.Timedelta(days=days_before_test)
        folds.append(
            (
                training & (times < start - INTRADAY_LEAD),
                training & (times >= start) & (times < start + pd.Timedelta(days=1)),
            )
        )
    return folds


def _calibrate_model(features, target, envelope, times, *, anchor=None, cloud=None):
    # Freeze the model before the first test forecast is issued.
    first_issue = TEST_START - INTRADAY_LEAD
    training = (times < first_issue) & np.isfinite(features).all(axis=1) & (envelope > 0)
    folds = _validation_folds(times, training)
    best_error = np.inf
    for strength in (0.03, 0.1, 0.3, 1.0):
        residuals = []
        predictions = []
        for fit, validation in folds:
            prediction = _ridge_predict(features, target, envelope, fit, strength)
            residuals.append(target[validation] - prediction[validation])
            predictions.append(prediction[validation])
        residual = np.concatenate(residuals)
        error = np.abs(residual).mean()
        if error < best_error:
            best_error, regularization, calibration_residuals = error, strength, residual
            calibration_predictions = np.concatenate(predictions)
    center = _ridge_predict(features, target, envelope, training, regularization)
    sigma = calibration_residuals.std(ddof=1)
    if cloud is not None:
        validation = np.logical_or.reduce([mask for _, mask in folds])
        validation_center = calibration_predictions + anchor[validation]
        observed = target[validation] + anchor[validation]
        error = np.abs(observed - np.maximum(validation_center, 0))
        design = _spread_features(validation_center, cloud[validation])
        weights = nnls(design, error)[0]
        # Use a conservative validation quantile for the short calibration record.
        scale = np.quantile(error / np.maximum(design @ weights, 1), 0.95) / NORMAL_OFFSETS[-1]
        sigma = np.maximum(_spread_features(center + anchor, cloud) @ weights * scale, 1)
    return center, sigma, regularization, times[training].max()


def _spread_features(center, cloud):
    return np.column_stack([np.ones(len(center)), np.maximum(center, 0) / 1000, cloud])


def _solar_day_ahead(power, clear_sky, training_target):
    """Use the previous day's irradiance as the archive-only day-ahead proxy."""
    times = power.index
    hourly = power.loc[times.minute == 0, "solar_mw"]
    hours = (times - START).total_seconds().to_numpy() / 3600
    prior_day = PchipInterpolator(hours[times.minute == 0], hourly, extrapolate=False)(hours - 24)
    prior_day = np.where(clear_sky > 0, prior_day, 0)
    target = training_target.reindex(times).to_numpy()
    valid = (times < TEST_START - INTRADAY_LEAD) & (clear_sky > 0) & np.isfinite(prior_day)

    def fit_scale(mask):
        return (prior_day[mask] @ target[mask]) / (prior_day[mask] @ prior_day[mask])

    return pd.Series(prior_day, index=times), fit_scale(valid) * prior_day


def _crps(actual: np.ndarray, center: np.ndarray, sigma: float | np.ndarray, active: np.ndarray) -> float:
    """Integrate the censored Gaussian CDF on either side of each observation."""
    score = actual.astype(float, copy=True)
    spread = np.broadcast_to(sigma, actual.shape)
    deterministic = active & (spread == 0)
    score[deterministic] = np.abs(actual[deterministic] - np.maximum(center[deterministic], 0))
    probabilistic = active & (spread > 0)
    y, mu, sd = actual[probabilistic], center[probabilistic], spread[probabilistic]
    tail = np.maximum(mu + 10 * sd - y, 0)
    below, _ = quad_vec(lambda t: y * ndtr((t * y - mu) / sd) ** 2, 0, 1)
    above, _ = quad_vec(lambda t: tail * ndtr((mu - y - t * tail) / sd) ** 2, 0, 1)
    score[probabilistic] = below + above
    return float(score.mean())


def build_forecasts(cfg, hubs, telemetry):
    """Fit residual bias and spread before TEST_START, then score 192 unseen targets."""
    train_weather, train_clear = _era5_generation(cfg, hubs, START, TEST_START)
    test_weather, test_clear = _era5_generation(cfg, hubs, TEST_START, TEST_END)
    train_target = train_weather if telemetry is None else _mtu(telemetry, START, TEST_START)
    test_target = test_weather if telemetry is None else _mtu(telemetry, TEST_START, TEST_END)
    clear_sky = pd.concat([train_clear, test_clear])
    times = clear_sky.index
    test = times >= TEST_START
    weather_power = pd.concat([train_weather, test_weather])
    baseline = _persistence(weather_power, clear_sky)
    raw_day_ahead, baseline["solar_mw"] = _solar_day_ahead(weather_power, clear_sky, train_target["solar_mw"])
    predictors = _weather_features(cfg, hubs, times)
    solar_anchor = np.zeros(len(times))
    solar_hubs = [hub.name for hub in cfg.hubs if hub.kind == "solar"]
    wallachia = solar_hubs.index("wallachia_solar") if "wallachia_solar" in solar_hubs else 0
    cloud = predictors["solar_mw"][:, wallachia * len(SOLAR_FEATURES) + 3] / 100
    if telemetry is not None:
        # Use only ENTSO-E samples before the issue time. Read raw observations
        # here so interpolation of verifying targets cannot leak into predictors.
        issue = times - INTRADAY_LEAD
        recent_solar = telemetry["solar_mw"].reindex(issue).to_numpy()
        previous_solar = telemetry["solar_mw"].reindex(issue - pd.Timedelta(minutes=15)).to_numpy()
        earlier_solar = telemetry["solar_mw"].reindex(issue - pd.Timedelta(minutes=30)).to_numpy()
        issue_clear = clear_sky.reindex(issue).to_numpy()
        clearness = np.full(len(times), 0.65)
        available = np.isfinite(recent_solar) & (issue_clear > 500)
        np.divide(recent_solar, issue_clear, out=clearness, where=available)
        # Start from the observed attenuation projected onto target-time solar
        # geometry. The ridge model learns corrections to this physical estimate.
        solar_anchor = clearness * clear_sky.to_numpy()
        lagged_nwp = raw_day_ahead.reindex(issue).to_numpy()
        ratio = np.where(
            available & np.isfinite(lagged_nwp),
            np.clip((recent_solar + 50) / (lagged_nwp + 50), 0.35, 1.15),
            0.75,
        )
        physics = ClearSkyPhysicsCalculator(cfg)
        elevation = physics.locations[solar_hubs[wallachia]].get_solarposition(times)["apparent_elevation"]
        trend = np.nan_to_num(recent_solar - earlier_solar) * np.sin(np.deg2rad(elevation.clip(lower=0).to_numpy()))
        predictors["solar_mw"] = np.column_stack(
            [
                predictors["solar_mw"],
                np.nan_to_num(recent_solar),
                np.nan_to_num(previous_solar),
                trend,
                ratio,
                clearness,
                np.nan_to_num(raw_day_ahead.to_numpy() * ratio),
            ]
        )
        # Fit cloud cover and disagreement with prior-day irradiance separately.
        cloud = np.column_stack([cloud, np.abs(1 - ratio)])
    forecasts = {}
    for target in TARGETS:
        solar = target == "solar_mw"
        envelope = clear_sky.to_numpy() / cfg.solar_capacity_mw if solar else np.ones(len(times))
        anchor = solar_anchor if solar else np.zeros(len(times))
        train_values = train_target[target].reindex(times).to_numpy() - anchor
        center, sigma, regularization, fit_end = _calibrate_model(
            predictors[target], train_values, envelope, times, anchor=anchor, cloud=cloud if solar else None
        )
        center = center[test] + anchor[test]
        active = envelope[test] > 0
        spread = sigma[test, None] if solar else sigma
        values = np.maximum(center[:, None] + spread * NORMAL_OFFSETS, 0)
        values[~active] = 0
        frame = pd.DataFrame(values, index=test_target.index, columns=QUANTILE_COLUMNS)
        frame["actual_mw"] = test_target[target]
        frame["baseline_mw"] = baseline[target].reindex(frame.index)
        frame.attrs["crps_mw"] = _crps(frame["actual_mw"].to_numpy(), center, np.ravel(spread), active)
        if solar:
            frame.attrs["daylight_coverage_pct"] = 100 * np.mean(
                (frame["actual_mw"].to_numpy()[active] >= values[active, 0])
                & (frame["actual_mw"].to_numpy()[active] <= values[active, -1])
            )
        forecasts[target] = frame.rename_axis("datetime").reset_index()
        print(f"{target}: fit through {fit_end}, ridge={regularization:.2f}")
    return forecasts
