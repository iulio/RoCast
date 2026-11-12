import numpy as np
import pandas as pd
import pytest

from rocast.config import get_settings
from rocast.ingestion.grid_weather_sampler import RHO_GROUND_MIN, synthetic_satellite_frames
from rocast.physics.clearsky_advection import ClearSkyPhysicsCalculator, PyStepsCloudNowcaster, hub_pixel


def test_heliosat_cloud_index_bounds():
    rho = np.random.default_rng(0).uniform(-0.2, 1.5, (24, 40))
    n = ClearSkyPhysicsCalculator.heliosat_cloud_index(rho, 0.08)
    assert n.min() >= 0.0 and n.max() <= 1.2
    kc = ClearSkyPhysicsCalculator.clearsky_index(n)
    assert kc.min() >= 0.05 and kc.max() <= 1.05


def test_clearsky_zero_at_midnight_utc():
    calc = ClearSkyPhysicsCalculator()
    times = pd.date_range("2025-06-21 00:00", periods=1, freq="15min", tz="UTC")
    for hub in get_settings().hubs:
        assert calc.hub_clearsky(hub.name, times)["ghi"].iloc[0] == 0.0
    noon = calc.hub_clearsky("wallachia_solar", pd.date_range("2025-06-21 10:00", periods=1, tz="UTC"))
    assert noon["ghi"].iloc[0] > 600.0


def test_nan_gate_and_imputation():
    nc = PyStepsCloudNowcaster()
    stack = np.random.default_rng(1).uniform(0, 1, (4, 24, 40))
    bad = stack.copy()
    bad[np.random.default_rng(2).random(bad.shape) < 0.10] = np.nan
    with pytest.raises(ValueError):
        nc.validate_satellite_stack(bad)
    ok = stack.copy()
    ok[0, :3, :3] = np.nan
    assert not np.isnan(nc.validate_satellite_stack(ok)).any()


def test_advection_shape_and_motion():
    times = pd.date_range("2025-06-02 10:00", periods=4, freq="15min", tz="UTC")
    n = ClearSkyPhysicsCalculator.heliosat_cloud_index(synthetic_satellite_frames(times)[:, 0], RHO_GROUND_MIN)
    nc = PyStepsCloudNowcaster()
    adv, vel = nc.nowcast(n, 24)
    assert adv.shape == (24, 24, 40) and vel.shape == (2, 24, 40)
    assert 0.0 <= adv.min() and adv.max() <= 1.2
    assert vel[0].mean() > 0.15  # eastward drift is recovered (true value 0.4 px/step)
    ghi = nc.hub_ghi_nowcast(adv, times[-1])
    assert ghi.shape == (24, 5) and (ghi >= 0).all().all()


def test_hub_pixel_inside_grid():
    cfg = get_settings()
    for hub in cfg.hubs:
        r, c = hub_pixel(hub, cfg.spatial_shape, cfg.bbox)
        assert 0 <= r < 24 and 0 <= c < 40
