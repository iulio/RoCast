"""Central configuration for RoCast."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class HubCoordinate(BaseModel):
    name: str
    lat: float
    lon: float
    alt: float
    capacity_mw: float
    kind: str  # "wind" or "solar"


def _default_hubs() -> list[HubCoordinate]:
    return [
        HubCoordinate(name="dobrogea_wind", lat=44.50, lon=28.30, alt=85.0, capacity_mw=2650.0, kind="wind"),
        HubCoordinate(name="moldova_wind", lat=45.80, lon=27.80, alt=140.0, capacity_mw=750.0, kind="wind"),
        HubCoordinate(name="wallachia_solar", lat=44.30, lon=25.50, alt=90.0, capacity_mw=1400.0, kind="solar"),
        HubCoordinate(name="oltenia_solar", lat=44.10, lon=23.80, alt=115.0, capacity_mw=850.0, kind="solar"),
        HubCoordinate(name="transylvania_solar", lat=46.50, lon=24.50, alt=340.0, capacity_mw=600.0, kind="solar"),
    ]


class RoCastSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROCAST_", env_file=".env", extra="ignore")

    bidding_zone: str = "10YRO-TEL------P"
    timezone: str = "Europe/Bucharest"
    mtu_minutes: int = 15
    hubs: list[HubCoordinate] = _default_hubs()
    bbox: tuple[float, float, float, float] = (20.0, 43.5, 30.0, 48.5)  # lon_min, lat_min, lon_max, lat_max
    spatial_shape: tuple[int, int] = (24, 40)
    eumetsat_channels: list[str] = ["VIS006", "VIS008", "IR_108", "WV_062"]
    earth2_variables: list[str] = ["u10m", "v10m", "u100m", "v100m", "t2m", "sp", "tcwv", "z500"]
    max_satellite_nan_ratio: float = 0.05
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    t_past: int = 96
    t_sat: int = 4
    t_nowcast: int = 24
    t_future: int = 192
    satellite_lag_minutes: int = 15
    max_interp_gap_mtu: int = 4
    d_model: int = 32
    solar_capacity_mw: float = 2850.0
    wind_capacity_mw: float = 3550.0

    entsoe_api_key: str = ""
    ngc_api_key: str = ""
    earth2_nim_url: str = "http://localhost:8000/v1/infer"
    openmeteo_url: str = "https://api.open-meteo.com/v1/forecast"


def get_settings() -> RoCastSettings:
    return RoCastSettings()
