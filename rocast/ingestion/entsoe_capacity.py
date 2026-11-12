"""Dynamic ENTSO-E installed capacity ingestion for Romania's bidding zone (10YRO-TEL------P)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.mappings import Area

ROOT = Path(__file__).resolve().parents[2]

# Fallback capacities used when ENTSO-E is unreachable (network timeout / no API key).
# These are the last known hardcoded values from rocast/config.py.
FALLBACK_SOLAR_CAPACITY_MW = 2850.0
FALLBACK_WIND_CAPACITY_MW = 3550.0

ROMANIA_BIDZONE = "10YRO-TEL------P"
ENTSO_E_COUNTRY_CODE = Area.RO  # Romania


class DynamicEntsoeCapacityManager:
    """Queries ENTSO-E Document A68 for Romania's installed generation capacity.

    Falls back to hardcoded capacities in rocast/config.py when the ENTSO-E
    API is unreachable (network timeout, missing API key, etc.).
    """

    def __init__(
        self,
        api_key: str | None = None,
        fallback_solar_mw: float = FALLBACK_SOLAR_CAPACITY_MW,
        fallback_wind_mw: float = FALLBACK_WIND_CAPACITY_MW,
    ) -> None:
        self.api_key = api_key
        self.fallback_solar_mw = fallback_solar_mw
        self.fallback_wind_mw = fallback_wind_mw
        self._client: EntsoePandasClient | None = None

    @property
    def client(self) -> EntsoePandasClient:
        if self._client is None:
            self._client = EntsoePandasClient(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------ public API
    def get_installed_capacity_mw(
        self,
        resource: str,
        forecast_year: int | None = None,
    ) -> float:
        """Return installed capacity (MW) for the given resource in Romania.

        Parameters
        ----------
        resource : str
            "solar" or "wind" (maps to ENTSO-E psr_type).
        forecast_year : int | None
            Year for which capacity is requested. Defaults to the current year.

        Returns
        -------
        float
            Installed capacity in MW. Falls back to the configured fallback value
            if ENTSO-E is unreachable.
        """
        if forecast_year is None:
            forecast_year = datetime.now(timezone.utc).year

        try:
            df = self._query_entsoe(resource, forecast_year)
            if df is not None and not df.empty:
                # Take the most recent value within the requested year
                return float(df.iloc[-1].iloc[0])
        except Exception:  # noqa: BLE001 - any network/parse failure triggers fallback
            pass

        return self.fallback_solar_mw if resource == "solar" else self.fallback_wind_mw

    def get_solar_capacity_mw(self, forecast_year: int | None = None) -> float:
        return self.get_installed_capacity_mw("solar", forecast_year)

    def get_wind_capacity_mw(self, forecast_year: int | None = None) -> float:
        return self.get_installed_capacity_mw("wind", forecast_year)

    # ------------------------------------------------------------------ internal
    def _query_entsoe(self, resource: str, forecast_year: int) -> pd.DataFrame | None:
        """Query ENTSO-E Document A68 for installed generation capacity."""
        psr_type = self._resource_to_psr_type(resource)

        start = pd.Timestamp(f"{forecast_year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{forecast_year}-12-31", tz="UTC")

        df = self.client.query_installed_generation_capacity(
            country_code=ENTSO_E_COUNTRY_CODE,
            start=start,
            end=end,
            psr_type=psr_type,
        )
        return df

    @staticmethod
    def _resource_to_psr_type(resource: str) -> str | None:
        """Map RoCast resource names to ENTSO-E psr_type codes."""
        mapping = {
            "solar": "Solar",
            "wind": "Wind Onshore",
        }
        return mapping.get(resource.lower())
