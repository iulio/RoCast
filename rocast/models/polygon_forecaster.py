"""Polygon-aware quantile forecaster for RoCast.

Combines a spatial mask (from RomaniaPolygonEngine) with meteorological tensors
(GHI, wind speed, cloud index) to aggregate only weather cells inside a
user-defined polygon, then scales output quantiles (q10, q25, q50, q75, q90)
to the polygon's installed capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rocast.geometry.spatial_polygon import RomaniaPolygonEngine, create_spatial_mask
from rocast.ingestion.entsoe_capacity import DynamicEntsoeCapacityManager

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


@dataclass
class PolygonForecastResult:
    """Result of a polygon-scoped forecast."""

    quantiles: dict[str, np.ndarray]  # q10, q25, q50, q75, q90 -> (T,)
    capacity_mw: float
    polygon_area_km2: float
    grid_cells_inside: int
    resource: str


class PolygonForecaster:
    """Forecasts generation quantiles scoped to a user-defined polygon inside Romania."""

    def __init__(
        self,
        polygon_engine: RomaniaPolygonEngine | None = None,
        capacity_manager: DynamicEntsoeCapacityManager | None = None,
    ) -> None:
        self.polygon_engine = polygon_engine or RomaniaPolygonEngine()
        self.capacity_manager = capacity_manager or DynamicEntsoeCapacityManager()

    # ------------------------------------------------------------------ public API
    def forecast(
        self,
        geometry_input,
        resource: str,
        hours: int,
        lons_1d: np.ndarray,
        lats_1d: np.ndarray,
        ghi: np.ndarray | None = None,
        wind_speed: np.ndarray | None = None,
        cloud_index: np.ndarray | None = None,
        capacity_mw: float | None = None,
        forecast_year: int | None = None,
    ) -> PolygonForecastResult:
        """Run a polygon-scoped forecast.

        Parameters
        ----------
        geometry_input : list of (lon, lat) | GeoJSON dict | WKT str
            User-defined polygon to forecast over.
        resource : str
            "solar" or "wind".
        hours : int
            Forecast horizon in hours (output will be 15-min intervals).
        lons_1d, lats_1d : np.ndarray
            1D coordinate arrays of the weather grid.
        ghi, wind_speed, cloud_index : np.ndarray, optional
            Weather tensors of shape (T, H, W) or (H, W). Only cells inside the
            polygon are aggregated.
        capacity_mw : float, optional
            User-specified installed capacity (MW). If None, derived from ENTSO-E
            proportional to the polygon's intersected area.
        forecast_year : int, optional
            Year for ENTSO-E capacity lookup.

        Returns
        -------
        PolygonForecastResult
        """
        # 1. Clip the user geometry to Romania's borders
        clip_result = self.polygon_engine.clip_to_romania(geometry_input)
        clipped_poly = clip_result.geometry

        # 2. Build the spatial boolean mask over the weather grid
        mask = create_spatial_mask(clipped_poly, lons_1d, lats_1d)
        grid_cells_inside = int(mask.sum())

        if grid_cells_inside == 0:
            raise ValueError(
                "No weather grid cells fall inside the clipped polygon. "
                "Try a larger polygon or a different location."
            )

        # 3. Determine installed capacity for this polygon
        if capacity_mw is None:
            national_capacity = self.capacity_manager.get_installed_capacity_mw(
                resource, forecast_year
            )
            # Scale national capacity by the fraction of Romania's area covered
            romania_area = self.polygon_engine.romania_polygon.area
            fraction = clip_result.intersection_area / romania_area if romania_area > 0 else 1.0
            capacity_mw = national_capacity * fraction
        capacity_mw = float(capacity_mw)

        # 4. Aggregate weather cells inside the polygon
        n_intervals = hours * 4  # 15-minute intervals
        quantile_values: dict[str, np.ndarray] = {}

        for qi, q in enumerate(QUANTILES):
            quantile_values[f"q{int(q * 100):02d}"] = self._aggregate_to_quantile(
                resource=resource,
                ghi=ghi,
                wind_speed=wind_speed,
                cloud_index=cloud_index,
                mask=mask,
                n_intervals=n_intervals,
                quantile_level=q,
                capacity_mw=capacity_mw,
            )

        return PolygonForecastResult(
            quantiles=quantile_values,
            capacity_mw=capacity_mw,
            polygon_area_km2=clip_result.intersection_area * 111.0 * 111.0,  # rough deg^2 -> km^2
            grid_cells_inside=grid_cells_inside,
            resource=resource,
        )

    # ------------------------------------------------------------------ internal
    @staticmethod
    def _aggregate_to_quantile(
        resource: str,
        ghi: np.ndarray | None,
        wind_speed: np.ndarray | None,
        cloud_index: np.ndarray | None,
        mask: np.ndarray,
        n_intervals: int,
        quantile_level: float,
        capacity_mw: float,
    ) -> np.ndarray:
        """Aggregate weather cells inside the polygon into a single quantile time series.

        This is a simplified deterministic aggregation: for solar, we use GHI
        adjusted by cloud index; for wind, we use wind speed. The result is
        scaled to the polygon's installed capacity.
        """
        # Select the primary weather variable for the resource
        if resource == "solar":
            primary = ghi if ghi is not None else cloud_index
        else:
            primary = wind_speed if wind_speed is not None else ghi

        if primary is None:
            # No weather data available — return a flat baseline scaled to capacity
            return np.full(n_intervals, capacity_mw * 0.1, dtype=float)

        arr = np.asarray(primary, dtype=float)

        # Reduce to (T, H, W) if needed
        if arr.ndim == 2:
            arr = arr[None, :, :]  # (1, H, W)
        if arr.ndim != 3:
            raise ValueError(f"Weather tensor must be 2D or 3D, got shape {arr.shape}")

        T, H, W = arr.shape

        # Apply spatial mask: zero out cells outside the polygon
        masked = arr * mask[None, :, :]  # broadcast mask over time

        # Aggregate: mean over cells inside the polygon
        cell_count = mask.sum()
        if cell_count > 0:
            spatial_mean = masked.sum(axis=(1, 2)) / cell_count  # (T,)
        else:
            spatial_mean = np.zeros(T, dtype=float)

        # Pad or truncate to n_intervals
        if T < n_intervals:
            pad = np.full(n_intervals - T, spatial_mean[-1] if T > 0 else 0.0)
            spatial_mean = np.concatenate([spatial_mean, pad])
        elif T > n_intervals:
            spatial_mean = spatial_mean[:n_intervals]

        # Normalize to [0, 1] capacity factor then scale to MW
        if resource == "solar":
            # GHI in W/m^2 -> capacity factor (rough: 1000 W/m^2 = 1.0 CF)
            cf = np.clip(spatial_mean / 1000.0, 0.0, 1.0)
        else:
            # Wind speed in m/s -> capacity factor (rough cubic scaling, rated at 12 m/s)
            cf = np.clip((spatial_mean / 12.0) ** 3, 0.0, 1.0)

        # Apply quantile spread around the median estimate
        spread_factor = 1.0 + (quantile_level - 0.5) * 0.4  # q10: 0.84x, q90: 1.16x
        values = cf * capacity_mw * spread_factor

        return np.clip(values, 0.0, capacity_mw)

    def to_csv(self, result: PolygonForecastResult, start_time: str | None = None) -> str:
        """Serialize a PolygonForecastResult to CSV text (15-minute intervals)."""
        import pandas as pd

        if start_time is None:
            start_time = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        n_intervals = len(next(iter(result.quantiles.values())))
        times = pd.date_range(start_time, periods=n_intervals, freq="15T")

        rows = []
        for i, ts in enumerate(times):
            row = {"timestamp": ts.isoformat()}
            for q_name, values in result.quantiles.items():
                row[q_name] = round(float(values[i]), 3)
            row["capacity_mw"] = round(result.capacity_mw, 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
