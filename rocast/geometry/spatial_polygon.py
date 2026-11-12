"""Spatial polygon engine for clipping user-defined areas to Romania's sovereign borders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as wkt_loads

ROOT = Path(__file__).resolve().parents[2]
ROMANIA_BOUNDARY_PATH = ROOT / "assets" / "geo" / "romania_boundary.geojson"


@dataclass(frozen=True)
class ClipResult:
    """Result of clipping a user geometry to Romania's borders."""

    geometry: BaseGeometry
    is_valid: bool
    is_entirely_inside: bool
    intersection_area: float


class RomaniaPolygonEngine:
    """Loads Romania's official boundary and clips arbitrary user geometries to it.

    Supports input as:
    * list of (lon, lat) tuples (closed or open ring)
    * GeoJSON feature / geometry dictionaries
    * WKT strings
    """

    def __init__(self, boundary_path: str | Path | None = None) -> None:
        self.boundary_path = Path(boundary_path) if boundary_path else ROMANIA_BOUNDARY_PATH
        self._romania_polygon: BaseGeometry | None = None

    # ------------------------------------------------------------------ boundary
    def _load_romania_boundary(self) -> BaseGeometry:
        """Load and cache Romania's sovereign boundary as a shapely geometry."""
        if self._romania_polygon is not None:
            return self._romania_polygon

        if not self.boundary_path.exists():
            raise FileNotFoundError(
                f"Romania boundary cache not found at {self.boundary_path}. "
                "Run scripts/generate_linkedin_hero.py once to populate it."
            )

        payload = json.loads(self.boundary_path.read_text(encoding="utf-8"))
        feature = payload["features"][0]
        geometry = feature["geometry"]
        geom = shape(geometry)
        self._romania_polygon = geom
        return geom

    @property
    def romania_polygon(self) -> BaseGeometry:
        return self._load_romania_boundary()

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse_input(geometry_input: str | dict | list[tuple[float, float]]) -> BaseGeometry:
        """Parse a user-supplied geometry into a shapely geometry."""
        if isinstance(geometry_input, str):
            # WKT string
            return wkt_loads(geometry_input)

        if isinstance(geometry_input, dict):
            # GeoJSON feature or bare geometry
            if geometry_input.get("type") == "Feature":
                geom_dict = geometry_input.get("geometry")
            else:
                geom_dict = geometry_input
            return shape(geom_dict)

        if isinstance(geometry_input, (list, tuple)):
            # List of (lon, lat) tuples
            coords = [(float(lon), float(lat)) for lon, lat in geometry_input]
            if len(coords) < 3:
                raise ValueError("A polygon needs at least 3 coordinate pairs")
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            return Polygon(coords)

        raise TypeError(
            f"Unsupported geometry input type: {type(geometry_input).__name__}. "
            "Expected a list of (lon, lat) tuples, a GeoJSON dict, or a WKT string."
        )

    # ------------------------------------------------------------------ clipping
    def clip_to_romania(
        self, geometry_input: str | dict | list[tuple[float, float]]
    ) -> ClipResult:
        """Clip a user geometry to Romania's sovereign borders.

        Returns a ClipResult with the intersected geometry. Raises ValueError if
        the input polygon does not overlap Romania at all.
        """
        user_geom = self._parse_input(geometry_input)
        romania = self.romania_polygon

        if not user_geom.is_valid:
            user_geom = user_geom.buffer(0)
        if not romania.is_valid:
            romania = romania.buffer(0)

        intersection = user_geom.intersection(romania)

        if intersection.is_empty or intersection.area <= 0.0:
            raise ValueError(
                "The supplied polygon does not overlap Romania's sovereign territory. "
                "Please provide coordinates within Romania's borders."
            )

        is_entirely_inside = romania.contains(user_geom) or user_geom.equals(intersection)
        intersection_area = float(intersection.area)

        return ClipResult(
            geometry=intersection,
            is_valid=intersection.is_valid,
            is_entirely_inside=is_entirely_inside,
            intersection_area=intersection_area,
        )

    def contains_xy(self, geometry_input: str | dict | list[tuple[float, float]], lon: float, lat: float) -> bool:
        """Return True if the point (lon, lat) lies inside the (clipped) user geometry."""
        result = self.clip_to_romania(geometry_input)
        geometry = result.geometry
        if hasattr(geometry, "contains_point"):
            return geometry.contains_point(lon, lat)
        from shapely.geometry import Point

        return geometry.covers(Point(lon, lat))

    # ------------------------------------------------------------------ spatial mask
    @staticmethod
    def create_spatial_mask(
        clipped_poly: BaseGeometry,
        lons_1d: np.ndarray,
        lats_1d: np.ndarray,
    ) -> np.ndarray:
        """Return a 2D boolean mask (len(lats_1d), len(lons_1d)) of grid cells inside the polygon.

        Parameters
        ----------
        clipped_poly : shapely geometry
            A polygon already clipped to Romania's borders.
        lons_1d : np.ndarray
            1D array of longitudes (x-axis).
        lats_1d : np.ndarray
            1D array of latitudes (y-axis).

        Returns
        -------
        np.ndarray
            Boolean array of shape (len(lats_1d), len(lons_1d)), True where the
            grid cell centre falls inside the polygon.
        """
        from shapely.geometry import Point

        lons = np.asarray(lons_1d, dtype=float)
        lats = np.asarray(lats_1d, dtype=float)

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        points = [Point(x, y) for x, y in zip(lon_grid.ravel(), lat_grid.ravel(), strict=True)]

        mask = np.array([clipped_poly.contains(p) for p in points], dtype=bool)
        return mask.reshape(len(lats), len(lons))


def create_spatial_mask(
    clipped_poly: BaseGeometry,
    lons_1d: np.ndarray,
    lats_1d: np.ndarray,
) -> np.ndarray:
    """Return a 2D boolean mask of grid cells inside ``clipped_poly``.

    Module-level alias for :meth:`RomaniaPolygonEngine.create_spatial_mask`, kept for
    callers that import the helper directly.
    """
    return RomaniaPolygonEngine.create_spatial_mask(clipped_poly, lons_1d, lats_1d)
