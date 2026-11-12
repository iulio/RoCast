"""Tests for the cached Romania boundary geometry used by the LinkedIn hero generator."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rocast.config import get_settings
from scripts.generate_linkedin_hero import (
    BOUNDARY_CACHE,
    CARPATHIANS,
    _build_boundary_patches,
    _extract_romania_geometry,
    _rings_from_geometry,
    load_romania_boundary,
)


def _point_in_ring(lon: float, lat: float, ring: np.ndarray) -> bool:
    """Ray-cast a point against a closed ring."""
    x, y = ring[:, 0], ring[:, 1]
    inside = False
    n = len(x)
    for i in range(n - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        if (y1 > lat) != (y2 > lat):
            x_at = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x_at:
                inside = not inside
    return inside


@pytest.fixture(scope="module")
def rings() -> list[np.ndarray]:
    return load_romania_boundary()


def test_boundary_cache_exists_and_is_romania():
    assert BOUNDARY_CACHE.exists(), "cached Romania boundary GeoJSON is missing"
    payload = json.loads(BOUNDARY_CACHE.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    properties = payload["features"][0]["properties"]
    assert properties["iso_a3"] == "ROU"
    assert properties["name"] == "Romania"
    assert properties["source"] and properties["license"]


def test_rings_are_closed_float_arrays(rings):
    assert rings, "no boundary rings returned"
    for ring in rings:
        assert ring.ndim == 2
        assert ring.shape[1] == 2
        assert ring.shape[0] >= 4
        assert ring.dtype == np.float64
        assert np.array_equal(ring[0], ring[-1]), "rings must be closed"


def test_boundary_is_high_resolution(rings):
    """The mainland outline must be far denser than the old hand-typed 32-point polygon."""
    assert max(len(ring) for ring in rings) > 1000


def test_boundary_bbox_matches_romania(rings):
    all_points = np.vstack(rings)
    lon_min = all_points[:, 0].min()
    lat_min = all_points[:, 1].min()
    lon_max = all_points[:, 0].max()
    lat_max = all_points[:, 1].max()
    assert 20.0 < lon_min < 21.0
    assert 29.0 < lon_max < 30.0
    assert 43.5 < lat_min < 44.0
    assert 48.0 < lat_max < 48.5


def test_boundary_fits_configured_bbox(rings):
    all_points = np.vstack(rings)
    lon_min, lat_min, lon_max, lat_max = get_settings().bbox
    assert all_points[:, 0].min() >= lon_min
    assert all_points[:, 0].max() <= lon_max
    assert all_points[:, 1].min() >= lat_min
    assert all_points[:, 1].max() <= lat_max


def test_hubs_and_carpathian_arc_lie_inside_romania(rings):
    mainland = max(rings, key=len)
    for lon, lat in CARPATHIANS:
        assert _point_in_ring(lon, lat, mainland), f"Carpathian point ({lon}, {lat}) outside Romania"
    for hub in get_settings().hubs:
        assert _point_in_ring(hub.lon, hub.lat, mainland), f"hub {hub.name} outside Romania"


def test_danube_delta_and_black_sea_coast_are_detailed(rings):
    """The delta and the Black Sea shoreline need many vertices, not blocky segments."""
    mainland = max(rings, key=len)
    delta = mainland[(mainland[:, 1] > 44.9) & (mainland[:, 0] > 28.8)]
    coast = mainland[(mainland[:, 1] <= 44.9) & (mainland[:, 1] > 43.6) & (mainland[:, 0] > 28.4)]
    assert len(delta) > 100, "Danube Delta is not detailed enough"
    assert len(coast) > 100, "Black Sea coastline is not detailed enough"


def test_prut_boundary_is_detailed(rings):
    """The eastern Prut river boundary should carry a dense run of vertices."""
    mainland = max(rings, key=len)
    prut = mainland[(mainland[:, 0] > 26.5) & (mainland[:, 1] > 44.0) & (mainland[:, 1] < 48.0)]
    assert len(prut) > 200, "Prut river boundary is not detailed enough"


def test_multipolygon_parts_are_parsed():
    """The loader must handle MultiPolygon input (Sacalin Island is a separate part)."""
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
            [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 2.0]]],
        ],
    }
    polygons = _rings_from_geometry(geometry)
    assert len(polygons) == 2
    assert all(len(rings) == 1 for rings in polygons)
    assert np.array_equal(polygons[0][0][0], polygons[0][0][-1])


def test_polygon_with_hole_is_parsed():
    """Interior rings (holes) must survive parsing so fills do not cover them."""
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
            [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0]],
        ],
    }
    polygons = _rings_from_geometry(geometry)
    assert len(polygons) == 1
    assert len(polygons[0]) == 2


def test_extract_romania_geometry_filters_by_iso_a3():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"ISO_A3": "FRA"}, "geometry": {"type": "Polygon", "coordinates": []}},
            {
                "type": "Feature",
                "properties": {"ISO_A3": "ROU"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            },
        ],
    }
    geometry = _extract_romania_geometry(payload)
    assert geometry["type"] == "Polygon"


def test_extract_romania_geometry_supports_datasets_property_names():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Romania", "ISO3166-1-Alpha-3": "ROU"},
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            }
        ],
    }
    assert _extract_romania_geometry(payload)["type"] == "Polygon"


def test_extract_romania_geometry_raises_without_romania():
    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"ISO_A3": "DEU"}, "geometry": {"type": "Polygon", "coordinates": []}}
        ],
    }
    with pytest.raises(ValueError, match="No Romania feature"):
        _extract_romania_geometry(payload)


def test_boundary_patch_uses_expected_styling(rings):
    from scripts.generate_linkedin_hero import _load_romania_polygons

    fill_patch, outline_patch = _build_boundary_patches(_load_romania_polygons())

    assert outline_patch.get_linewidth() == 2.0
    assert outline_patch.get_edgecolor()[:3] == (0.0, 245 / 255, 212 / 255)
    assert outline_patch.get_facecolor()[3] == 0.0
    assert outline_patch.get_zorder() == 5
    assert outline_patch.get_path_effects()

    assert fill_patch.get_edgecolor()[3] == 0.0
    assert fill_patch.get_facecolor()[:3] == (0.0, 245 / 255, 212 / 255)
    assert fill_patch.get_facecolor()[3] == pytest.approx(0.03)
    assert fill_patch.get_zorder() == 5
    assert not fill_patch.get_path_effects()
