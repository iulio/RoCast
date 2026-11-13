"""Generate the RoCast LinkedIn hero from the Romanian ERA5 archive.

The Romanian outline is rendered from a cached, production-grade vector boundary stored in
``rocast/geometry/romania_boundary.geojson``.  The cache is populated on first run from geoBoundaries
ADM0 (CC BY 3.0) with Natural Earth 1:10m/1:50m fallbacks; every later run is fully offline.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch, Polygon
from scipy.ndimage import zoom

from rocast.backtest.predico_evaluator import PredicoEvaluator
from rocast.config import get_settings
from rocast.geometry.spatial_polygon import ROMANIA_BOUNDARY_PATH
from rocast.physics.clearsky_advection import PyStepsCloudNowcaster
from scripts.hero_archive import (
    INTRADAY_LEAD,
    QUANTILE_COLUMNS,
    TEST_END,
    TEST_START,
    build_forecasts,
    load_entsoe,
    load_weather,
)

BG, PANEL, FG, GRID = "#0b0f1a", "#111827", "#e5e7eb", "#2a3347"
CYAN, CORAL = "#22d3ee", "#fb7185"
BOUNDARY_COLOR = "#00F5D4"

BOUNDARY_CACHE = ROMANIA_BOUNDARY_PATH
BOUNDARY_USER_AGENT = "RoCast-hero-generator/1.0 (LinkedIn hero asset builder)"
BOUNDARY_TIMEOUT = 120.0
CLOUD_PEAK = pd.Timestamp("2025-06-21 11:45", tz="UTC")

# Property keys used by the various public boundary datasets to identify Romania.
_ISO_A3_KEYS = ("ISO_A3", "ADM0_A3", "ISO3166-1-Alpha-3", "shapeISO", "shapeISO3")
_NAME_KEYS = ("ADMIN", "name", "NAME", "shapeName", "NAME_EN")


@dataclass(frozen=True)
class BoundarySource:
    """A candidate upstream source for the Romanian national boundary."""

    label: str
    url: str
    kind: str
    license: str


BOUNDARY_SOURCES: tuple[BoundarySource, ...] = (
    BoundarySource(
        label="geoBoundaries ADM0 (Romania)",
        url="https://www.geoboundaries.org/api/current/gbOpen/ROU/ADM0/",
        kind="geoboundaries_api",
        license="CC BY 3.0 (geoBoundaries / StatSilk, boundary year 2018)",
    ),
    BoundarySource(
        label="geoBoundaries ADM0 (Romania), pinned release",
        url=(
            "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/ROU/ADM0/"
            "geoBoundaries-ROU-ADM0.geojson"
        ),
        kind="geojson",
        license="CC BY 3.0 (geoBoundaries / StatSilk, boundary year 2018)",
    ),
    BoundarySource(
        label="Natural Earth 1:10m admin_0_countries",
        url=(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
            "ne_10m_admin_0_countries.geojson"
        ),
        kind="geojson",
        license="Public domain (Natural Earth)",
    ),
    BoundarySource(
        label="Natural Earth 1:50m admin_0_countries",
        url=(
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
            "ne_50m_admin_0_countries.geojson"
        ),
        kind="geojson",
        license="Public domain (Natural Earth)",
    ),
    BoundarySource(
        label="datasets/geo-countries countries.geojson",
        url="https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson",
        kind="geojson",
        license="ODbL (datasets/geo-countries)",
    ),
)


def _http_get_json(url: str, timeout: float = BOUNDARY_TIMEOUT) -> dict:
    """Fetch and decode a JSON document using only the standard library."""
    request = urllib.request.Request(url, headers={"User-Agent": BOUNDARY_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _feature_is_romania(properties: dict) -> bool:
    """Return True when a GeoJSON feature's properties identify Romania (ROU)."""
    for key in _ISO_A3_KEYS:
        value = properties.get(key)
        if isinstance(value, str) and value.strip().upper() == "ROU":
            return True
    for key in _NAME_KEYS:
        value = properties.get(key)
        if isinstance(value, str) and value.strip().lower() == "romania":
            return True
    return False


def _extract_romania_geometry(payload: dict) -> dict:
    """Return the Romania geometry from a GeoJSON FeatureCollection, Feature or bare geometry."""
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a GeoJSON object, got {type(payload).__name__}")
    geo_type = payload.get("type")
    if geo_type in ("Polygon", "MultiPolygon"):
        return payload
    if geo_type == "Feature":
        geometry = payload.get("geometry")
        if not geometry:
            raise ValueError("GeoJSON Feature carries no geometry")
        return geometry
    if geo_type == "FeatureCollection":
        for feature in payload.get("features") or []:
            properties = feature.get("properties") or {}
            if _feature_is_romania(properties):
                geometry = feature.get("geometry")
                if geometry:
                    return geometry
        raise ValueError("No Romania feature found in the GeoJSON FeatureCollection")
    raise ValueError(f"Unsupported GeoJSON type: {geo_type!r}")


def _ring_to_array(coordinates: list) -> np.ndarray:
    """Convert a GeoJSON linear ring into a closed (N, 2) float array."""
    ring = np.asarray(coordinates, dtype=float)
    if ring.ndim != 2 or ring.shape[0] < 3 or ring.shape[1] < 2:
        raise ValueError("Invalid GeoJSON ring coordinates")
    ring = ring[:, :2]
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[0]])
    return ring


def _rings_from_geometry(geometry: dict) -> list[list[np.ndarray]]:
    """Parse a Polygon/MultiPolygon into one list of rings per polygon (exterior first, then holes)."""
    geo_type = geometry.get("type")
    if geo_type == "Polygon":
        polygons = [geometry.get("coordinates") or []]
    elif geo_type == "MultiPolygon":
        polygons = geometry.get("coordinates") or []
    else:
        raise ValueError(f"Unsupported geometry type: {geo_type!r}")
    parsed: list[list[np.ndarray]] = []
    for polygon in polygons:
        rings = [_ring_to_array(ring) for ring in polygon]
        if rings:
            parsed.append(rings)
    if not parsed:
        raise ValueError("Geometry contains no coordinate rings")
    return parsed


def _download_geometry(source: BoundarySource) -> dict:
    """Download the raw GeoJSON payload advertised by ``source``."""
    if source.kind == "geoboundaries_api":
        metadata = _http_get_json(source.url)
        download_url = metadata.get("gjDownloadURL")
        if not download_url:
            raise RuntimeError("geoBoundaries API response did not contain 'gjDownloadURL'")
        return _http_get_json(download_url)
    return _http_get_json(source.url)


def _write_boundary_cache(source: BoundarySource, geometry: dict) -> None:
    """Persist the Romania-only geometry plus provenance metadata to the local cache."""
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Romania",
                    "iso_a3": "ROU",
                    "source": source.label,
                    "source_url": source.url,
                    "license": source.license,
                    "retrieved": datetime.now(UTC).date().isoformat(),
                },
                "geometry": geometry,
            }
        ],
    }
    BOUNDARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    BOUNDARY_CACHE.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _load_romania_polygons() -> list[list[np.ndarray]]:
    """Return Romania's polygons as ring lists, preferring the local cache over the network."""
    if BOUNDARY_CACHE.exists():
        payload = json.loads(BOUNDARY_CACHE.read_text(encoding="utf-8"))
        return _rings_from_geometry(_extract_romania_geometry(payload))

    failures: list[str] = []
    for source in BOUNDARY_SOURCES:
        try:
            geometry = _extract_romania_geometry(_download_geometry(source))
            polygons = _rings_from_geometry(geometry)
        except Exception as exc:
            failures.append(f"  - {source.label}: {exc}")
            continue
        _write_boundary_cache(source, geometry)
        return polygons
    raise RuntimeError(
        "Could not download the Romania boundary and no cached copy exists at "
        f"{BOUNDARY_CACHE}. Place a Romania GeoJSON there manually, or restore network access.\n" + "\n".join(failures)
    )


def load_romania_boundary() -> list[np.ndarray]:
    """Return the exterior rings of the Romanian national boundary as (N, 2) float arrays."""
    return [rings[0] for rings in _load_romania_polygons()]


def _boundary_attribution() -> str:
    """Return a short credit line describing the boundary source used for the figure."""
    if BOUNDARY_CACHE.exists():
        try:
            payload = json.loads(BOUNDARY_CACHE.read_text(encoding="utf-8"))
            features = payload.get("features") or []
            properties = features[0].get("properties") or {}
            source = properties.get("source")
            license_name = properties.get("license")
            if source:
                return f"Boundary: {source} — {license_name}"
        except (OSError, ValueError, IndexError, AttributeError):
            pass
    return "Boundary: Natural Earth / geoBoundaries"


def _signed_area(ring: np.ndarray) -> float:
    """Return the signed area of a closed ring (positive when counter-clockwise)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _orient_ring(ring: np.ndarray, counter_clockwise: bool) -> np.ndarray:
    """Return the ring wound in the requested direction so compound fills cut holes correctly."""
    area = _signed_area(ring)
    if (area >= 0.0) != counter_clockwise:
        return ring[::-1]
    return ring


def _build_boundary_path(polygons: list[list[np.ndarray]]) -> mpath.Path:
    """Build a compound matplotlib path from Romania's polygons (exteriors plus any holes)."""
    vertices: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for rings in polygons:
        for index, ring in enumerate(rings):
            oriented = _orient_ring(ring, counter_clockwise=index == 0)
            points = oriented[:-1]
            ring_vertices = np.vstack([points, points[0]])
            ring_codes = np.full(len(ring_vertices), mpath.Path.LINETO, dtype=mpath.Path.code_type)
            ring_codes[0] = mpath.Path.MOVETO
            ring_codes[-1] = mpath.Path.CLOSEPOLY
            vertices.append(ring_vertices)
            codes.append(ring_codes)
    return mpath.Path(np.vstack(vertices), np.concatenate(codes))


def _build_boundary_patches(polygons: list[list[np.ndarray]]) -> list[PathPatch]:
    """Create the Romania outline as a subtle fill patch plus a glowing cyan stroke patch.

    The two are kept separate on purpose: ``matplotlib.patheffects.Stroke`` re-draws the
    entire path -- fill included -- with the effect's foreground colour and alpha, so
    applying the glow to a filled patch floods the interior with opaque cyan. The fill
    therefore lives on an effect-free patch and only the outline carries the effects.
    """
    path = _build_boundary_path(polygons)
    fill_patch = PathPatch(
        path,
        facecolor=to_rgba(BOUNDARY_COLOR, 0.03),
        edgecolor="none",
        zorder=5,
    )
    outline_patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=BOUNDARY_COLOR,
        linewidth=2.0,
        zorder=5,
    )
    outline_patch.set_path_effects(
        [
            pe.Stroke(linewidth=5.0, foreground="#000000", alpha=0.55),
            pe.Stroke(linewidth=3.2, foreground=BOUNDARY_COLOR, alpha=0.35),
            pe.Normal(),
        ]
    )
    return [fill_patch, outline_patch]


CARPATHIANS = [
    (24.0, 47.6),
    (25.3, 47.4),
    (26.3, 46.7),
    (26.4, 45.9),
    (25.6, 45.55),
    (24.5, 45.65),
    (23.2, 45.55),
    (22.4, 45.3),
]
SE_WIND_CORRIDOR = [
    (27.10, 45.92),
    (27.70, 46.14),
    (28.36, 45.70),
    (28.95, 44.66),
    (28.62, 44.12),
    (27.94, 44.22),
    (27.49, 45.16),
]
LABELS = {
    "dobrogea_wind": "Dobrogea Wind\n2.65 GW",
    "moldova_wind": "Focșani Gate / Moldova\n0.75 GW",
    "wallachia_solar": "Wallachia PV\n1.40 GW",
    "oltenia_solar": "Oltenia PV\n0.85 GW",
    "transylvania_solar": "Transylvania PV\n0.60 GW",
}


def draw_map(ax, cfg, when: pd.Timestamp, boundary: list[list[np.ndarray]], grid) -> None:
    grid_lons, grid_lats, frames = grid
    shape = (len(grid_lats), len(grid_lons))
    extent = (grid_lons[0], grid_lons[-1], grid_lats[-1], grid_lats[0])
    hourly = pd.date_range(when.floor("h") - pd.Timedelta(hours=3), periods=5, freq="h")
    cloud = np.stack([frame.loc[hourly, "cloud_cover"].to_numpy() for frame in frames], axis=1)
    dense = tuple(np.array(cfg.spatial_shape) / shape)
    cloud = zoom(cloud.reshape((-1, *shape)) / 100, (1, *dense), order=1)
    fraction = (when - when.floor("h")) / pd.Timedelta(hours=1)
    cloud_at_time = (1 - fraction) * cloud[-2] + fraction * cloud[-1]
    speed = np.array(
        [
            (1 - fraction) * frame.at[hourly[-2], "wind_speed_100m"]
            + fraction * frame.at[hourly[-1], "wind_speed_100m"]
            for frame in frames
        ]
    ).reshape(shape)
    im = ax.imshow(
        zoom(speed, dense, order=1),
        extent=extent,
        origin="upper",
        cmap="viridis",
        alpha=0.9,
        aspect="auto",
        vmin=2,
        vmax=16,
    )
    lons = np.linspace(extent[0], extent[1], cloud.shape[2])
    lats = np.linspace(extent[3], extent[2], cloud.shape[1])
    _, vel = PyStepsCloudNowcaster(cfg).nowcast(cloud[:-1], cfg.t_nowcast)
    ax.contour(
        lons,
        lats,
        cloud_at_time,
        levels=[0.35, 0.6, 0.85],
        colors=["#22d3ee", "#a5f3fc", "#ffffff"],
        linestyles="dashed",
        linewidths=1.6,
        alpha=0.35,
        zorder=5,
    )
    s = 2
    u, v = vel[0][::s, ::3], -vel[1][::s, ::3]
    magnitude = np.hypot(u, v)
    factor = np.clip(magnitude, 1.0, 8.0) / np.maximum(magnitude, 1e-3)
    ax.quiver(
        lons[::3],
        lats[::s],
        u * factor,
        v * factor,
        color="white",
        alpha=0.72,
        scale=140,
        width=0.0028,
        headwidth=3.5,
        zorder=4,
    )

    for patch in _build_boundary_patches(boundary):
        ax.add_patch(patch)
    ax.add_patch(Polygon(SE_WIND_CORRIDOR, facecolor=to_rgba(CYAN, 0.15), edgecolor="#a5f3fc", linewidth=1.6, zorder=5))
    ax.text(28.15, 46.17, "SE Wind Corridor", color="#a5f3fc", fontsize=8, zorder=6)
    cx, cy = zip(*CARPATHIANS, strict=False)
    ax.plot(cx, cy, color="#f59e0b", lw=2.2, ls=(0, (6, 3)), alpha=0.9)
    ax.text(24.45, 47.69, "Carpathian Ridge", color="#fbbf24", fontsize=9, ha="center", zorder=6)
    ax.text(29.3, 44.0, "BLACK SEA", color="#93c5fd", fontsize=11, style="italic", ha="center")
    for hub in cfg.hubs:
        wind = hub.kind == "wind"
        ax.scatter(
            hub.lon,
            hub.lat,
            s=140,
            marker="^" if wind else "o",
            c=CYAN if wind else CORAL,
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
        )
        dx = -0.25 if hub.lon > 27 or hub.name == "transylvania_solar" else 0.25
        ax.annotate(
            LABELS[hub.name],
            (hub.lon, hub.lat),
            xytext=(hub.lon + dx, hub.lat + 0.3),
            color="white",
            fontsize=8,
            ha="right" if dx < 0 else "left",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#0b0f1acc", ec=CYAN if wind else CORAL, lw=0.8),
        )
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("ERA5 100 m wind speed (m/s)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])
    ax.set_aspect(1.42)
    ax.set_xlabel("Longitude (°E)", color=FG)
    ax.set_ylabel("Latitude (°N)", color=FG)
    ax.set_title(
        f"A · Romania SEN: ERA5 / Earth-2 wind • Heliosat cloud index • pySTEPS flow ({when:%d %b %H:%M} UTC)",
        color=FG,
        fontsize=9,
        loc="left",
    )
    ax.text(
        0.0,
        -0.135,
        "Shown: ERA5 wind and cloud cover via Open-Meteo · pySTEPS flow from ERA5 cloud cover\n"
        + _boundary_attribution(),
        transform=ax.transAxes,
        color=FG,
        fontsize=6,
        alpha=0.75,
        ha="left",
        va="top",
    )


def draw_fan(ax, df: pd.DataFrame, title: str, color_event: str, event: str, actual_label: str) -> None:
    x = df["datetime"]
    ax.fill_between(x, df["q10"], df["q90"], color=CYAN, alpha=0.18, label="P10–P90")
    ax.fill_between(x, df["q25"], df["q75"], color=CYAN, alpha=0.35, label="P25–P75")
    ax.plot(x, df["q50"], color=CYAN, lw=2.0, label="Calibrated P50")
    ax.plot(x, df["actual_mw"], color="white", lw=1.6, label=actual_label)
    baseline_label = "Day-ahead ERA5 baseline" if event == "drop" else "2 h ERA5 persistence"
    ax.plot(x, df["baseline_mw"], color=CORAL, lw=1.5, ls="--", label=baseline_label)
    if event == "drop":
        event_row = df.loc[df["datetime"].eq(CLOUD_PEAK)].iloc[0]
        event_y = event_row["actual_mw"]
        event_text = (0.37, 0.85)
    else:
        event_row = df.loc[df["actual_mw"].diff(8).idxmax()]
        event_y = event_row["actual_mw"]
        event_text = (event_row["datetime"], 1650)
    label = "21 Jun midday" if event == "drop" else "Largest 2 h wind increase"
    ax.annotate(
        label,
        (event_row["datetime"], event_y),
        xytext=event_text,
        textcoords="axes fraction" if event == "drop" else "data",
        color=color_event,
        fontsize=9,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=color_event, lw=1.2),
    )
    ax.set_title(title, color=FG, fontsize=10, loc="left")
    ax.set_ylabel("MW", color=FG)
    ax.set_facecolor(PANEL)
    ax.grid(color=GRID, lw=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))


def main() -> Path:
    cfg = get_settings()
    boundary = _load_romania_polygons()
    hubs, grid = load_weather(ROOT / "data", cfg)
    telemetry = load_entsoe(ROOT / "data", cfg)
    actual_label = "ERA5-derived SEN benchmark (MW)" if telemetry is None else "ENTSO-E actual"
    fc = build_forecasts(cfg, hubs, telemetry)
    evaluator = PredicoEvaluator()
    metrics = {
        target: evaluator.compute_metrics(df["actual_mw"], df[QUANTILE_COLUMNS], df["baseline_mw"])
        | {"crps_mw": df.attrs["crps_mw"]}
        for target, df in fc.items()
    }
    solar = fc["solar_mw"]
    # For the deterministic baseline, CRPS is the MAE of the plotted series.
    solar_baseline_crps = metrics["solar_mw"]["mae_nwp_mw"]
    metrics["solar_mw"]["skill_score_vs_nwp_pct"] = (
        100.0 * (1.0 - solar.attrs["crps_mw"] / solar_baseline_crps) if solar_baseline_crps > 0 else float("nan")
    )
    plt.rcParams.update(
        {
            "text.color": FG,
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "axes.edgecolor": GRID,
            "font.family": "DejaVu Sans",
        }
    )
    fig = plt.figure(figsize=(18, 9.5), facecolor=BG)
    fig.text(
        0.5,
        0.965,
        "RoCast · Probabilistic Solar & Wind Forecasting for the Romanian SEN (10YRO-TEL------P)",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="white",
    )
    fig.text(
        0.5,
        0.925,
        "NVIDIA Earth2Studio (FourCastNet 3 + CorrDiff) • EUMETSAT + pySTEPS Optical Flow • "
        "pvlib Clear-Sky Physics • OpenClimateFix PVNet 3D Fusion",
        ha="center",
        va="center",
        fontsize=12,
        color=CYAN,
    )

    ax_map = fig.add_axes([0.04, 0.08, 0.40, 0.80], facecolor=PANEL)
    wind_df = fc["wind_onshore_mw"]
    draw_map(ax_map, cfg, CLOUD_PEAK, boundary, grid)

    ax_s = fig.add_axes([0.53, 0.52, 0.44, 0.36])
    ax_w = fig.add_axes([0.53, 0.09, 0.44, 0.36], sharex=ax_s)
    lead = INTRADAY_LEAD / pd.Timedelta(hours=1)
    draw_fan(ax_s, fc["solar_mw"], f"B · 48 h rolling {lead:g} h hindcast — Solar (MW)", CORAL, "drop", actual_label)
    draw_fan(ax_w, wind_df, "Onshore Wind (MW)", CYAN, "surge", actual_label)
    ax_s.set_ylim(0, max(2600, ax_s.get_ylim()[1]))
    ax_w.set_ylim(0, 1950)
    ax_w.set_xlim(TEST_START, TEST_END)
    ax_s.tick_params(labelbottom=False)
    ax_s.legend(loc="upper right", ncol=5, fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=FG)
    ax_w.legend(loc="upper right", fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=FG)

    rows = ["PREDICO BACKTEST SCORECARD (RO ZONE)", f"{'':<8}{'SOLAR':>9}{'WIND':>9}"]
    for key, name in [
        ("pinball_loss_mw", "Pinball"),
        ("crps_mw", "CRPS"),
        ("mae_p50_mw", "MAE P50"),
        ("p10_p90_coverage_pct", "Cov all%"),
        ("skill_score_vs_nwp_pct", "Skill %"),
    ]:
        value_format = "+9.1f" if key == "skill_score_vs_nwp_pct" else "9.1f"
        rows.append(
            f"{name:<8}{metrics['solar_mw'][key]:{value_format}}{metrics['wind_onshore_mw'][key]:{value_format}}"
        )
    rows.append(f"{'Cov day%':<8}{solar.attrs['daylight_coverage_pct']:>9.1f}{'—':>9}")
    rows.append(
        "[ERA5 archive · 5d train / 48h OOS test]" if telemetry is None else "[ENTSO-E + ERA5 · 48h out-of-sample]"
    )
    rows.append("Solar CRPS vs day-ahead ERA5 · Wind MAE")
    ax_w.text(
        TEST_START + pd.Timedelta(hours=4),
        1850,
        "\n".join(rows),
        family="monospace",
        fontsize=8.5,
        color="white",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0b0f1a", ec=CYAN, lw=1.0),
        zorder=10,
    )

    path = ROOT / "assets" / "rocast_linkedin_hero.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor=BG)
    plt.close(fig)
    for destination in (ROOT / path.name, ROOT / "outputs" / path.name):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        print(f"Wrote {destination}")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    main()
