"""Generate the dark-mode dual-panel RoCast LinkedIn hero image (assets/rocast_linkedin_hero.png).

The Romanian outline is rendered from a cached, production-grade vector boundary stored in
``assets/geo/romania_boundary.geojson``.  The cache is populated on first run from geoBoundaries
ADM0 (CC BY 3.0) with Natural Earth 1:10m/1:50m fallbacks; every later run is fully offline.
"""

from __future__ import annotations

import json
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
from matplotlib.patches import PathPatch

from rocast.config import get_settings
from rocast.earth2.nvidia_weather_engine import NvidiaEarth2Engine
from rocast.ingestion.grid_weather_sampler import RHO_GROUND_MIN, synthetic_satellite_frames
from rocast.physics.clearsky_advection import ClearSkyPhysicsCalculator, PyStepsCloudNowcaster

BG, PANEL, FG, GRID = "#0b0f1a", "#111827", "#e5e7eb", "#2a3347"
CYAN, CORAL = "#22d3ee", "#fb7185"
BOUNDARY_COLOR = "#00F5D4"

BOUNDARY_CACHE = ROOT / "assets" / "geo" / "romania_boundary.geojson"
BOUNDARY_USER_AGENT = "RoCast-hero-generator/1.0 (LinkedIn hero asset builder)"
BOUNDARY_TIMEOUT = 120.0

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
        f"{BOUNDARY_CACHE}. Place a Romania GeoJSON there manually, or restore network access.\n"
        + "\n".join(failures)
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
LABELS = {
    "dobrogea_wind": "Dobrogea Wind\n2.65 GW",
    "moldova_wind": "Focșani Gate / Moldova\n0.75 GW",
    "wallachia_solar": "Wallachia PV\n1.40 GW",
    "oltenia_solar": "Oltenia PV\n0.85 GW",
    "transylvania_solar": "Transylvania PV\n0.60 GW",
}


def load_artifacts(out: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    needed = [
        out / "backtest_metrics.json",
        out / "backtest_forecast_solar_mw.csv",
        out / "backtest_forecast_wind_onshore_mw.csv",
    ]
    if not all(p.exists() for p in needed):
        from rocast.cli import run_backtest

        run_backtest("synthetic", epochs=3, output_dir=out, verbose=False)
    metrics = json.loads(needed[0].read_text())
    fc = {
        k: pd.read_csv(out / f"backtest_forecast_{k}.csv", parse_dates=["datetime"])
        for k in ("solar_mw", "wind_onshore_mw")
    }
    return metrics, fc


def draw_map(ax, cfg, when: pd.Timestamp, boundary: list[list[np.ndarray]]) -> None:
    lon_min, lat_min, lon_max, lat_max = cfg.bbox
    engine = NvidiaEarth2Engine(cfg)
    tensor, _ = engine.generate_synthetic_earth2_ensemble(pd.DatetimeIndex([when]))
    ws100 = np.hypot(tensor[0, 2], tensor[0, 3])
    extent = (lon_min, lon_max, lat_min, lat_max)
    im = ax.imshow(ws100, extent=extent, origin="upper", cmap="viridis", alpha=0.9, aspect="auto", vmin=2, vmax=16)

    sat_t = pd.date_range(when - pd.Timedelta(minutes=60), periods=4, freq="15min")
    frames = synthetic_satellite_frames(sat_t, cfg)
    n_obs = ClearSkyPhysicsCalculator.heliosat_cloud_index(frames[:, 0], RHO_GROUND_MIN)
    nowcaster = PyStepsCloudNowcaster(cfg)
    _, vel = nowcaster.nowcast(n_obs, cfg.t_nowcast)
    h, w = n_obs[-1].shape
    lons = lon_min + (np.arange(w) + 0.5) * (lon_max - lon_min) / w
    lats = lat_max - (np.arange(h) + 0.5) * (lat_max - lat_min) / h
    ax.contour(
        lons,
        lats,
        n_obs[-1],
        levels=[0.35, 0.6, 0.85],
        colors=["#fde68a", "#fbbf24", "#f8fafc"],
        linewidths=1.0,
        alpha=0.9,
    )
    s = 2
    ax.quiver(
        lons[::3], lats[::s], vel[0][::s, ::3], -vel[1][::s, ::3], color="white", alpha=0.75, scale=22, width=0.003
    )

    for patch in _build_boundary_patches(boundary):
        ax.add_patch(patch)
    cx, cy = zip(*CARPATHIANS, strict=False)
    ax.plot(cx, cy, color="#f59e0b", lw=2.2, ls=(0, (6, 3)), alpha=0.9)
    ax.text(25.6, 46.95, "Carpathian Ridge", color="#fbbf24", fontsize=9, rotation=-60, ha="center")
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
        dx = -0.2 if hub.lon > 27 else 0.25
        ax.annotate(
            LABELS[hub.name],
            (hub.lon, hub.lat),
            xytext=(hub.lon + dx, hub.lat + 0.3),
            color="white",
            fontsize=8,
            ha="right" if dx < 0 else "left",
            bbox=dict(boxstyle="round,pad=0.25", fc="#0b0f1acc", ec=CYAN if wind else CORAL, lw=0.8),
        )
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Earth-2 100 m wind speed (m/s)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (°E)", color=FG)
    ax.set_ylabel("Latitude (°N)", color=FG)
    ax.set_title(
        f"A · Romania SEN: Earth-2 wind • Heliosat clouds • pySTEPS flow  ({when:%Y-%m-%d %H:%M} UTC)",
        color=FG,
        fontsize=11,
        loc="left",
    )
    ax.text(
        0.0,
        -0.135,
        _boundary_attribution(),
        transform=ax.transAxes,
        color=FG,
        fontsize=6,
        alpha=0.75,
        ha="left",
        va="top",
    )


def draw_fan(ax, df: pd.DataFrame, title: str, color_event: str, event: str) -> None:
    x = df["datetime"]
    ax.fill_between(x, df["q10"], df["q90"], color=CYAN, alpha=0.18, label="P10–P90")
    ax.fill_between(x, df["q25"], df["q75"], color=CYAN, alpha=0.35, label="P25–P75")
    ax.plot(x, df["q50"], color=CYAN, lw=2.0, label="RoCast P50")
    ax.plot(x, df["actual_mw"], color="white", lw=1.6, label="ENTSO-E actual")
    ax.plot(x, df["nwp_mw"], color=CORAL, lw=1.5, ls="--", label="25 km NWP baseline")
    if event == "drop":
        i = int((df["nwp_mw"] - df["actual_mw"]).where(df["actual_mw"] > 200).idxmax())
    else:
        i = int(df["actual_mw"].diff(8).idxmax())
    label = "Wallachia cloud drop vs NWP" if event == "drop" else "Dobrogea / Focșani Gate wind surge"
    ax.annotate(
        label,
        (x[i], df["actual_mw"][i]),
        xytext=(0.62 if event == "drop" else 0.5, 0.68 if event == "drop" else 0.86),
        textcoords="axes fraction",
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
    out = ROOT / "outputs"
    boundary = _load_romania_polygons()
    metrics, fc = load_artifacts(out)
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
    when = wind_df.loc[wind_df["actual_mw"].diff(8).idxmax(), "datetime"]
    when = pd.Timestamp(when).tz_localize(None) if pd.Timestamp(when).tzinfo else pd.Timestamp(when)
    draw_map(ax_map, cfg, when, boundary)

    ax_s = fig.add_axes([0.53, 0.52, 0.44, 0.36])
    ax_w = fig.add_axes([0.53, 0.09, 0.44, 0.36], sharex=ax_s)
    draw_fan(ax_s, fc["solar_mw"], "B · 48 h day-ahead, 15-min MTU — Solar (MW)", CORAL, "drop")
    draw_fan(ax_w, wind_df, "Onshore Wind (MW)", CYAN, "surge")
    ax_s.tick_params(labelbottom=False)
    ax_s.legend(loc="upper right", ncol=5, fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=FG)

    t = metrics["targets"]
    rows = ["PREDICO BACKTEST SCORECARD (RO ZONE)", f"{'':<8}{'SOLAR':>9}{'WIND':>9}"]
    for key, name in [
        ("pinball_loss_mw", "Pinball"),
        ("crps_mw", "CRPS"),
        ("mae_p50_mw", "MAE P50"),
        ("p10_p90_coverage_pct", "Cov %"),
        ("skill_score_vs_nwp_pct", "Skill %"),
    ]:
        rows.append(f"{name:<8}{t['solar_mw'][key]:>9.1f}{t['wind_onshore_mw'][key]:>9.1f}")
    rows.append(f"[{metrics['mode']} data, {metrics['n_test_windows']} test days]")
    ax_w.text(
        0.60,
        0.42,
        "\n".join(rows),
        transform=ax_w.transAxes,
        family="monospace",
        fontsize=8.5,
        color="white",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#0b0f1a", ec=CYAN, lw=1.0),
        zorder=10,
    )

    path = ROOT / "assets" / "rocast_linkedin_hero.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, facecolor=BG)
    plt.close(fig)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


if __name__ == "__main__":
    main()
