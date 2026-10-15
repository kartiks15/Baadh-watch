"""Paths, credentials, thresholds and the analysis grid."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from affine import Affine
from dotenv import load_dotenv
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = DATA / "runs"
CACHE = DATA / "cache"
MODELS = ROOT / "backend" / "models"
# validation/ holds Copernicus EMS reference maps. Pipeline code must never read it (data rules).
VALIDATION = ROOT / "validation"

load_dotenv(ROOT / ".env")

SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "")
# ohsome API v2 (HeiGIT): pre-event OSM for any area and date. Free key: https://account.heigit.org/signup
OHSOME_API_KEY = os.getenv("OHSOME_API_KEY", "")
OHSOME_URL = os.getenv("OHSOME_URL", "https://api.heigit.org/ohsome-api/v2").rstrip("/")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-5-5")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-5.5")
OPENROUTER_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "8000"))  # reasoning + report (Nepali needs more tokens)
# "anthropic" (direct API) or "openrouter"; picked automatically from whichever key is set
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "") or ("anthropic" if ANTHROPIC_API_KEY else
                                                  "openrouter" if OPENROUTER_API_KEY else "")

PIPELINE_VERSION = "0.3"


@dataclass
class Thresholds:
    # flood-reachable zone. Two regimes, chosen from terrain relief:
    #  mountain (relief > 1000 m): debris floods surge tens of metres above the normal channel
    #  lowland: classic riverine flooding stays within ~15 m of drainage on gentle ground
    relief_mountain_m: float = 1000.0
    hand_max_m: float = 60.0
    slope_max_deg: float = 35.0
    hand_max_lowland_m: float = 15.0
    slope_max_lowland_deg: float = 10.0
    # local incidence angle outside this range = layover/foreshortening or radar shadow
    lia_min_deg: float = 15.0
    lia_max_deg: float = 75.0
    # SAR change detection on Lee-filtered sigma0 (dB), same relative orbit
    change_db: float = 3.0          # fallback when only one pre-event image exists
    cfar_quantile: float = 99.0     # threshold = this percentile of pre-vs-pre change (≈1% false alarms / band)
    change_floor_db: float = 1.5
    water_prob: float = 0.5
    # fallback water thresholds when no model is available (sigma0 dB)
    water_vv_db: float = -18.0
    water_vh_db: float = -25.0
    # Sentinel-2 change
    ndvi_drop: float = 0.25
    mndwi_rise: float = 0.30
    s2_max_cloud_frac: float = 0.6
    # cleanup
    min_area_ha: float = 0.3
    # impact
    building_risk_buffer_m: float = 30.0
    road_buffer_m: float = 10.0
    bridge_buffer_m: float = 50.0
    # network
    detour_ratio: float = 1.5
    snap_max_m: float = 1000.0
    settlement_radius_m: float = 500.0
    # Nepal Census 2021 average household size (4.37); one household per building is a proxy.
    people_per_building: float = 4.4


@dataclass
class Grid:
    """Analysis grid in the AOI's UTM zone (north-up)."""
    epsg: int
    res: float
    x0: float
    y1: float
    width: int
    height: int
    bbox: tuple[float, float, float, float] = field(default=(0, 0, 0, 0))  # lon/lat W,S,E,N

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y1 - self.height * self.res, self.x0 + self.width * self.res, self.y1)

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def pixel_area_ha(self) -> float:
        return self.res * self.res / 10_000

    def corners_lonlat(self) -> list[list[float]]:
        """TL, TR, BR, BL corners in lon/lat (MapLibre image-source order)."""
        t = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        x0, y0, x1, y1 = self.bounds
        return [list(t.transform(x, y)) for x, y in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_grid(bbox: tuple[float, float, float, float], res: float | None = None) -> Grid:
    w, s, e, n = bbox
    epsg = utm_epsg((w + e) / 2, (s + n) / 2)
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    xs, ys = [], []
    for lon in (w, (w + e) / 2, e):
        for lat in (s, (s + n) / 2, n):
            x, y = t.transform(lon, lat)
            xs.append(x)
            ys.append(y)
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    if res is None:
        res = 10.0 if side <= 20_000 else 20.0
    x0 = math.floor(min(xs) / res) * res
    y1 = math.ceil(max(ys) / res) * res
    width = int(math.ceil((max(xs) - x0) / res))
    height = int(math.ceil((y1 - min(ys)) / res))
    return Grid(epsg=epsg, res=res, x0=x0, y1=y1, width=width, height=height, bbox=tuple(bbox))


def run_id(bbox, event_date: str, res: float | None) -> str:
    key = json.dumps([[round(v, 4) for v in bbox], event_date, res, PIPELINE_VERSION])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def thresholds_dict(th: Thresholds) -> dict:
    return asdict(th)
