"""Raw satellite data access.

Sentinel-1 GRD and Sentinel-2 L2A come from the Sentinel Hub Process API on the Copernicus Data Space
Ecosystem (free account). The Copernicus DEM GLO-30 comes from the public AWS Open Data COGs.
Everything is cached under data/cache so re-runs cost no processing units.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timedelta

import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, reproject, transform_bounds

from .config import CACHE, SH_CLIENT_ID, SH_CLIENT_SECRET, Grid
from .scenes import Pass

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
DEM_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
MAX_PX = 2400

_token: dict = {}


class CredentialsMissing(RuntimeError):
    pass


def _get_token() -> str:
    if _token.get("exp", 0) > time.time() + 60:
        return _token["value"]
    if not (SH_CLIENT_ID and SH_CLIENT_SECRET):
        raise CredentialsMissing("Set SH_CLIENT_ID and SH_CLIENT_SECRET in .env (CDSE Sentinel Hub OAuth client).")
    r = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": SH_CLIENT_ID,
                                       "client_secret": SH_CLIENT_SECRET}, timeout=30)
    r.raise_for_status()
    js = r.json()
    _token.update(value=js["access_token"], exp=time.time() + js.get("expires_in", 600))
    return _token["value"]


def _process(bounds, epsg: int, width: int, height: int, data: list, evalscript: str) -> np.ndarray:
    body = {
        "input": {"bounds": {"bbox": list(bounds),
                             "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                  "data": data},
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    for attempt in range(5):
        r = requests.post(PROCESS_URL, json=body, timeout=300,
                          headers={"Authorization": f"Bearer {_get_token()}", "Accept": "image/tiff"})
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Sentinel Hub {r.status_code}: {r.text[:500]}")
        with rasterio.MemoryFile(r.content) as mem, mem.open() as src:
            return src.read()
    raise RuntimeError("Sentinel Hub: too many retries")


def _fetch_grid(grid: Grid, data: list, evalscript: str, nbands: int, cache_key: dict) -> np.ndarray:
    key = hashlib.sha1(json.dumps([cache_key, grid.epsg, grid.res, grid.x0, grid.y1, grid.width, grid.height],
                                  sort_keys=True, default=str).encode()).hexdigest()
    path = CACHE / "sh" / f"{key}.npy"
    if path.exists():
        return np.load(path)
    out = np.zeros((nbands, grid.height, grid.width), dtype=np.uint16)
    for r0 in range(0, grid.height, MAX_PX):
        for c0 in range(0, grid.width, MAX_PX):
            h, w = min(MAX_PX, grid.height - r0), min(MAX_PX, grid.width - c0)
            x0 = grid.x0 + c0 * grid.res
            y1 = grid.y1 - r0 * grid.res
            bounds = (x0, y1 - h * grid.res, x0 + w * grid.res, y1)
            out[:, r0:r0 + h, c0:c0 + w] = _process(bounds, grid.epsg, w, h, data, evalscript)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out


S1_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "localIncidenceAngle", "dataMask"]}],
          output: {bands: 4, sampleType: "UINT16"}};
}
function db(v) {
  if (!(v > 0)) return 1;
  var d = 10 * Math.log(v) / Math.LN10;
  return Math.round((Math.min(Math.max(d, -50), 10) + 50) * 1000) + 1;
}
function evaluatePixel(s) {
  if (s.dataMask == 0) return [0, 0, 0, 0];
  return [db(s.VV), db(s.VH), Math.round(s.localIncidenceAngle * 100), 1];
}"""


def fetch_s1(grid: Grid, p: Pass) -> dict:
    """Sigma0 (dB), orthorectified with Copernicus DEM 30 m, Lee-filtered, for one pass."""
    t0 = (p.start - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = (p.end + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = [{
        "type": "sentinel-1-grd",
        "dataFilter": {"timeRange": {"from": t0, "to": t1}, "acquisitionMode": "IW", "polarization": "DV",
                       "orbitDirection": p.direction.upper(), "resolution": "HIGH"},
        "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True, "demInstance": "COPERNICUS_30",
                       "speckleFilter": {"type": "LEE", "windowSizeX": 5, "windowSizeY": 5},
                       "upsampling": "BILINEAR"},
    }]
    raw = _fetch_grid(grid, data, S1_EVALSCRIPT, 4, {"s1": p.names, "v": 2})
    valid = raw[3] > 0
    vv = np.where(valid, (raw[0].astype(np.float32) - 1) / 1000 - 50, np.nan)
    vh = np.where(valid, (raw[1].astype(np.float32) - 1) / 1000 - 50, np.nan)
    lia = np.where(valid, raw[2].astype(np.float32) / 100, np.nan)
    return {"vv": vv, "vh": vh, "lia": lia, "valid": valid, "pass": p}


S2_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["B02", "B03", "B04", "B08", "B11", "SCL", "dataMask"]}],
          output: {bands: 7, sampleType: "UINT16"}};
}
function evaluatePixel(s) {
  return [s.B02 * 10000, s.B03 * 10000, s.B04 * 10000, s.B08 * 10000, s.B11 * 10000, s.SCL, s.dataMask];
}"""
SCL_EVALSCRIPT = """//VERSION=3
function setup() { return {input: [{bands: ["SCL", "dataMask"]}], output: {bands: 2, sampleType: "UINT8"}}; }
function evaluatePixel(s) { return [s.SCL, s.dataMask]; }"""
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

# SCL classes treated as unusable: no data, saturated, cloud shadow, cloud medium/high, cirrus
S2_BAD_SCL = (0, 1, 3, 8, 9, 10)


def _s2_data(day_from: str, day_to: str, order: str = "mostRecent") -> list:
    return [{"type": "sentinel-2-l2a",
             "dataFilter": {"timeRange": {"from": f"{day_from}T00:00:00Z", "to": f"{day_to}T23:59:59Z"},
                            "mosaickingOrder": order, "maxCloudCoverage": 100},
             "processing": {"upsampling": "BILINEAR"}}]


def s2_days(bbox, start: datetime, end: datetime) -> list[str]:
    """Distinct Sentinel-2 L2A acquisition days over the bbox (CDSE catalogue)."""
    r = requests.post(CATALOG_URL, headers={"Authorization": f"Bearer {_get_token()}"}, timeout=60, json={
        "collections": ["sentinel-2-l2a"], "bbox": list(bbox), "limit": 100,
        "datetime": f"{start:%Y-%m-%d}T00:00:00Z/{end:%Y-%m-%d}T23:59:59Z",
        "fields": {"include": ["properties.datetime"], "exclude": ["assets", "links", "geometry"]}})
    r.raise_for_status()
    return sorted({f["properties"]["datetime"][:10] for f in r.json().get("features", [])})


def s2_clear_fraction(bbox, day: str) -> float:
    """Share of the bbox that is cloud-free on `day`, from SCL on a coarse (≈100 m) grid: costs almost nothing."""
    from .config import make_grid

    coarse = make_grid(bbox, 100.0)
    raw = _fetch_grid(coarse, _s2_data(day, day), SCL_EVALSCRIPT, 2, {"scl": day, "v": 1})
    valid = raw[1] > 0
    return float((valid & ~np.isin(raw[0], S2_BAD_SCL)).sum()) / max(int(valid.sum()), 1) * float(valid.mean())


def fetch_s2_day(grid: Grid, day: str) -> dict:
    raw = _fetch_grid(grid, _s2_data(day, day), S2_EVALSCRIPT, 7, {"s2day": day, "v": 2})
    clear = (raw[6] > 0) & ~np.isin(raw[5], S2_BAD_SCL)
    f = raw[:5].astype(np.float32) / 10000
    return {"b2": f[0], "b3": f[1], "b4": f[2], "b8": f[3], "b11": f[4], "clear": clear, "day": day}


def s2_composite(grid: Grid, bbox, start: datetime, end: datetime, prefer: str, n: int = 3,
                 min_clear: float = 0.15) -> dict | None:
    """Per-pixel composite of the `n` clearest days in [start, end]: for every pixel take the clear observation
    closest to the event (`prefer`='earliest' after it, 'latest' before it)."""
    days = s2_days(bbox, start, end)
    scored = [(d, s2_clear_fraction(bbox, d)) for d in days]
    good = [d for d, c in sorted(scored, key=lambda x: -x[1])[:n] if c >= min_clear]
    if not good:
        return None
    good.sort(reverse=(prefer == "latest"))  # closest to the event first
    out = None
    for d in good:
        img = fetch_s2_day(grid, d)
        if out is None:
            out = {k: np.where(img["clear"], img[k], np.nan) for k in ("b2", "b3", "b4", "b8", "b11")}
            out["clear"] = img["clear"].copy()
            continue
        fill = img["clear"] & ~out["clear"]
        for k in ("b2", "b3", "b4", "b8", "b11"):
            out[k][fill] = img[k][fill]
        out["clear"] |= img["clear"]
    out.update(days=good, clear_frac=float(out["clear"].mean()),
               screened={d: round(c, 3) for d, c in scored})
    return out


def _dem_tile_names(bbox) -> list[str]:
    w, s, e, n = bbox
    names = []
    for lat in range(int(np.floor(s)), int(np.floor(n)) + 1):
        for lon in range(int(np.floor(w)), int(np.floor(e)) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
            names.append(f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM")
    return names


def _dem_tile(name: str):
    path = CACHE / "dem" / f"{name}.tif"
    if not path.exists():
        r = requests.get(DEM_URL.format(n=name), timeout=300)
        if r.status_code in (403, 404):
            return None  # ocean tile
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    return path


def fetch_dem(bounds_utm, epsg: int, res: float = 30.0):
    """Copernicus DEM GLO-30 resampled onto a UTM grid covering `bounds_utm`. Returns (dem, transform)."""
    from affine import Affine

    x0, y0, x1, y1 = bounds_utm
    width, height = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    transform = Affine(res, 0, x0, 0, -res, y1)
    dem = np.full((height, width), np.nan, dtype=np.float32)
    bbox_ll = transform_bounds(f"EPSG:{epsg}", "EPSG:4326", x0, y0, x1, y1)
    for name in _dem_tile_names(bbox_ll):
        path = _dem_tile(name)
        if path is None:
            continue
        with rasterio.open(path) as src:
            tmp = np.full_like(dem, np.nan)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=f"EPSG:{epsg}",
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
            dem = np.where(np.isnan(dem), tmp, dem)
    if np.isnan(dem).all():
        raise RuntimeError("Copernicus DEM: no tiles found for this area")
    return dem, transform
