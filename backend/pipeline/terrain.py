"""Terrain products from the Copernicus DEM: slope, D8 flow, HAND, and downstream flow-path tracing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject

from .config import Grid

if not hasattr(np, "in1d"):  # pysheds 0.5 still calls np.in1d, removed in NumPy 2.4
    np.in1d = np.isin

DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
STEP = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def compute_terrain(dem: np.ndarray, transform, epsg: int, rivers=None, channel_km2: float = 5.0) -> dict:
    """HAND relative to channels = (flow accumulation > channel_km2) ∪ pre-event OSM rivers."""
    from pysheds.grid import Grid as PGrid

    res = transform.a
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "dem.tif"
        with rasterio.open(path, "w", driver="GTiff", width=dem.shape[1], height=dem.shape[0], count=1,
                           dtype="float32", crs=f"EPSG:{epsg}", transform=transform, nodata=-9999) as dst:
            dst.write(np.nan_to_num(dem, nan=-9999).astype(np.float32), 1)
        pg = PGrid.from_raster(str(path))
        rdem = pg.read_raster(str(path))
        filled = pg.fill_pits(rdem)
        flooded = pg.fill_depressions(filled)
        inflated = pg.resolve_flats(flooded)
        fdir = pg.flowdir(inflated, dirmap=DIRMAP)
        acc = pg.accumulation(fdir, dirmap=DIRMAP)
        channels = np.asarray(acc) * res * res / 1e6 > channel_km2
        if rivers is not None and len(rivers):
            burned = rasterize(((g, 1) for g in rivers), out_shape=dem.shape, transform=transform,
                               fill=0, all_touched=True, dtype="uint8").astype(bool)
            channels |= burned
        from pysheds.sview import Raster, ViewFinder
        vf = ViewFinder(affine=rdem.affine, shape=rdem.shape, crs=rdem.crs, nodata=False)
        hand = pg.compute_hand(fdir, rdem, Raster(channels, viewfinder=vf), dirmap=DIRMAP)
        hand = np.asarray(hand, dtype=np.float32)
        hand[~np.isfinite(hand)] = np.nan
        fdir = np.asarray(fdir).astype(np.int32)
        acc = np.asarray(acc).astype(np.float32)

    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    return {"dem": dem, "fdir": fdir, "acc": acc, "hand": hand, "slope": slope, "channels": channels,
            "transform": transform, "epsg": epsg}


def to_grid(arr: np.ndarray, transform, epsg: int, grid: Grid, resampling=Resampling.bilinear) -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan, dtype=np.float32)
    reproject(arr.astype(np.float32), out, src_transform=transform, src_crs=f"EPSG:{epsg}", src_nodata=np.nan,
              dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan, resampling=resampling)
    return out


def trace_downstream(terrain: dict, x: float, y: float, snap_m: float = 300.0, max_km: float = 150.0) -> dict:
    """Follow D8 flow directions from (x, y) in UTM until the DEM edge. Snaps to the strongest channel nearby."""
    t = terrain["transform"]
    fdir, acc, dem = terrain["fdir"], terrain["acc"], terrain["dem"]
    res = t.a
    col, row = ~t * (x, y)
    row, col = int(row), int(col)
    h, w = fdir.shape
    if not (0 <= row < h and 0 <= col < w):
        raise ValueError("point is outside the analysed terrain")
    r = int(snap_m / res)
    r0, r1, c0, c1 = max(row - r, 0), min(row + r + 1, h), max(col - r, 0), min(col + r + 1, w)
    win = acc[r0:r1, c0:c1]
    dr, dc = np.unravel_index(np.nanargmax(win), win.shape)
    row, col = r0 + dr, c0 + dc

    coords, dist, elev = [], [], []
    d = 0.0
    seen = set()
    while 0 <= row < h and 0 <= col < w and (row, col) not in seen and d < max_km * 1000:
        seen.add((row, col))
        cx, cy = t * (col + 0.5, row + 0.5)
        coords.append((cx, cy))
        dist.append(d)
        elev.append(float(dem[row, col]))
        step = STEP.get(int(fdir[row, col]))
        if step is None:
            break
        row, col = row + step[0], col + step[1]
        d += res * (1.4142 if step[0] and step[1] else 1.0)
    return {"coords": coords, "dist_m": dist, "elev_m": elev}
