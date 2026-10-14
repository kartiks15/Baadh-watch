"""Compare a finished run with Copernicus EMS rapid-mapping vectors (e.g. EMSR927). CHECKING ONLY.

EMS products are never pipeline inputs (challenge data rules). This is the only module that reads validation/.
Download the EMSR927 vector packages from https://mapping.emergency.copernicus.eu/activations/EMSR927/
and unzip them anywhere under validation/. Then:

    python -m eval.evaluate_emsr --run <run_id>

Writes data/runs/<id>/emsr927_eval.json and emsr927_agreement.png (green = both, red = only ours,
blue = only EMS). Credit: "European Union, Copernicus Emergency Management Service data".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union

from pipeline.config import RUNS, VALIDATION

DAMAGED = ("destroyed", "damaged", "possibly damaged", "affected", "flooded", "eroded", "washed")


def _latest_products(files: list[Path]) -> list[Path]:
    """One file per layer: prefer GeoJSON over shapefile, and per AOI the latest monitoring product/version."""
    import re

    by_stem = {}
    for p in files:
        stem = p.with_suffix("")
        if stem not in by_stem or p.suffix in (".json", ".geojson"):
            by_stem[stem] = p
    best: dict[str, tuple] = {}
    for p in by_stem.values():
        m = re.search(r"AOI(\d+)_\w+?_(PRODUCT|MONIT(\d+))_.*_v(\d+)", p.name)
        if not m:
            continue
        rank = (int(m.group(3) or 0), int(m.group(4)))
        key = m.group(1)
        prod = p.name[:p.name.index(m.group(2)) + len(m.group(2))]
        if key not in best or rank > best[key][0]:
            best[key] = (rank, prod)
    keep = {prod for _, prod in best.values()}
    return [p for p in by_stem.values() if any(p.name.startswith(k) for k in keep)] or list(by_stem.values())


def load_layers(root: Path) -> dict[str, gpd.GeoDataFrame]:
    out: dict[str, list] = {"event": [], "aoi": [], "built": [], "transport": []}
    files = _latest_products([p for ext in ("*.json", "*.geojson", "*.shp", "*.gpkg") for p in root.rglob(ext)])
    for p in files:
        name = p.name.lower()
        kind = ("event" if "observedevent" in name else "aoi" if "areaofinterest" in name else
                "built" if "builtup" in name else "transport" if "transportationl" in name else None)
        if kind is None:
            continue
        try:
            g = gpd.read_file(p)
        except Exception as ex:  # noqa: BLE001
            print("skip", p.name, ex)
            continue
        if g.crs is None:
            g = g.set_crs(4326)
        g["_src"] = p.name
        out[kind].append(g.to_crs(4326))
    return {k: (pd.concat(v, ignore_index=True) if v else gpd.GeoDataFrame(geometry=[], crs=4326))
            for k, v in out.items()}


def _damaged(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if df.empty:
        return df
    cols = [c for c in df.columns if c != "geometry" and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
    txt = df[cols].astype(str).agg(" ".join, axis=1).str.lower() if cols else pd.Series("", index=df.index)
    mask = txt.apply(lambda s: any(k in s for k in DAMAGED) and "no visible damage" not in s and "not affected" not in s)
    return df[mask]


def evaluate(run_dir: Path, root: Path = VALIDATION) -> dict:
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    L = load_layers(root)
    if L["event"].empty:
        raise SystemExit(f"No EMS observedEvent layers found under {root}")
    with rasterio.open(run_dir / "affected.tif") as src:
        ours = src.read(1) > 0
        transform, crs, shape = src.transform, src.crs, src.shape
    grid_poly = box(transform.c, transform.f + transform.e * shape[0], transform.c + transform.a * shape[1], transform.f)

    ev = L["event"].to_crs(crs)
    aoi = L["aoi"].to_crs(crs) if not L["aoi"].empty else gpd.GeoDataFrame(geometry=[ev.unary_union.envelope], crs=crs)
    aoi_geom = unary_union(aoi.geometry).intersection(grid_poly)
    if aoi_geom.is_empty:
        raise SystemExit("EMS areas of interest do not overlap this run's area")
    aoi_mask = rasterize([(aoi_geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    ref = rasterize(((g, 1) for g in ev.geometry if g is not None and not g.is_empty), out_shape=shape,
                    transform=transform, fill=0, dtype="uint8", all_touched=False).astype(bool)

    def scores(a, b, m):
        tp, fp, fn = int((a & b & m).sum()), int((a & ~b & m).sum()), int((~a & b & m).sum())
        px = abs(transform.a * transform.e) / 1e4
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"iou": round(tp / max(tp + fp + fn, 1), 3), "precision": round(prec, 3), "recall": round(rec, 3),
                "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "ours_ha": round((a & m).sum() * px, 1),
                "ems_ha": round((b & m).sum() * px, 1), "both_ha": round(tp * px, 1)}

    extent = scores(ours, ref, aoi_mask)
    # tolerance: count a hit if within 2 pixels (EMS uses VHR imagery, we use 10–20 m pixels)
    ours_d = ndimage.binary_dilation(ours, iterations=2)
    ref_d = ndimage.binary_dilation(ref, iterations=2)
    tol = {"precision": round(float((ours & ref_d & aoi_mask).sum()) / max(float((ours & aoi_mask).sum()), 1), 3),
           "recall": round(float((ref & ours_d & aoi_mask).sum()) / max(float((ref & aoi_mask).sum()), 1), 3)}

    res = {"extent": extent, "extent_2px_tolerance": tol}

    built = _damaged(L["built"].to_crs(crs)) if not L["built"].empty else L["built"]
    if not built.empty:
        pts = built.geometry.centroid
        pts = pts[pts.within(aoi_geom)]
        inv = ~transform
        hits = 0
        for p in pts:
            c, r = inv * (p.x, p.y)
            r, c = int(r), int(c)
            if 0 <= r < shape[0] and 0 <= c < shape[1] and ours_d[r, c]:
                hits += 1
        res["buildings"] = {"ems_damaged_or_destroyed": int(len(pts)), "inside_our_affected_area": hits,
                            "recall": round(hits / max(len(pts), 1), 3),
                            "our_buildings_hit": summ["buildings"]["in_water"] + summ["buildings"]["in_debris"]}

    tr = _damaged(L["transport"].to_crs(crs)) if not L["transport"].empty else L["transport"]
    if not tr.empty:
        ems_km = float(tr.geometry.intersection(aoi_geom).length.sum()) / 1000
        rb = gpd.read_file(run_dir / "roads_blocked.geojson").to_crs(crs)
        our_km = float(rb.geometry.intersection(aoi_geom).length.sum()) / 1000 if not rb.empty else 0.0
        near = float(rb.geometry.intersection(aoi_geom).intersection(unary_union(tr.geometry).buffer(30)).length.sum()) / 1000 \
            if not rb.empty else 0.0
        res["roads"] = {"ems_affected_km": round(ems_km, 2), "our_blocked_km": round(our_km, 2),
                        "our_blocked_km_matching_ems": round(near, 2)}

    rgba = np.zeros((*shape, 4), np.uint8)
    rgba[ours & ref & aoi_mask] = (22, 163, 74, 220)
    rgba[ours & ~ref & aoi_mask] = (220, 38, 38, 200)
    rgba[~ours & ref & aoi_mask] = (37, 99, 235, 200)
    Image.fromarray(rgba, "RGBA").save(run_dir / "emsr927_agreement.png")
    out = {"summary": res, "reference_files": sorted(set(L["event"]["_src"])),
           "note": "Reference: European Union, Copernicus Emergency Management Service data (EMSR927). Used for "
                   "checking only; never as a pipeline input. Thresholds were not tuned on this reference."}
    (run_dir / "emsr927_eval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--validation", default=str(VALIDATION))
    a = ap.parse_args()
    print(json.dumps(evaluate(RUNS / a.run, Path(a.validation))["summary"], indent=1))
