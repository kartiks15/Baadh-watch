"""Flood and debris detection by fusing independent evidence on the valley floor.

Evidence sources (each is a yes/no vote per pixel):
  1. model      – U-Net says water after the event but not before (same orbit track).
  2. sar_change – sigma0 changed by more than `change_db` between the pre-event median and the post image
                  (darkening = water / smooth wet mud, brightening = rough boulder & debris deposits).
  3. s2_change  – Sentinel-2 NDVI loss (vegetation scoured or buried) or MNDWI rise, where both images are clear.
All evidence is restricted to the valley floor (HAND, slope) and to pixels free of radar layover/shadow.
"""
from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from .config import Grid, Thresholds


def _nanmedian_stack(arrs: list[np.ndarray]) -> np.ndarray:
    if len(arrs) == 1:
        return arrs[0]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.stack(arrs), axis=0)


def _smooth(a: np.ndarray, k: int = 7) -> np.ndarray:
    """NaN-aware k×k mean (debris deposits are extended areas; averaging beats speckle)."""
    m = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(m, a, 0.0), k)
    den = ndimage.uniform_filter(m.astype(np.float64), k)
    return np.where(den > 0.3, num / np.maximum(den, 1e-6), np.nan)


def _cfar_thresholds(pres: list[dict], zone: np.ndarray, th: Thresholds) -> tuple[float, float, str]:
    """Scene-adaptive change thresholds: the `cfar_quantile` of |Δσ0| between two PRE-event images on the
    valley floor, i.e. how much the ground 'changes' when nothing happened. No reference maps involved."""
    if len(pres) < 2 or zone.sum() < 500:
        return th.change_db, th.change_db, "fixed"
    out = []
    for band in ("vv", "vh"):
        d = np.abs(_smooth(pres[0][band]) - _smooth(pres[1][band]))[zone]
        d = d[np.isfinite(d)]
        out.append(max(float(np.percentile(d, th.cfar_quantile)), th.change_floor_db) if len(d) else th.change_db)
    return out[0], out[1], "cfar"


def track_evidence(post: dict, pres: list[dict], valley: np.ndarray, model, th: Thresholds) -> dict:
    pre_vv = _nanmedian_stack([p["vv"] for p in pres])
    pre_vh = _nanmedian_stack([p["vh"] for p in pres])
    valid = post["valid"] & np.isfinite(pre_vv)
    lia = post["lia"]
    reliable = valid & (lia >= th.lia_min_deg) & (lia <= th.lia_max_deg)
    zone = reliable & valley

    p_post = model.predict(post["vv"], post["vh"], th)
    p_pre = model.predict(pre_vv, pre_vh, th)
    new_water = (p_post > th.water_prob) & (p_pre <= th.water_prob) & zone

    t_vv, t_vh, mode = _cfar_thresholds(pres, zone, th)
    dvv = _smooth(post["vv"]) - _smooth(pre_vv)
    dvh = _smooth(post["vh"]) - _smooth(pre_vh)
    sar_change = ((np.abs(np.nan_to_num(dvv)) > t_vv) | (np.abs(np.nan_to_num(dvh)) > t_vh)) & zone
    darkened_water = (np.nan_to_num(dvv) < -t_vv) & (post["vv"] < th.water_vv_db + 3) & zone
    return {"new_water": new_water, "sar_change": sar_change, "water_like": new_water | darkened_water,
            "zone": zone, "reliable": reliable, "p_post": p_post, "p_pre": p_pre, "dvv": dvv,
            "pre_vv": pre_vv, "post_vv": post["vv"],
            "thresholds_db": {"vv": round(t_vv, 2), "vh": round(t_vh, 2), "mode": mode}}


def s2_evidence(pre: dict | None, post: dict | None, valley: np.ndarray, th: Thresholds) -> dict | None:
    if pre is None or post is None:
        return None
    clear = pre["clear"] & post["clear"]
    if clear.mean() < 0.02:
        return None
    with np.errstate(all="ignore"):
        ndvi = lambda d: (d["b8"] - d["b4"]) / (d["b8"] + d["b4"] + 1e-6)
        mndwi = lambda d: (d["b3"] - d["b11"]) / (d["b3"] + d["b11"] + 1e-6)
        ndvi_drop = np.nan_to_num(ndvi(pre) - ndvi(post))
        d_mndwi = np.nan_to_num(mndwi(post) - mndwi(pre))
        water_post, water_pre = np.nan_to_num(mndwi(post)) > 0.1, np.nan_to_num(mndwi(pre)) > 0.1
    # vegetation scoured/buried, or the wet channel moved (new water, or water replaced by sediment)
    change = ((ndvi_drop > th.ndvi_drop) | (np.abs(d_mndwi) > th.mndwi_rise)) & clear & valley
    water = water_post & ~water_pre & clear & valley
    return {"s2_change": change, "s2_water": water, "clear": clear}


def fuse(tracks: list[dict], s2: dict | None, valley: np.ndarray, grid: Grid, th: Thresholds) -> dict:
    shape_ = valley.shape
    model_vote = np.zeros(shape_, bool)
    sar_vote = np.zeros(shape_, bool)
    water_like = np.zeros(shape_, bool)
    observed = np.zeros(shape_, bool)
    for t in tracks:
        model_vote |= t["new_water"]
        sar_vote |= t["sar_change"]
        water_like |= t["water_like"]
        observed |= t["zone"]
    s2_vote = s2["s2_change"] if s2 else np.zeros(shape_, bool)
    if s2:
        water_like |= s2["s2_water"]
        observed |= s2["clear"] & valley
    votes = model_vote.astype(np.uint8) + sar_vote + s2_vote

    affected = votes >= 1
    affected = ndimage.binary_opening(affected, structure=np.ones((2, 2)))
    affected = ndimage.binary_closing(affected, structure=np.ones((3, 3))) & valley
    min_px = max(int(th.min_area_ha / grid.pixel_area_ha), 4)
    lab, n = ndimage.label(affected, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(affected, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= min_px
        affected = keep[lab]

    cls = np.zeros(shape_, np.uint8)
    cls[affected] = 2  # debris / changed surface
    cls[affected & water_like] = 1  # water
    votes = np.where(affected, np.maximum(votes, 1), 0).astype(np.uint8)
    n_sources = 2 + (1 if s2 else 0)
    valley_px = int(valley.sum())
    stats = {
        "affected_ha": round(float(affected.sum()) * grid.pixel_area_ha, 1),
        "water_ha": round(float((cls == 1).sum()) * grid.pixel_area_ha, 1),
        "debris_ha": round(float((cls == 2).sum()) * grid.pixel_area_ha, 1),
        "high_conf_ha": round(float((votes >= 2).sum()) * grid.pixel_area_ha, 1),
        "valley_floor_ha": round(valley_px * grid.pixel_area_ha, 1),
        "valley_observed_pct": round(100 * float((observed & valley).sum()) / max(valley_px, 1), 1),
        "evidence_sources": n_sources,
        "sar_thresholds_db": [t["thresholds_db"] for t in tracks],
        "votes_px": {"model": int((model_vote & affected).sum()), "sar_change": int((sar_vote & affected).sum()),
                     "s2_change": int((s2_vote & affected).sum())},
    }
    return {"class": cls, "votes": votes, "observed": observed, "stats": stats}


def vectorize(cls: np.ndarray, votes: np.ndarray, grid: Grid) -> list[tuple]:
    """Connected regions → (UTM polygon, properties). Confidence from the share of multi-source pixels."""
    out = []
    for value, name in ((1, "water"), (2, "debris")):
        mask = cls == value
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if not n:
            continue
        idx = np.arange(1, n + 1)
        area_px = ndimage.sum(mask, lab, idx)
        multi = ndimage.sum(votes >= 2, lab, idx)
        mean_votes = ndimage.mean(votes, lab, idx)
        parts: dict[int, list] = {}
        for geom, v in shapes(lab.astype(np.int32), mask=mask, transform=grid.transform, connectivity=8):
            parts.setdefault(int(v), []).append(shape(geom))
        for k, geoms in parts.items():
            g = unary_union(geoms).simplify(grid.res * 0.75)
            share = multi[k - 1] / max(area_px[k - 1], 1)
            ha = area_px[k - 1] * grid.pixel_area_ha
            conf = "high" if share >= 0.25 else ("medium" if ha >= 1.0 else "low")
            out.append((g, {"class": name, "confidence": conf, "area_ha": round(float(ha), 2),
                            "mean_votes": round(float(mean_votes[k - 1]), 2)}))
    return out
