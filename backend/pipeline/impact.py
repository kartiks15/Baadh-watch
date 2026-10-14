"""What was damaged: overlay pre-event OSM buildings and bridges on the detected flood/debris areas."""
from __future__ import annotations

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform

from .config import Grid, Thresholds

CONF_RANK = {"low": 1, "medium": 2, "high": 3}
BRIDGE_VALUES_NO = {"no", "false", "0"}


def project_fc(fc: dict, epsg: int) -> tuple[list, list[dict]]:
    """GeoJSON FeatureCollection (WGS84) → (UTM shapely geometries, properties)."""
    t = Transformer.from_crs(4326, epsg, always_xy=True).transform
    geoms, props = [], []
    for f in fc.get("features", []):
        try:
            g = shp_transform(t, shape(f["geometry"]))
        except Exception:
            continue
        if g.is_empty:
            continue
        geoms.append(g)
        props.append(f.get("properties", {}))
    return geoms, props


class Affected:
    """Spatial index over detected flood/debris polygons (UTM)."""

    def __init__(self, polys: list[tuple]):
        self.geoms = np.array([g for g, _ in polys], dtype=object)
        self.props = [p for _, p in polys]
        self.tree = shapely.STRtree(self.geoms) if len(self.geoms) else None

    def hits(self, geoms, distance: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        if self.tree is None or not len(geoms):
            return np.array([], int), np.array([], int)
        arr = np.asarray(geoms, dtype=object)
        if distance > 0:
            idx = self.tree.query(arr, predicate="dwithin", distance=distance)
        else:
            idx = self.tree.query(arr, predicate="intersects")
        return idx[0], idx[1]


def assess_buildings(fc: dict, aff: Affected, grid: Grid, th: Thresholds) -> dict:
    geoms, props = project_fc(fc, grid.epsg)
    # local extracts store buildings as centroid + equivalent radius: turn them into footprint-sized discs
    geoms = [g.buffer(max(p.get("_radius", 4.0), 2.0), 8) if g.geom_type == "Point" else g
             for g, p in zip(geoms, props)]
    aoi = box(*grid.bounds)
    inside = [i for i, g in enumerate(geoms) if aoi.contains(g.centroid)]
    geoms = [geoms[i] for i in inside]
    props = [props[i] for i in inside]
    n = len(geoms)
    status = np.array(["ok"] * n, dtype=object)
    conf = np.zeros(n, int)
    b_idx, p_idx = aff.hits(geoms)
    for b, p in zip(b_idx, p_idx):
        cls = aff.props[p]["class"]
        new = "in_water" if cls == "water" else "in_debris"
        if status[b] != "in_water":
            status[b] = new
        conf[b] = max(conf[b], CONF_RANK[aff.props[p]["confidence"]])
    b_idx, p_idx = aff.hits(geoms, th.building_risk_buffer_m)
    for b, p in zip(b_idx, p_idx):
        if status[b] == "ok":
            status[b] = "at_risk"
            conf[b] = max(conf[b], CONF_RANK[aff.props[p]["confidence"]])
    counts = {k: int((status == k).sum()) for k in ("in_water", "in_debris", "at_risk", "ok")}
    counts["total"] = n
    features = []
    for g, p, s, c in zip(geoms, props, status, conf):
        if s == "ok":
            continue
        features.append((g, {"status": s, "confidence": {1: "low", 2: "medium", 3: "high"}.get(int(c), "low"),
                             "building": p.get("building", "yes"), "name": p.get("name", ""),
                             "osm_id": p.get("@osmId", "")}))
    centroids = np.array([[g.centroid.x, g.centroid.y] for g in geoms]) if n else np.zeros((0, 2))
    return {"counts": counts, "features": features, "centroids": centroids, "status": status}


def is_bridge(tags: dict) -> bool:
    v = str(tags.get("bridge", "no")).lower()
    return v not in BRIDGE_VALUES_NO


def assess_bridges(road_geoms: list, road_props: list[dict], aff: Affected, grid: Grid, th: Thresholds) -> dict:
    """Road bridges within `bridge_buffer_m` of an affected channel are marked as likely damaged / washed out."""
    aoi = box(*grid.bounds)
    idx = [i for i, p in enumerate(road_props) if is_bridge(p) and aoi.intersects(road_geoms[i])]
    geoms = [road_geoms[i] for i in idx]
    damaged = set()
    b_idx, _ = aff.hits(geoms, th.bridge_buffer_m)
    for b in b_idx:
        damaged.add(idx[b])
    features = []
    for i in idx:
        p = road_props[i]
        features.append((road_geoms[i], {"status": "likely_damaged" if i in damaged else "no_flood_detected",
                                         "name": p.get("name", "") or p.get("bridge:name", ""),
                                         "highway": p.get("highway", ""), "osm_id": p.get("@osmId", "")}))
    return {"damaged_idx": damaged, "features": features,
            "counts": {"total": len(idx), "likely_damaged": len(damaged)}}
