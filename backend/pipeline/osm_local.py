"""Local pre-event OSM snapshots from dated Geofabrik extracts (e.g. nepal-260801.osm.pbf = OSM on 1 Aug 2026).

`python -m pipeline.osm_local data/osm_extracts/nepal-260801.osm.pbf` builds a compact index once
(a few minutes); afterwards any bbox query returns in seconds without depending on public Overpass servers.
Buildings are stored as centroid + equivalent radius to keep the national index small.
"""
from __future__ import annotations

import math
import pickle
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .config import DATA

EXTRACTS = DATA / "osm_extracts"
ROAD_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                "service", "track", "road"}
KEEP_TAGS = ("highway", "name", "name:ne", "name:en", "ref", "bridge", "place", "amenity", "healthcare",
             "waterway", "building")
HEALTH_AMENITY = {"hospital", "clinic", "doctors"}
HEALTH_CARE = {"hospital", "clinic", "centre", "doctor"}
PLACES = {"city", "town", "village", "hamlet"}
M_PER_DEG = 111_320.0


def _tags(t) -> dict:
    return {k: t.get(k) for k in KEEP_TAGS if k in t}


def build_index(pbf: Path) -> Path:
    import osmium

    m = re.search(r"-(\d{6})\.osm\.pbf$", pbf.name)
    if not m:
        raise ValueError("expected a dated Geofabrik file such as nepal-260801.osm.pbf")
    out = pbf.with_suffix("").with_suffix("")  # .../nepal-260801
    out.mkdir(exist_ok=True)
    roads, rivers, places, health = [], [], [], []
    b_lon, b_lat, b_rad = [], [], []
    t0 = time.time()
    fp = (osmium.FileProcessor(str(pbf)).with_locations()
          .with_filter(osmium.filter.KeyFilter("highway", "building", "place", "amenity", "healthcare", "waterway")))
    n = 0
    for obj in fp:
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n:,} objects, {time.time() - t0:.0f}s", flush=True)
        t = obj.tags
        if obj.is_node():
            loc = obj.location
            if not loc.valid():
                continue
            if t.get("place") in PLACES:
                places.append((loc.lon, loc.lat, _tags(t), f"node/{obj.id}"))
            if t.get("amenity") in HEALTH_AMENITY or t.get("healthcare") in HEALTH_CARE:
                health.append((loc.lon, loc.lat, _tags(t), f"node/{obj.id}"))
            continue
        if not obj.is_way():
            continue
        try:
            xy = np.array([(nd.lon, nd.lat) for nd in obj.nodes], dtype=np.float64)
        except osmium.InvalidLocationError:
            continue
        if len(xy) < 2:
            continue
        if "building" in t:
            if len(xy) >= 4:
                lon0, lat0 = xy[:, 0].mean(), xy[:, 1].mean()
                kx = M_PER_DEG * math.cos(math.radians(lat0))
                x, y = (xy[:, 0] - lon0) * kx, (xy[:, 1] - lat0) * M_PER_DEG
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                b_lon.append(lon0)
                b_lat.append(lat0)
                b_rad.append(math.sqrt(area / math.pi))
            continue
        if t.get("highway") in ROAD_CLASSES:
            roads.append((xy.astype(np.float32), _tags(t), f"way/{obj.id}"))
        elif t.get("waterway") == "river":
            rivers.append((xy.astype(np.float32), _tags(t), f"way/{obj.id}"))
        if t.get("amenity") in HEALTH_AMENITY or t.get("healthcare") in HEALTH_CARE:
            health.append((float(xy[:, 0].mean()), float(xy[:, 1].mean()), _tags(t), f"way/{obj.id}"))

    def lines(items):
        bb = np.array([[c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()] for c, _, _ in items]) \
            if items else np.zeros((0, 4))
        return {"items": items, "bbox": bb}

    def points(items):
        return {"items": items, "xy": np.array([[a, b] for a, b, _, _ in items]) if items else np.zeros((0, 2))}

    for name, obj in (("roads", lines(roads)), ("rivers", lines(rivers)), ("places", points(places)),
                      ("health", points(health)),
                      ("buildings", {"lon": np.array(b_lon), "lat": np.array(b_lat),
                                     "radius": np.array(b_rad, dtype=np.float32)})):
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f, protocol=5)
    print(f"indexed {len(roads):,} roads, {len(b_lon):,} buildings, {len(places):,} places, {len(health):,} "
          f"health, {len(rivers):,} rivers in {time.time() - t0:.0f}s -> {out}")
    return out


def available() -> list[tuple[date, Path]]:
    out = []
    for d in EXTRACTS.glob("*-[0-9][0-9][0-9][0-9][0-9][0-9]"):
        if d.is_dir() and (d / "roads.pkl").exists():
            out.append((datetime.strptime(d.name[-6:], "%y%m%d").date(), d))
    return sorted(out)


_cache: dict = {}


def _load(d: Path, layer: str):
    key = (d, layer)
    if key not in _cache:
        with open(d / f"{layer}.pkl", "rb") as f:
            _cache[key] = pickle.load(f)
    return _cache[key]


def covers(d: Path, bbox) -> bool:
    """True if the extract has roads around all four corners of the bbox (cheap coverage test)."""
    r = _load(d, "roads")["bbox"]
    if not len(r):
        return False
    w, s, e, n = bbox
    return r[:, 0].min() <= w and r[:, 1].min() <= s and r[:, 2].max() >= e and r[:, 3].max() >= n


def query(d: Path, layer: str, bbox) -> dict:
    w, s, e, n = bbox
    feats = []
    if layer in ("roads", "rivers"):
        data = _load(d, layer)
        bb = data["bbox"]
        idx = np.where((bb[:, 2] >= w) & (bb[:, 0] <= e) & (bb[:, 3] >= s) & (bb[:, 1] <= n))[0]
        for i in idx:
            c, tags, oid = data["items"][i]
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": c.astype(float).tolist()},
                          "properties": {**tags, "@osmId": oid}})
    elif layer in ("places", "health"):
        data = _load(d, layer)
        xy = data["xy"]
        if len(xy):
            for i in np.where((xy[:, 0] >= w) & (xy[:, 0] <= e) & (xy[:, 1] >= s) & (xy[:, 1] <= n))[0]:
                lon, lat, tags, oid = data["items"][i]
                feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                              "properties": {**tags, "@osmId": oid}})
    elif layer == "buildings":
        data = _load(d, layer)
        lon, lat = data["lon"], data["lat"]
        for i in np.where((lon >= w) & (lon <= e) & (lat >= s) & (lat <= n))[0]:
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon[i], lat[i]]},
                          "properties": {"building": "yes", "_radius": float(data["radius"][i])}})
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    build_index(Path(sys.argv[1]))
