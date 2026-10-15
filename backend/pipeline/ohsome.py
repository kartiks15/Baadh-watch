"""Pre-event OpenStreetMap from the ohsome API v2 (HeiGIT), the source the challenge recommends.

ohsome serves OSM *history*, so asking for a single timestamp returns the map exactly as it was at that
moment: edits made after the disaster never leak in. Works for any area and any date since 2007.
Needs a free API key (https://account.heigit.org/signup) in OHSOME_API_KEY. Results are cached on disk.
Data model: https://docs.ohsome.org/ohsome-api/v2/reference/data_model.html
"""
from __future__ import annotations

import hashlib
import io
import json
import time

import pyarrow.parquet as pq
import requests
import shapely
from shapely.geometry import mapping

from .config import CACHE, OHSOME_API_KEY, OHSOME_URL

ROAD_CLASSES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                "service", "track", "road")
FILTERS = {  # ohsome filter language: https://docs.ohsome.org/ohsome-api/v2/reference/filter.html
    "roads": f"type:way and highway in ({', '.join(ROAD_CLASSES)})",
    "buildings": "building=* and geometry:polygon",
    "places": "type:node and place in (city, town, village, hamlet)",
    "health": "amenity in (hospital, clinic, doctors) or healthcare in (hospital, clinic, centre, doctor)",
    "rivers": "type:way and waterway=river",
}
POINT_LAYERS = {"places", "health"}  # the pipeline expects one point per facility / settlement


class OhsomeError(RuntimeError):
    pass


def available() -> bool:
    return bool(OHSOME_API_KEY)


def _features(table, layer: str) -> list[dict]:
    cols = table.column_names
    types, ids = table.column("osm_type").to_pylist(), table.column("osm_id").to_pylist()
    tags = table.column("tags").to_pylist() if "tags" in cols else [None] * table.num_rows
    geoms = shapely.from_wkb(table.column("geom").to_pylist())
    feats = []
    for t, i, tg, g in zip(types, ids, tags, geoms):
        if g is None or g.is_empty:
            continue
        props = dict(tg) if isinstance(tg, list) else (tg or {})  # MAP arrives as [(key, value), ...]
        props["@osmId"] = f"{t}/{i}"
        if layer in POINT_LAYERS and g.geom_type != "Point":
            g = g.representative_point()
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": props})
    return feats


def query(layer: str, bbox, when: str, deadline: float = float("inf"), check=None) -> dict:
    """GeoJSON FeatureCollection (WGS84) of `layer` as OSM was at `when` (YYYY-MM-DD, 00:00 UTC)."""
    if not OHSOME_API_KEY:
        raise OhsomeError("OHSOME_API_KEY is not set")
    body = {"filter": FILTERS[layer], "aoi": [round(v, 5) for v in bbox], "time": f"{when}T00:00:00Z", "clip": False}
    key = hashlib.sha1(json.dumps([OHSOME_URL, body], sort_keys=True).encode()).hexdigest()
    path = CACHE / "osm" / "ohsome" / f"{layer}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    headers = {"authorization": OHSOME_API_KEY, "Content-Type": "application/json"}
    for attempt in range(6):
        if check:
            check()
        left = deadline - time.time()
        if left <= 5:
            raise OhsomeError("the ohsome API did not answer in time")
        try:
            r = requests.post(f"{OHSOME_URL}/extraction/features.parquet", json=body, headers=headers,
                              timeout=min(300, left))
        except requests.RequestException as ex:
            if attempt == 5:
                raise OhsomeError(f"ohsome API unreachable ({ex.__class__.__name__})") from ex
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 200:
            fc = {"type": "FeatureCollection", "features": _features(pq.read_table(io.BytesIO(r.content)), layer),
                  "snapshot": when, "source": "ohsome API v2"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fc), encoding="utf-8")
            return fc
        if r.status_code in (401, 403):
            raise OhsomeError("the ohsome API key was rejected (check OHSOME_API_KEY in .env)")
        if r.status_code in (429, 502, 503, 504):  # quota / busy: back off and retry
            time.sleep(min(10 * (attempt + 1), max(deadline - time.time() - 5, 0)))
            continue
        raise OhsomeError(f"ohsome API error {r.status_code}: {r.text[:200]}")
    raise OhsomeError("the ohsome API stayed busy or over quota")
