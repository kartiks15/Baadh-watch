"""Pre-event OpenStreetMap (© OpenStreetMap contributors, ODbL).

Data comes from Overpass "attic" queries with `[date:...]` set to the day before the event, which return
OSM exactly as it was at that moment. Edits mapped after the disaster (which would leak post-event damage
information) are therefore never used. The ohsome v1 extraction endpoints suggested by the challenge now
return 403 and v2 needs a separate API key, so Overpass attic queries serve the same pre-event snapshot.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import date, timedelta

import requests

from .config import CACHE

UA = {"User-Agent": "baadhi-watch/0.1 (hackathon flood-mapping prototype)"}
MIRRORS = (  # mirrors that answer attic ([date:...]) queries
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
TILE_DEG = {"roads": 0.15, "buildings": 0.15}  # heavy layers are tiled; light ones go in one request

ROAD_CLASSES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street",
                "service", "track", "road")

QUERIES = {
    "roads": f'way["highway"~"^({"|".join(ROAD_CLASSES)})$"];out tags geom;',
    "buildings": 'way["building"];out tags geom;',
    "places": 'node["place"~"^(city|town|village|hamlet)$"];out tags;',
    "health": '(nwr["amenity"~"^(hospital|clinic|doctors)$"];nwr["healthcare"~"^(hospital|clinic|centre|doctor)$"];);'
              'out tags center;',
    "rivers": 'way["waterway"="river"];out tags geom;',
}


def snapshot_date(event_date: date) -> str:
    """OSM state at 00:00 UTC on the day before the event (always pre-event)."""
    return (event_date - timedelta(days=1)).isoformat()


def buffer_bbox(bbox, km: float):
    w, s, e, n = bbox
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians((s + n) / 2)))
    return (w - dlon, s - dlat, e + dlon, n + dlat)


def _to_geojson(elements: list, layer: str) -> dict:
    feats = []
    for el in elements:
        tags = el.get("tags", {})
        props = {**tags, "@osmId": f"{el['type']}/{el['id']}"}
        geom = None
        if el["type"] == "node" and "lat" in el:
            geom = {"type": "Point", "coordinates": [el["lon"], el["lat"]]}
        elif "center" in el:
            geom = {"type": "Point", "coordinates": [el["center"]["lon"], el["center"]["lat"]]}
        elif "geometry" in el:
            pts = [[p["lon"], p["lat"]] for p in el["geometry"] if p]
            if len(pts) < 2:
                continue
            if layer == "buildings":
                if len(pts) < 4 or pts[0] != pts[-1]:
                    continue
                geom = {"type": "Polygon", "coordinates": [pts]}
            else:
                geom = {"type": "LineString", "coordinates": pts}
        if geom:
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def _query_tile(layer: str, bbox, when: str) -> list:
    w, s, e, n = bbox
    head = f"[out:json][timeout:180][maxsize:1073741824][bbox:{s:.5f},{w:.5f},{n:.5f},{e:.5f}]"
    key = hashlib.sha1((head + when + QUERIES[layer]).encode()).hexdigest()
    path = CACHE / "osm" / "tiles" / f"{layer}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    errors = []
    q = head + f'[date:"{when}T00:00:00Z"];' + QUERIES[layer]  # attic query: OSM as it was at `when`
    for url in MIRRORS:
        for attempt in range(6):
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=150)
            except requests.RequestException as ex:
                errors.append(f"{url}: {ex.__class__.__name__}")
                break
            if r.status_code in (429, 504):  # busy / rate limited: back off and retry
                errors.append(f"{url}: HTTP {r.status_code}")
                time.sleep(4 * (attempt + 1))
                continue
            if r.status_code != 200 or not r.text.lstrip().startswith("{"):
                errors.append(f"{url}: HTTP {r.status_code}")
                break
            js = r.json()
            if "error" in js.get("remark", "").lower():
                errors.append(f"{url}: {js['remark'][:120]}")
                break
            els = js.get("elements", [])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(els), encoding="utf-8")
            return els
    raise RuntimeError(f"Overpass failed for {layer} {bbox}: " + "; ".join(errors))


def _tiles(bbox, step: float):
    w, s, e, n = bbox
    nx, ny = max(1, math.ceil((e - w) / step)), max(1, math.ceil((n - s) / step))
    dx, dy = (e - w) / nx, (n - s) / ny
    return [(w + i * dx, s + j * dy, w + (i + 1) * dx, s + (j + 1) * dy) for i in range(nx) for j in range(ny)]


def query(layer: str, bbox, when: str) -> dict:
    """GeoJSON FeatureCollection (WGS84) of `layer` as it was at `when` (YYYY-MM-DD, 00:00 UTC)."""
    seen, elements = set(), []
    for tile in _tiles(bbox, TILE_DEG.get(layer, 10.0)):  # sequential: public Overpass rate-limits per IP
        for el in _query_tile(layer, tile, when):
            k = (el["type"], el["id"])
            if k not in seen:
                seen.add(k)
                elements.append(el)
    fc = _to_geojson(elements, layer)
    fc["snapshot"] = when
    return fc


def fetch_all(bbox, event_date: date) -> dict:
    """Pre-event OSM layers. Prefers a local dated Geofabrik extract (fast, offline); falls back to Overpass."""
    from . import osm_local

    wide = buffer_bbox(bbox, 15)  # roads/facilities beyond the AOI so routes out of the valley exist
    near = buffer_bbox(bbox, 1)
    layers = {"roads": wide, "buildings": near, "places": wide, "health": wide, "rivers": buffer_bbox(bbox, 10)}
    local = [(d, path) for d, path in osm_local.available() if d < event_date and osm_local.covers(path, bbox)]
    if local:
        d, path = local[-1]
        out = {k: osm_local.query(path, k, b) for k, b in layers.items()}
        out.update(snapshot=d.isoformat(), source=f"Geofabrik extract {path.name}.osm.pbf")
        return out
    when = snapshot_date(event_date)
    out = {k: query(k, b, when) for k, b in layers.items()}
    out.update(snapshot=when, source="Overpass attic query")
    return out
