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
TILE_DEG = {"roads": 0.2, "buildings": 0.1}  # heavy layers are tiled on a fixed grid; light ones go in one request

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


class OverpassTimeout(RuntimeError):
    pass


BUSY_MESSAGE = ("Couldn't download the pre-flood map (OpenStreetMap) for this area: the free public map servers are "
                "overloaded right now (HTTP 504 'busy'). Parts already downloaded are saved, so click Run analysis "
                "again in a few minutes to resume. Areas in Nepal use a stored map and are not affected. For reliable "
                "access anywhere, add a free ohsome API key (OHSOME_API_KEY in .env).")


def _query_tile(layer: str, bbox, when: str, deadline: float = float("inf"), check=None) -> list:
    """One attic query, cached on disk. Busy servers (429/504) are retried with backoff until `deadline`."""
    w, s, e, n = bbox
    head = f"[out:json][timeout:180][maxsize:1073741824][bbox:{s:.5f},{w:.5f},{n:.5f},{e:.5f}]"
    key = hashlib.sha1((head + when + QUERIES[layer]).encode()).hexdigest()
    path = CACHE / "osm" / "tiles" / f"{layer}_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    q = head + f'[date:"{when}T00:00:00Z"];' + QUERIES[layer]  # attic query: OSM as it was at `when`
    attempt = 0
    while True:
        for url in MIRRORS:
            if check:
                check()
            left = deadline - time.time()
            if left <= 5:
                raise OverpassTimeout(BUSY_MESSAGE)
            try:
                r = requests.post(url, data={"data": q}, headers=UA, timeout=min(90 if url == MIRRORS[0] else 45, left))
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                js = r.json()
                if "error" not in js.get("remark", "").lower():
                    els = js.get("elements", [])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(els), encoding="utf-8")
                    return els
        attempt += 1  # every mirror busy or failing: wait a little longer each round, then try again
        time.sleep(min(5 * attempt, 20, max(deadline - time.time() - 5, 0)))


def _tiles(bbox, step: float):
    """Tiles snapped to a fixed global grid, so different areas reuse each other's cached downloads."""
    w, s, e, n = bbox
    i0, i1 = math.floor(w / step), math.ceil(e / step)
    j0, j1 = math.floor(s / step), math.ceil(n / step)
    return [(round(i * step, 6), round(j * step, 6), round((i + 1) * step, 6), round((j + 1) * step, 6))
            for i in range(i0, i1) for j in range(j0, j1)]


def query(layer: str, bbox, when: str, deadline: float = float("inf"), check=None) -> dict:
    """GeoJSON FeatureCollection (WGS84) of `layer` as it was at `when` (YYYY-MM-DD, 00:00 UTC)."""
    seen, elements = set(), []
    tiles = _tiles(bbox, TILE_DEG[layer]) if layer in TILE_DEG else [tuple(bbox)]  # light layers: one request
    for tile in tiles:  # sequential: public Overpass rate-limits per IP
        for el in _query_tile(layer, tile, when, deadline, check):
            k = (el["type"], el["id"])
            if k not in seen:
                seen.add(k)
                elements.append(el)
    fc = _to_geojson(elements, layer)
    fc["snapshot"] = when
    return fc


def fetch_all(bbox, event_date: date, check=None, budget_s: float = 420, log=None) -> dict:
    """Pre-event OSM layers, from the first source that works:
    1. a local dated Geofabrik extract covering the area (Nepal: instant, offline);
    2. the ohsome API v2 (any area and date; needs OHSOME_API_KEY);
    3. public Overpass attic queries (no key, but often overloaded)."""
    from . import ohsome, osm_local

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
    deadline = time.time() + budget_s
    if ohsome.available():
        try:
            out = {k: ohsome.query(k, b, when, deadline, check) for k, b in layers.items()}
            out.update(snapshot=when, source="ohsome API v2 (HeiGIT), OSM history")
            return out
        except ohsome.OhsomeError as ex:  # fall back to public Overpass, and say why
            if log:
                log(f"ohsome unavailable ({ex}); trying public Overpass servers")
    out = {k: query(k, b, when, deadline, check) for k, b in layers.items()}
    out.update(snapshot=when, source="Overpass attic query")
    return out
