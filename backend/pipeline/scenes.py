"""Sentinel-1 scene search (CDSE OData catalogue) and same-track before/after pairing."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import requests
from shapely import wkt
from shapely.geometry import box, shape
from shapely.ops import unary_union

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


@dataclass
class Pass:
    """One satellite pass over the AOI (one or more consecutive GRD slices of a datatake)."""
    start: datetime
    end: datetime
    rel_orbit: int
    direction: str
    platform: str
    names: list[str] = field(default_factory=list)
    coverage: float = 0.0  # fraction of the AOI covered

    @property
    def label(self) -> str:
        return f"S1{self.platform} {self.start:%Y-%m-%d %H:%M} UTC, track {self.rel_orbit} {self.direction.lower()}"

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "rel_orbit": self.rel_orbit,
                "direction": self.direction, "platform": self.platform, "products": self.names,
                "coverage": round(self.coverage, 3), "label": self.label}


@dataclass
class Pair:
    post: Pass
    pres: list[Pass]

    @property
    def gap_days(self) -> float:
        return (self.post.start - self.pres[0].start).total_seconds() / 86400

    def to_dict(self) -> dict:
        return {"post": self.post.to_dict(), "pres": [p.to_dict() for p in self.pres],
                "gap_days": round(self.gap_days, 1)}


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def search_s1(bbox, start: datetime, end: datetime) -> list[dict]:
    w, s, e, n = bbox
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    flt = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
        f"ContentDate/Start gt {start:%Y-%m-%dT%H:%M:%S.000Z} and ContentDate/Start lt {end:%Y-%m-%dT%H:%M:%S.000Z}"
    )
    out, url = [], ODATA
    params = {"$filter": flt, "$expand": "Attributes", "$orderby": "ContentDate/Start asc", "$top": 200}
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        out.extend(js.get("value", []))
        url, params = js.get("@odata.nextLink"), None
    return [p for p in out if "_COG" not in p["Name"]]


def group_passes(products: list[dict], bbox) -> list[Pass]:
    aoi = box(*bbox)
    rows = []
    for p in products:
        attrs = {a["Name"]: a["Value"] for a in p.get("Attributes", [])}
        try:
            geom = shape(p["GeoFootprint"]) if p.get("GeoFootprint") else wkt.loads(p["Footprint"].split(";")[-1].rstrip("'"))
        except Exception:
            geom = aoi
        rows.append(dict(start=_parse_time(p["ContentDate"]["Start"]), end=_parse_time(p["ContentDate"]["End"]),
                         rel=int(attrs.get("relativeOrbitNumber", -1)), dir=str(attrs.get("orbitDirection", "")),
                         plat=str(attrs.get("platformSerialIdentifier", "?")), name=p["Name"], geom=geom))
    rows.sort(key=lambda r: r["start"])
    passes: list[Pass] = []
    geoms: list[list] = []
    for r in rows:
        last = passes[-1] if passes else None
        if last and last.rel_orbit == r["rel"] and last.platform == r["plat"] and \
                (r["start"] - last.end) < timedelta(minutes=3):
            last.end = max(last.end, r["end"])
            last.names.append(r["name"])
            geoms[-1].append(r["geom"])
        else:
            passes.append(Pass(r["start"], r["end"], r["rel"], r["dir"], r["plat"], [r["name"]]))
            geoms.append([r["geom"]])
    for p, g in zip(passes, geoms):
        p.coverage = unary_union(g).intersection(aoi).area / aoi.area
    return passes


def select_pairs(bbox, event_date: date, event_time: str | None = None, max_post_days: int = 14,
                 max_pre_days: int = 36, n_pre: int = 2, max_tracks: int = 2,
                 min_coverage: float = 0.5) -> tuple[list[Pair], list[Pass]]:
    """Return before/after pairs on the same relative orbit, earliest post first.

    Post passes start after the event (same day only if `event_time` HH:MM UTC is given and the pass is later).
    Up to `max_tracks` pairs on distinct tracks are returned, preferring opposite look directions
    so that radar shadow in narrow valleys is filled from the other side.
    """
    t_event = datetime.combine(event_date, datetime.min.time(), tzinfo=timezone.utc)
    if event_time:
        hh, mm = (int(v) for v in event_time.split(":"))
        post_from = t_event + timedelta(hours=hh, minutes=mm)
    else:
        post_from = t_event + timedelta(days=1)
    products = search_s1(bbox, t_event - timedelta(days=max_pre_days), post_from + timedelta(days=max_post_days))
    passes = [p for p in group_passes(products, bbox) if p.coverage >= min_coverage]
    pres_all = [p for p in passes if p.end < t_event]
    posts = [p for p in passes if p.start >= post_from]

    pairs: list[Pair] = []
    for post in posts:
        same = [p for p in pres_all if p.rel_orbit == post.rel_orbit]
        if same:
            pairs.append(Pair(post, sorted(same, key=lambda p: p.start, reverse=True)[:n_pre]))
    chosen: list[Pair] = []
    for pr in pairs:  # earliest post first; one pair per track
        if any(c.post.rel_orbit == pr.post.rel_orbit for c in chosen):
            continue
        if chosen and len(chosen) < max_tracks:
            dirs = {c.post.direction for c in chosen}
            if pr.post.direction in dirs and any(p.post.direction not in dirs for p in pairs
                                                 if p.post.rel_orbit not in {c.post.rel_orbit for c in chosen}):
                continue  # wait for an opposite-direction track
        chosen.append(pr)
        if len(chosen) >= max_tracks:
            break
    return chosen, passes
