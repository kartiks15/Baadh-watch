"""Who is cut off: road connectivity from each settlement to the nearest town or hospital, before vs after."""
from __future__ import annotations

import heapq

import networkx as nx
import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, box
from shapely.ops import linemerge

from .config import Grid, Thresholds
from .impact import Affected, is_bridge

HOSPITAL_TAGS = ({"amenity": "hospital"}, {"healthcare": "hospital"})
HEALTH_POST_TAGS = ({"amenity": "clinic"}, {"amenity": "doctors"}, {"healthcare": "clinic"},
                    {"healthcare": "centre"}, {"healthcare": "doctor"})


def _key(x: float, y: float) -> tuple:
    return (round(x, 1), round(y, 1))


def _match(tags: dict, options) -> bool:
    return any(all(tags.get(k) == v for k, v in o.items()) for o in options)


def build_graph(road_geoms: list, road_props: list[dict], aff: Affected, damaged_bridges: set,
                th: Thresholds) -> tuple[nx.Graph, list]:
    """Graph keyed on OSM vertex coordinates (ways share nodes at junctions, so topology is preserved)."""
    segs, meta = [], []
    for i, g in enumerate(road_geoms):
        lines = [g] if g.geom_type == "LineString" else list(getattr(g, "geoms", []))
        for line in lines:
            cs = list(line.coords)
            for a, b in zip(cs[:-1], cs[1:]):
                if a != b:
                    segs.append(LineString([a, b]))
                    meta.append(i)
    blocked = np.zeros(len(segs), bool)
    s_idx, _ = aff.hits(segs, th.road_buffer_m)
    blocked[s_idx] = True
    for k, i in enumerate(meta):
        if i in damaged_bridges:
            blocked[k] = True
    G = nx.Graph()
    for seg, i, bl in zip(segs, meta, blocked):
        (x0, y0), (x1, y1) = seg.coords
        u, v = _key(x0, y0), _key(x1, y1)
        L = seg.length
        if G.has_edge(u, v) and G[u][v]["length"] <= L:
            continue
        p = road_props[i]
        G.add_edge(u, v, length=L, way=i, blocked=bool(bl), bridge=is_bridge(p),
                   name=p.get("name", "") or p.get("ref", ""), highway=p.get("highway", ""))
    return G, list(zip(segs, meta, blocked))


def _snap(G: nx.Graph, pts: np.ndarray, max_m: float):
    nodes = list(G.nodes)
    if not nodes or not len(pts):
        return [None] * len(pts), [np.inf] * len(pts)
    tree = cKDTree(np.array(nodes))
    d, idx = tree.query(pts, distance_upper_bound=max_m)
    return [nodes[j] if np.isfinite(dd) else None for dd, j in zip(d, idx)], list(d)


def _dijkstra(G: nx.Graph, sources: dict) -> tuple[dict, dict, dict]:
    """Multi-source Dijkstra: distance, originating source and predecessor for every reachable node."""
    dist, origin, pred = {}, {}, {}
    heap = [(0.0, i, n, n, None) for i, n in enumerate(sources)]
    heapq.heapify(heap)
    counter = len(heap)
    while heap:
        d, _, n, o, p = heapq.heappop(heap)
        if n in dist:
            continue
        dist[n], origin[n], pred[n] = d, o, p
        for nb, attr in G[n].items():
            if nb not in dist:
                counter += 1
                heapq.heappush(heap, (d + attr["length"], counter, nb, o, n))
    return dist, origin, pred


def analyse(G: nx.Graph, places_geoms: list, places_props: list[dict], health_geoms: list, health_props: list[dict],
            building_centroids: np.ndarray, grid: Grid, th: Thresholds) -> dict:
    aoi = box(*grid.bounds)
    G_post = nx.Graph()
    G_post.add_nodes_from(G.nodes)
    G_post.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if not d["blocked"])

    # targets: hospitals + towns/cities
    targets, hospitals = [], []
    for g, p in zip(health_geoms, health_props):
        c = g.centroid
        if _match(p, HOSPITAL_TAGS):
            name = p.get("name", "") or "Hospital"
            targets.append((c.x, c.y, name, "hospital"))
            hospitals.append((c.x, c.y, name, "hospital"))
    for g, p in zip(places_geoms, places_props):
        if p.get("place") in ("town", "city"):
            targets.append((g.x, g.y, p.get("name", "") or "Town", p.get("place")))

    def source_map(items):
        if not items:
            return {}
        nodes, _ = _snap(G, np.array([[x, y] for x, y, *_ in items]), th.snap_max_m)
        return {n: {"name": it[2], "kind": it[3]} for n, it in zip(nodes, items) if n is not None}

    src_all, src_hosp = source_map(targets), source_map(hospitals)
    pre_d, pre_origin, pre_pred = _dijkstra(G, src_all)
    post_d, _, _ = _dijkstra(G_post, src_all)
    hpre_d, _, _ = _dijkstra(G, src_hosp)
    hpost_d, _, _ = _dijkstra(G_post, src_hosp)

    # settlements inside the AOI
    sett = [(g, p) for g, p in zip(places_geoms, places_props) if aoi.contains(g) and p.get("name")]
    pts = np.array([[g.x, g.y] for g, _ in sett]) if sett else np.zeros((0, 2))
    snapped, snap_d = _snap(G, pts, th.snap_max_m)
    btree = cKDTree(building_centroids) if len(building_centroids) else None
    comp_post = {n: i for i, c in enumerate(nx.connected_components(G_post)) for n in c}
    target_comps = {comp_post[n] for n in src_all if n in comp_post}

    rows = []
    for (g, p), node, sd in zip(sett, snapped, snap_d):
        n_bld = len(btree.query_ball_point([g.x, g.y], th.settlement_radius_m)) if btree is not None else 0
        row = {"name": p.get("name"), "name_ne": p.get("name:ne", ""), "name_en": p.get("name:en", ""),
               "place": p.get("place"),
               "x": g.x, "y": g.y, "buildings_500m": n_bld,
               "est_people": int(round(n_bld * th.people_per_building)), "osm_id": p.get("@osmId", "")}
        if node is None or node not in pre_d:
            row.update(status="no_road_pre", pre_km=None, post_km=None, nearest=None, cause=None)
        else:
            pre_km = (pre_d[node] + sd) / 1000
            post_km = (post_d[node] + sd) / 1000 if node in post_d else None
            if post_km is None:
                status = "cut_off"
            elif post_km > th.detour_ratio * pre_km and post_km - pre_km > 1.0:
                status = "detour"
            else:
                status = "connected"
            row.update(status=status, pre_km=round(pre_km, 1), post_km=None if post_km is None else round(post_km, 1),
                       nearest=src_all[pre_origin[node]]["name"])
            row["cause"] = _first_break(G, pre_pred, node) if status in ("cut_off", "detour") else None
        row["hospital_pre_km"] = round((hpre_d[node] + sd) / 1000, 1) if node in hpre_d else None
        row["hospital_post_km"] = round((hpost_d[node] + sd) / 1000, 1) if node in hpost_d else None
        row["hospital_lost"] = row["hospital_pre_km"] is not None and row["hospital_post_km"] is None
        row["isolated_pocket"] = node is not None and comp_post.get(node) not in target_comps
        rows.append(row)

    order = {"cut_off": 0, "detour": 1, "connected": 2, "no_road_pre": 3}
    rows.sort(key=lambda r: (order[r["status"]], -r["est_people"]))
    counts = {k: sum(r["status"] == k for r in rows) for k in order}
    counts["people_cut_off"] = sum(r["est_people"] for r in rows if r["status"] == "cut_off")
    counts["hospital_access_lost"] = sum(r["hospital_lost"] for r in rows)
    counts["targets"] = len(src_all)
    counts["hospitals"] = len(src_hosp)
    return {"settlements": rows, "counts": counts,
            "targets": [{"x": x, "y": y, "name": n, "kind": k} for x, y, n, k in targets]}


def _first_break(G: nx.Graph, pred: dict, node):
    """Blocked edge on the settlement's pre-event route that is closest to the settlement."""
    dist, u = 0.0, node
    while pred.get(u) is not None:
        v = pred[u]
        d = G[u][v]
        if d["blocked"]:
            return {"km_along_route": round(dist / 1000, 1), "bridge": d["bridge"], "road": d["name"],
                    "highway": d["highway"], "x": (u[0] + v[0]) / 2, "y": (u[1] + v[1]) / 2}
        dist += d["length"]
        u = v
    return None


def blocked_roads(segments: list, road_props: list[dict], grid: Grid) -> tuple[list, dict]:
    """Merge blocked segments per way into lines; summary km by road class (inside the AOI)."""
    aoi = box(*grid.bounds)
    by_way: dict[int, list] = {}
    for seg, i, bl in segments:
        if bl:
            by_way.setdefault(i, []).append(seg)
    feats, km = [], {}
    for i, segs in by_way.items():
        merged = linemerge(shapely.MultiLineString([list(s.coords) for s in segs]))
        merged = merged.intersection(aoi)
        if merged.is_empty:
            continue
        p = road_props[i]
        hw = p.get("highway", "road")
        km[hw] = km.get(hw, 0) + merged.length / 1000
        feats.append((merged, {"highway": hw, "name": p.get("name", "") or p.get("ref", ""),
                               "bridge": is_bridge(p), "length_m": round(merged.length), "osm_id": p.get("@osmId", "")}))
    return feats, {k: round(v, 2) for k, v in sorted(km.items(), key=lambda kv: -kv[1])}
