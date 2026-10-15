"""End-to-end pipeline: area + flood date → flood/debris map, damage, cut-off settlements, situation report.

    python -m pipeline.run --bbox 85.12,27.88,85.45,28.30 --date 2026-08-26
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform

from . import detect, fetch, impact, network, osm, scenes, sitrep, terrain
from .config import RUNS, Grid, Thresholds, make_grid, run_id, thresholds_dict
from .model import WaterModel

STEPS = ["scenes", "osm", "terrain", "radar", "optical", "detect", "impact", "network", "report"]
CANCELLED: set[str] = set()  # run ids the API asked to stop


class RunCancelled(Exception):
    pass


class Run:
    def __init__(self, bbox, event_date: str, res: float | None = None, event_time: str | None = None,
                 max_tracks: int = 2, use_s2: bool = True):
        self.bbox = tuple(float(v) for v in bbox)
        self.event_date = event_date
        self.res = res
        self.event_time = event_time
        self.max_tracks = max_tracks
        self.use_s2 = use_s2
        self.id = run_id(self.bbox, event_date, res)
        self.dir = RUNS / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.th = Thresholds()
        self.state = {"id": self.id, "bbox": list(self.bbox), "event_date": event_date, "status": "queued",
                      "step": None, "progress": 0, "log": [], "started": None, "finished": None, "error": None}

    # ------------------------------------------------------------------ bookkeeping
    def _save(self):
        (self.dir / "run.json").write_text(json.dumps(self.state, indent=1, default=str), encoding="utf-8")

    def check_cancel(self):
        if self.id in CANCELLED:
            raise RunCancelled()

    def log(self, step: str, msg: str):
        self.check_cancel()
        self.state["step"] = step
        self.state["progress"] = round(100 * STEPS.index(step) / len(STEPS))
        self.state["log"].append(f"{datetime.now():%H:%M:%S} [{step}] {msg}")
        try:
            print(self.state["log"][-1], flush=True)
        except UnicodeEncodeError:  # Windows consoles without UTF-8
            print(self.state["log"][-1].encode("ascii", "replace").decode(), flush=True)
        self._save()

    def to_ll(self, g):
        return shp_transform(Transformer.from_crs(self.grid.epsg, 4326, always_xy=True).transform, g)

    def write_fc(self, name: str, items: list[tuple]):
        feats = []
        for g, props in items:
            if g is None or g.is_empty:
                continue
            gj = mapping(self.to_ll(g))
            feats.append({"type": "Feature", "geometry": _round(gj), "properties": props})
        (self.dir / f"{name}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                                  encoding="utf-8")

    # ------------------------------------------------------------------ pipeline
    def execute(self) -> dict:
        self.state.update(status="running", started=datetime.now(timezone.utc).isoformat())
        self._save()
        t0 = time.time()
        try:
            out = self._execute()
            self.state.update(status="done", progress=100, step="done",
                              finished=datetime.now(timezone.utc).isoformat(), seconds=round(time.time() - t0))
            self._save()
            return out
        except RunCancelled:
            CANCELLED.discard(self.id)
            self.state.update(status="cancelled", error="Cancelled by user.",
                              finished=datetime.now(timezone.utc).isoformat())
            self._save()
            raise
        except Exception as ex:  # noqa: BLE001
            friendly = isinstance(ex, osm.OverpassTimeout)  # already a plain-language message
            self.state.update(status="error", error=str(ex) if friendly else f"{ex.__class__.__name__}: {ex}",
                              trace=traceback.format_exc()[-3000:])
            self._save()
            raise

    def _execute(self) -> dict:
        ev = date.fromisoformat(self.event_date)
        self.grid = grid = make_grid(self.bbox, self.res)
        th = self.th

        self.log("scenes", f"grid {grid.width}x{grid.height} px @ {grid.res:g} m, EPSG:{grid.epsg}")
        pairs, passes = scenes.select_pairs(self.bbox, ev, self.event_time, max_tracks=self.max_tracks)
        if not pairs:
            raise RuntimeError("No Sentinel-1 before/after pair on the same orbit track covers this area and date.")
        for p in pairs:
            self.log("scenes", f"after {p.post.label} | before {', '.join(x.label for x in p.pres)}")

        self.log("osm", "pre-event OpenStreetMap snapshot")
        o = osm.fetch_all(self.bbox, ev, check=self.check_cancel, log=lambda m: self.log("osm", m))
        self.log("osm", f"snapshot {o['snapshot']} ({o.get('source')}): {len(o['roads']['features'])} roads, "
                        f"{len(o['buildings']['features'])} buildings, {len(o['places']['features'])} places")

        self.log("terrain", "Copernicus DEM GLO-30 -> slope, HAND")
        x0, y0, x1, y1 = grid.bounds
        buf = 10_000
        dem, dem_t = fetch.fetch_dem((x0 - buf, y0 - buf, x1 + buf, y1 + buf), grid.epsg)
        rivers, _ = impact.project_fc(o["rivers"], grid.epsg)
        ter = terrain.compute_terrain(dem, dem_t, grid.epsg, rivers=rivers)
        np.savez_compressed(self.dir / "terrain.npz", dem=ter["dem"], fdir=ter["fdir"], acc=ter["acc"],
                            hand=ter["hand"], transform=np.array(dem_t)[:6], epsg=grid.epsg)
        hand = terrain.to_grid(ter["hand"], dem_t, grid.epsg, grid)
        slope = terrain.to_grid(ter["slope"], dem_t, grid.epsg, grid)
        dem_g = terrain.to_grid(ter["dem"], dem_t, grid.epsg, grid)
        relief = float(np.nanpercentile(dem_g, 95) - np.nanpercentile(dem_g, 5))
        mountain = relief > th.relief_mountain_m
        hmax, smax = (th.hand_max_m, th.slope_max_deg) if mountain else (th.hand_max_lowland_m, th.slope_max_lowland_deg)
        valley = (hand < hmax) & (slope < smax)
        self.regime = {"regime": "mountain (debris flood)" if mountain else "lowland (riverine flood)",
                       "relief_m": round(relief), "hand_max_m": hmax, "slope_max_deg": smax}
        self.log("terrain", f"{self.regime['regime']}: relief {relief:.0f} m -> zone HAND<{hmax:g} m, slope<{smax:g} deg")

        model = WaterModel()
        tracks = []
        for p in pairs:
            self.log("radar", f"Sentinel-1 {p.post.label}")
            post = fetch.fetch_s1(grid, p.post)
            self.check_cancel()
            pres = [fetch.fetch_s1(grid, x) for x in p.pres]
            tracks.append(detect.track_evidence(post, pres, valley, model, th))

        s2, s2_note = None, "not used"
        if self.use_s2:
            self.log("optical", "Sentinel-2 L2A: screening dates for clear sky over the area")
            evt = datetime.combine(ev, datetime.min.time())
            try:
                s2_pre = fetch.s2_composite(grid, self.bbox, evt - timedelta(days=45), evt - timedelta(days=1), "latest")
                s2_post = fetch.s2_composite(grid, self.bbox, evt, evt + timedelta(days=14), "earliest")
                if s2_pre is None or s2_post is None:
                    s2_note = "skipped: no sufficiently clear Sentinel-2 date before and after the event"
                else:
                    s2 = detect.s2_evidence(s2_pre, s2_post, valley, th)
                    both = round(100 * float((s2_pre["clear"] & s2_post["clear"]).mean()), 1)
                    s2_note = (f"before {', '.join(s2_pre['days'])}; after {', '.join(s2_post['days'])}; "
                               f"{both}% of area clear in both")
                    self._write_optical(s2_pre, s2_post)
            except Exception as ex:  # noqa: BLE001 - optical data is optional evidence
                s2_note = f"unavailable ({ex.__class__.__name__}: {str(ex)[:80]})"
            self.log("optical", s2_note)

        self.log("detect", f"fusing evidence ({model.name})")
        fused = detect.fuse(tracks, s2, valley, grid, th)
        polys = detect.vectorize(fused["class"], fused["votes"], grid)
        self._write_rasters(tracks, fused, valley)
        self.write_fc("flood", polys)
        self.log("detect", f"{fused['stats']['affected_ha']} ha affected in {len(polys)} polygons")

        self.log("impact", "buildings, bridges")
        aff = impact.Affected(polys)
        bld = impact.assess_buildings(o["buildings"], aff, grid, th)
        road_geoms, road_props = impact.project_fc(o["roads"], grid.epsg)
        bridges = impact.assess_bridges(road_geoms, road_props, aff, grid, th)
        self.write_fc("buildings_affected", bld["features"])
        np.savez_compressed(self.dir / "buildings.npz", xy=bld["centroids"])  # for flow-path exposure
        self.write_fc("bridges", bridges["features"])

        self.log("network", "road graph before vs after, cut-off settlements")
        G, segs = network.build_graph(road_geoms, road_props, aff, bridges["damaged_idx"], th)
        place_geoms, place_props = impact.project_fc(o["places"], grid.epsg)
        health_geoms, health_props = impact.project_fc(o["health"], grid.epsg)
        net = network.analyse(G, place_geoms, place_props, health_geoms, health_props, bld["centroids"], grid, th)
        blocked, km = network.blocked_roads(segs, road_props, grid)
        to_ll = Transformer.from_crs(grid.epsg, 4326, always_xy=True)
        for r in net["settlements"]:
            r["lon"], r["lat"] = (round(v, 6) for v in to_ll.transform(r["x"], r["y"]))
        self.write_fc("roads_blocked", blocked)
        self._write_roads(road_geoms, road_props)
        self._write_settlements(net)

        self.log("report", "facts, situation report, exports")
        summary = {
            "id": self.id, "bbox": list(self.bbox), "event_date": self.event_date,
            "grid": {"epsg": grid.epsg, "res": grid.res, "width": grid.width, "height": grid.height,
                     "corners": grid.corners_lonlat()},
            "pairs": [p.to_dict() for p in pairs], "s2_note": s2_note, "osm_snapshot": o["snapshot"], "osm_source": o.get("source"),
            "model": model.name, "detect": {**fused["stats"], "zone": self.regime}, "buildings": bld["counts"],
            "bridges": bridges["counts"], "roads": {"km_by_class": km},
            "network": {"counts": net["counts"], "settlements": net["settlements"]},
            "thresholds": thresholds_dict(th),
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        facts = sitrep.build_facts(summary)
        sitrep.save(facts, self.dir / "facts.json")
        for lang in ("en", "ne"):
            (self.dir / f"sitrep_{lang}.md").write_text(sitrep.sitrep(facts, lang, use_llm=False)["markdown"],
                                                       encoding="utf-8")
        self._write_exports(net)
        return summary

    # ------------------------------------------------------------------ outputs
    def _png(self, name: str, rgba: np.ndarray, max_px: int = 2048):
        img = Image.fromarray(rgba, "RGBA")
        if max(img.size) > max_px:
            s = max_px / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
        img.save(self.dir / f"{name}.png", optimize=True)

    def _write_rasters(self, tracks, fused, valley):
        t = tracks[0]

        def gray(db):
            v = np.clip((np.nan_to_num(db, nan=-30) + 25) / 30, 0, 1)
            return (v * 255).astype(np.uint8)

        alpha = np.where(np.isfinite(t["post_vv"]) & np.isfinite(t["pre_vv"]), 255, 0).astype(np.uint8)
        pre, post = gray(t["pre_vv"]), gray(t["post_vv"])
        self._png("s1_before", np.dstack([pre, pre, pre, alpha]))
        self._png("s1_after", np.dstack([post, post, post, alpha]))
        # change composite: red = darkened (water / wet mud), cyan = brightened (debris, boulders)
        self._png("s1_change", np.dstack([pre, post, post, alpha]))
        v = np.zeros((*valley.shape, 4), np.uint8)
        v[valley] = (90, 160, 255, 70)
        self._png("valley", v)
        with rasterio.open(self.dir / "affected.tif", "w", driver="GTiff", width=self.grid.width,
                           height=self.grid.height, count=2, dtype="uint8", crs=self.grid.crs,
                           transform=self.grid.transform, compress="deflate") as dst:
            dst.write(fused["class"], 1)
            dst.write(fused["votes"], 2)
            dst.set_band_description(1, "class: 0 none, 1 water, 2 debris")
            dst.set_band_description(2, "evidence votes")

    def _write_optical(self, pre: dict, post: dict):
        for name, img in (("s2_before", pre), ("s2_after", post)):
            rgb = np.dstack([img["b4"], img["b3"], img["b2"]])
            rgb = np.clip(np.nan_to_num(rgb) / 0.25, 0, 1) ** 0.8  # simple true-colour stretch
            alpha = np.where(img["clear"], 255, 0).astype(np.uint8)
            self._png(name, np.dstack([(rgb * 255).astype(np.uint8), alpha]))

    def _write_roads(self, geoms, props):
        aoi = box(*self.grid.bounds).buffer(2000)
        keep = [(g.intersection(aoi).simplify(5), {"highway": p.get("highway", ""), "name": p.get("name", ""),
                                                    "bridge": impact.is_bridge(p)})
                for g, p in zip(geoms, props) if g.intersects(aoi)]
        self.write_fc("roads", keep)

    def _write_settlements(self, net):
        from shapely.geometry import Point

        items = []
        for r in net["settlements"]:
            props = {k: v for k, v in r.items() if k not in ("x", "y", "cause")}
            if r.get("cause"):
                props["break_km"] = r["cause"]["km_along_route"]
                props["break_is_bridge"] = r["cause"]["bridge"]
                props["break_road"] = r["cause"]["road"]
            items.append((Point(r["x"], r["y"]), props))
        self.write_fc("settlements", items)
        self.write_fc("targets", [(Point(t["x"], t["y"]), {"name": t["name"], "kind": t["kind"]})
                                  for t in net["targets"]])
        breaks = [(Point(r["cause"]["x"], r["cause"]["y"]), {"settlement": r["name"], "bridge": r["cause"]["bridge"],
                                                             "road": r["cause"]["road"]})
                  for r in net["settlements"] if r.get("cause")]
        self.write_fc("breaks", breaks)

    def _write_exports(self, net):
        tr = Transformer.from_crs(self.grid.epsg, 4326, always_xy=True)
        rows = [r for r in net["settlements"] if r["status"] in ("cut_off", "detour")]
        with open(self.dir / "priority.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rank", "name", "status", "est_people", "lat", "lon", "nearest_before", "road_km_before",
                        "road_km_after", "hospital_km_after", "first_break_km", "first_break_is_bridge"])
            for i, r in enumerate(rows, 1):
                lon, lat = tr.transform(r["x"], r["y"])
                c = r.get("cause") or {}
                w.writerow([i, r["name"], r["status"], r["est_people"], round(lat, 5), round(lon, 5), r["nearest"],
                            r["pre_km"], r["post_km"], r["hospital_post_km"], c.get("km_along_route"), c.get("bridge")])
        pms = []
        for r in rows:
            lon, lat = tr.transform(r["x"], r["y"])
            pms.append(f"<Placemark><name>{escape(str(r['name']))}</name><description>{escape(r['status'])}, "
                       f"~{r['est_people']} people</description><Point><coordinates>{lon:.5f},{lat:.5f}</coordinates>"
                       f"</Point></Placemark>")
        (self.dir / "cutoff.kml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            f"<name>Cut-off settlements {self.event_date}</name>{''.join(pms)}</Document></kml>", encoding="utf-8")


def _round(gj):
    def r(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(c[0], 6), round(c[1], 6)]
        return [r(x) for x in c]
    gj = dict(gj)
    gj["coordinates"] = r(gj["coordinates"])
    return gj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", required=True, help="W,S,E,N in degrees")
    ap.add_argument("--date", required=True, help="flood date YYYY-MM-DD")
    ap.add_argument("--time", default=None, help="event time HH:MM UTC (allows same-day post images)")
    ap.add_argument("--res", type=float, default=None, help="pixel size in m (default 10 m, 20 m for large areas)")
    ap.add_argument("--tracks", type=int, default=2, help="max orbit tracks to fuse")
    ap.add_argument("--no-s2", action="store_true")
    a = ap.parse_args()
    bbox = [float(v) for v in a.bbox.split(",")]
    run = Run(bbox, a.date, a.res, a.time, a.tracks, not a.no_s2)
    out = run.execute()
    print(json.dumps({k: out[k] for k in ("detect", "buildings", "bridges")} | {"network": out["network"]["counts"]},
                     indent=1))
    print("outputs in", run.dir)


if __name__ == "__main__":
    main()
