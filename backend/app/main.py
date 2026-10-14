"""Baadhi Watch API: start pipeline runs, serve map layers, situation reports, Q&A and flow-path tracing.

    uvicorn app.main:app --port 8000      (from backend/)
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from affine import Affine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyproj import Transformer
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from pipeline import sitrep, terrain
from pipeline.config import LLM_PROVIDER, MODELS, ROOT, RUNS, SH_CLIENT_ID, run_id
from pipeline.model import WaterModel
from pipeline.run import Run

app = FastAPI(title="Baadhi Watch", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_lock = threading.Lock()  # one pipeline run at a time (protects the processing-unit quota)
_active: dict[str, threading.Thread] = {}
SERVED = {".geojson": "application/geo+json", ".png": "image/png", ".csv": "text/csv", ".kml":
          "application/vnd.google-earth.kml+xml", ".md": "text/markdown", ".json": "application/json",
          ".tif": "image/tiff"}
CASE_STUDY = {"name": "Bhote Koshi–Trishuli debris flood (EMSR927)", "bbox": [84.93, 27.78, 85.45, 28.30],
              "event_date": "2026-08-26"}


class RunRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="W,S,E,N")
    event_date: str
    event_time: str | None = None
    res: float | None = None
    use_s2: bool = True


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000)
    lang: str = "en"


class TraceRequest(BaseModel):
    lon: float
    lat: float


def _run_dir(rid: str) -> Path:
    d = RUNS / rid
    if not rid.isalnum() or not d.is_dir():
        raise HTTPException(404, "unknown run")
    return d


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@app.get("/api/health")
def health():
    meta = _read(MODELS / "model_meta.json") or {}
    return {"sentinel_hub": bool(SH_CLIENT_ID), "llm": bool(LLM_PROVIDER), "llm_provider": LLM_PROVIDER,
            "model": WaterModel().name,
            "model_metrics": meta.get("metrics"), "case_study": CASE_STUDY}


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(RUNS.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = _read(d / "run.json")
        if st:
            out.append({k: st.get(k) for k in ("id", "bbox", "event_date", "status", "progress", "finished")})
    return out


@app.post("/api/runs")
def start_run(req: RunRequest):
    w, s, e, n = req.bbox
    if not (w < e and s < n) or (e - w) > 1.0 or (n - s) > 1.0:
        raise HTTPException(400, "bbox must be W<E, S<N and at most 1° on each side")
    rid = run_id(tuple(req.bbox), req.event_date, req.res)
    st = _read(RUNS / rid / "run.json")
    if st and st["status"] == "done":
        return {"id": rid, "status": "done"}
    if rid in _active and _active[rid].is_alive():
        return {"id": rid, "status": "running"}
    run = Run(req.bbox, req.event_date, req.res, req.event_time, use_s2=req.use_s2)
    run._save()

    def work():
        with _lock:
            try:
                run.execute()
            except Exception:  # noqa: BLE001 - error is recorded in run.json
                pass

    t = threading.Thread(target=work, daemon=True)
    _active[rid] = t
    t.start()
    return {"id": rid, "status": "queued"}


@app.get("/api/runs/{rid}")
def get_run(rid: str):
    d = _run_dir(rid)
    st = _read(d / "run.json") or {}
    st.pop("trace", None) if st.get("status") != "error" else None
    return {"state": st, "summary": _read(d / "summary.json")}


@app.get("/api/runs/{rid}/files/{name}")
def get_file(rid: str, name: str):
    d = _run_dir(rid)
    p = d / name
    if p.suffix not in SERVED or p.parent != d or not p.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(p, media_type=SERVED[p.suffix])


@app.get("/api/runs/{rid}/sitrep")
def get_sitrep(rid: str, lang: str = "en", ai: bool = False):
    d = _run_dir(rid)
    facts = _read(d / "facts.json")
    if not facts:
        raise HTTPException(409, "run not finished")
    lang = "ne" if lang == "ne" else "en"
    cache = d / f"sitrep_{'ai' if ai else 'tpl'}_{lang}.json"
    if cache.exists():
        return _read(cache)
    out = sitrep.sitrep(facts, lang, use_llm=ai)
    if ai and not out["source"].startswith("template"):
        cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


@app.post("/api/runs/{rid}/ask")
def ask(rid: str, req: AskRequest):
    d = _run_dir(rid)
    facts = _read(d / "facts.json")
    if not facts:
        raise HTTPException(409, "run not finished")
    return sitrep.answer(facts, req.question, "ne" if req.lang == "ne" else "en")


@app.post("/api/runs/{rid}/trace")
def trace(rid: str, req: TraceRequest):
    """Bonus: follow the D8 flow path downstream from a clicked point; list settlements along the corridor."""
    d = _run_dir(rid)
    summ = _read(d / "summary.json")
    tz = d / "terrain.npz"
    if not summ or not tz.exists():
        raise HTTPException(409, "run not finished")
    z = np.load(tz)
    epsg = int(z["epsg"])
    ter = {"dem": z["dem"], "fdir": z["fdir"], "acc": z["acc"], "transform": Affine(*z["transform"])}
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    x, y = fwd.transform(req.lon, req.lat)
    try:
        path = terrain.trace_downstream(ter, x, y)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    if len(path["coords"]) < 2:
        raise HTTPException(400, "no downstream path from this point")
    line = LineString(path["coords"])
    hand = z["hand"]
    t = ter["transform"]

    def hand_at(xs, ys):
        cols, rows = ~t * (np.asarray(xs), np.asarray(ys))
        rows, cols = rows.astype(int), cols.astype(int)
        ok = (rows >= 0) & (rows < hand.shape[0]) & (cols >= 0) & (cols < hand.shape[1])
        out = np.full(len(rows), np.nan)
        out[ok] = hand[rows[ok], cols[ok]]
        return out

    setts = summ["network"]["settlements"]
    sxy = np.array([[r["x"], r["y"]] for r in setts]) if setts else np.zeros((0, 2))
    # buildings close to river level along the path (the houses a flood wave would actually reach)
    low_count = np.zeros(len(setts), int)
    low_km = np.full(len(setts), np.inf)
    bfile = d / "buildings.npz"
    n_low = 0
    if bfile.exists() and len(sxy):
        bxy = np.load(bfile)["xy"]
        if len(bxy):
            pts = shapely.points(bxy)
            near = shapely.distance(pts, line) <= 1500
            hb = hand_at(bxy[:, 0], bxy[:, 1])
            low = near & np.isfinite(hb) & (hb <= 25)
            n_low = int(low.sum())
            if n_low:
                dd, idx = cKDTree(sxy).query(bxy[low], distance_upper_bound=2000)
                kms = shapely.line_locate_point(line, pts[low]) / 1000
                for j, k, dist_ in zip(idx, kms, dd):
                    if np.isfinite(dist_):
                        low_count[j] += 1
                        low_km[j] = min(low_km[j], k)
    hs = hand_at(sxy[:, 0], sxy[:, 1]) if len(sxy) else np.zeros(0)
    along = []
    for i, r in enumerate(setts):
        p = Point(r["x"], r["y"])
        dist = line.distance(p)
        h = hs[i]
        on_corridor = dist <= 1500 and np.isfinite(h) and h <= 100
        if not on_corridor and low_count[i] == 0:
            continue
        km = low_km[i] if np.isfinite(low_km[i]) else line.project(p) / 1000
        risk = ("high" if low_count[i] > 0 or (np.isfinite(h) and h <= 20) else
                "medium" if np.isfinite(h) and h <= 50 else "low")
        lon, lat = inv.transform(r["x"], r["y"])
        along.append({"name": r["name"], "name_en": r.get("name_en", ""), "km_downstream": round(float(km), 1),
                      "height_above_river_m": None if not np.isfinite(h) else round(float(h), 1),
                      "dist_to_channel_m": round(dist), "buildings_near_river": int(low_count[i]),
                      "est_people_near_river": int(round(low_count[i] * 4.4)),
                      "est_people": r["est_people"], "status": r["status"], "lon": lon, "lat": lat, "risk": risk,
                      # illustrative only: arrival if a flood front moved at 8 m/s (fast) … 3 m/s (slow)
                      "eta_min": [round(km * 1000 / 8 / 60), round(km * 1000 / 3 / 60)]})
    along.sort(key=lambda a: a["km_downstream"])
    coords = [list(inv.transform(cx, cy)) for cx, cy in path["coords"][::3]]
    return {"path": {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                     "properties": {"length_km": round(path["dist_m"][-1] / 1000, 1),
                                    "drop_m": round(path["elev_m"][0] - path["elev_m"][-1])}},
            "settlements": along, "buildings_near_river": n_low,
            "note": "Flow path from Copernicus DEM (D8). Listed: settlements ≤1.5 km from the channel and ≤100 m above "
                    "it, or with buildings ≤25 m above river level along the path (high risk). Arrival times assume "
                    "a front speed of 3–8 m/s and are illustrative, not a forecast."}


@app.get("/api/model")
def model_metrics():
    return _read(MODELS / "metrics.json") or {}


dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
else:
    @app.get("/")
    def root():
        return JSONResponse({"message": "Baadhi Watch API. Build the frontend (npm run build) or run it with "
                                        "npm run dev.", "docs": "/docs"})
