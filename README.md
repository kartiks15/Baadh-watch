# Baadhi Watch: flood damage and cut-off settlements from space

*Baadhi (बाढी) means flood in Nepali.* Built for the Multimodal AI Hackathon 2026, Track B: **Mapping Flood Damage from Space**.

Give it **an area and a flood date**. Starting from raw satellite data, it answers the three questions rescue teams ask:

1. **Where did the flood hit?** Sentinel-1 radar before/after change detection on the same orbit track, a U-Net water model, and Sentinel-2 optical change when the sky is clear, all fused on the valley floor (HAND from the Copernicus DEM).
2. **What was damaged?** Pre-event OpenStreetMap buildings, roads and bridges overlaid on the flood and debris map.
3. **Who is cut off?** The road network before and after. Settlements that lost their road link to the nearest town or hospital, ranked by estimated population, with the first broken point on their route.

It also provides a **one-page situation report in English and Nepali**, a **grounded Q&A copilot** that cannot invent numbers, a **downstream flow-path tracer** (bonus), and **CSV/KML exports** for field and helicopter teams.

> Educational prototype, not an operational tool.

## Quick start

```bash
# 1. Python 3.12 env (uv or pip)
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r backend/requirements.txt osmium

# 2. credentials: copy .env.example to .env and fill in the Sentinel Hub OAuth client (free CDSE account)
#    https://shapps.dataspace.copernicus.eu/dashboard/ → User settings → OAuth clients

# 3. (recommended) local pre-event OSM snapshot for Nepal, used instead of public Overpass servers
curl -L -o data/osm_extracts/nepal-260801.osm.pbf https://download.geofabrik.de/asia/nepal-260801.osm.pbf
cd backend && python -m pipeline.osm_local ../data/osm_extracts/nepal-260801.osm.pbf

# 4. run the case study end-to-end from raw data (command line)
python -m pipeline.run --bbox 84.93,27.78,85.45,28.30 --date 2026-08-26

# 5. dashboard
uvicorn app.main:app --port 8000            # API (from backend/)
cd ../frontend && npm install && npm run dev  # UI on http://localhost:5173
```

`npm run build` in `frontend/` lets the API serve the UI itself at http://localhost:8000.

Outputs go to `data/runs/<id>/`:

| File | Contents |
|---|---|
| `flood.geojson` | flood water and debris polygons, with confidence |
| `buildings_affected.geojson` | building damage |
| `roads_blocked.geojson`, `bridges.geojson` | roads and bridges |
| `settlements.geojson` | cut-off status per settlement |
| `priority.csv`, `cutoff.kml` | priority list for field and helicopter teams |
| `sitrep_en.md`, `sitrep_ne.md` | one-page situation reports |
| `facts.json` | every number the reports may use |
| `affected.tif` | class and evidence-vote raster |
| `s1_*.png` | radar before, after and change images |

## How it works

```
area + date ──► CDSE catalogue: S1 passes; pair post-event with pre-event images on the SAME relative orbit
          ├──► Sentinel Hub: sigma0 VV/VH, orthorectified (Copernicus DEM 30 m), Lee-filtered, + local incidence angle
          ├──► Sentinel-2 L2A: every date screened for cloud over the area (SCL at 100 m), clearest dates
          │    composited per pixel (closest to the event first) — used only where clear
          ├──► Copernicus DEM GLO-30 → slope, D8 flow, HAND (pysheds; channels ∪ pre-event OSM rivers)
          └──► pre-event OSM (dated Geofabrik extract, or Overpass attic query at event−1 day)
evidence in the flood-reachable zone (terrain-adaptive, no layover/shadow):
   mountains (relief > 1 km): HAND < 60 m, slope < 35°  — debris floods surge far above normal flood level
   lowlands:                  HAND < 15 m, slope < 10°
   model      U-Net P(water) after & not before
   sar_change |Δσ0| (7×7 mean) above a CFAR threshold = 99th percentile of the change between the two
              PRE-event images in the same zone (scene-adaptive, ≈1 % false alarms per band)
   s2_change  NDVI loss > 0.25 or |ΔMNDWI| > 0.3 (pixels clear in both composites)
→ affected = any vote; confidence = how many independent sources agree → polygons
→ buildings / roads / bridges intersected → road graph minus blocked edges → multi-source Dijkstra
  from hospitals + towns, before vs after → cut off / detour / connected, first break on the route
→ facts.json → situation report (template, or LLM with placeholder-only numbers)
```

Design choices worth knowing:

- **Same-track pairing.** Different tracks view the terrain from different sides, so post and pre images always share a relative orbit. When an opposite-direction track exists we fuse it as well, which fills radar shadow in narrow gorges.
- **Terrain-adaptive flood zone.** Steep hillslopes cause most SAR false alarms, so evidence is limited to ground the flood could reach (HAND + slope). Debris floods surge far higher than riverine floods, so the zone widens automatically in mountains.
- **Scene-adaptive radar threshold (CFAR).** The change threshold is learnt from the scene's own no-event pair (two pre-event images on the same track), so it transfers to any area and date the judges pick.
- **Transparent fusion.** Every polygon carries how many independent sources agree. The thresholds are fixed values from the literature and were **not tuned on EMSR927**.
- **Grounded copilot.** The language model may only write numbers and settlement names as `{{key}}` placeholders from `facts.json`. Code substitutes the values (in Devanagari digits for Nepali). Any output with a raw digit or an unknown key is rejected and retried, then falls back to a deterministic template. Nepali output must also pass a script check: no Cyrillic or other foreign scripts, and no words glued from Latin and Devanagari. With Nemotron, Nepali reports usually fail this check and fall back to the hand-written Nepali template, while English works well. Unit-tested in `backend/tests/test_core.py`.
- **LLM providers.** Set either key:
  - `ANTHROPIC_API_KEY`: direct API, `claude-opus-5-5`, with server-side refusal fallback.
  - `OPENROUTER_API_KEY`: `OPENROUTER_MODEL`, default `anthropic/claude-opus-5.5`. The demo uses `nvidia/nemotron-3-super-120b-a12b` (NVIDIA Nemotron 3 Super), about $0.001–0.004 per report.

  Without either key, the deterministic template is used.
- **Live robustness.** Everything is cached (Sentinel Hub tiles, DEM, OSM), and re-runs of the same area and date are instant. One request uses only a few hundred Sentinel Hub processing units (10 m resolution for areas ≤ 20 km across, 20 m above that).

## AI component: flood segmentation model

- **Architecture:** U-Net with a ResNet-34 encoder (`segmentation_models_pytorch`).
- **Input:** Sentinel-1 VV/VH sigma0 in dB.
- **Training data:** Sen1Floods11 hand-labelled chips. Training uses CE + Dice loss, random crops, flips and rotations, and radiometric jitter, which makes the model tolerant of calibration differences between processors.
- **Training script:** `ml/train_sen1floods11.py`. It runs on a local GPU or Colab; use `ml/train_sen1floods11.ipynb` for Colab.
- **Inference:** ONNX on CPU (`backend/models/water_unet.onnx`).

| Split (never trained on) | IoU (water) | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.647 | 0.785 | 0.746 | 0.829 |
| Test (held-out chips) | 0.663 | 0.797 | 0.778 | 0.818 |
| **Bolivia (held-out event)** | **0.657** | **0.793** | 0.772 | 0.816 |
| Trishuli 2026 (unseen Himalayan scene), fused system vs EMSR927 | **0.708** | **0.829** | 0.804 | 0.856 |

Full case study vs EMSR927 (all four EMS areas):

| Check | Result |
|---|---|
| Extent IoU / F1 | 0.71 / 0.83; with 2-px tolerance P 0.90, R 0.95 |
| Building recall | 4,503 of 4,685 EMS-damaged buildings fall inside our affected area (0.96) |
| Roads | 74.6 km of our 100 km blocked roads coincide with EMS's 91.8 km |

The run takes 526 s from raw data. See `report/report.md` §5, including the honest account of one post-check change.

## Checking against Copernicus EMS (EMSR927)

1. Download the EMSR927 vector products from https://mapping.emergency.copernicus.eu/activations/EMSR927/.
2. Unzip them under `validation/`.
3. Run:

```bash
cd backend && python -m eval.evaluate_emsr --run <run_id>
```

This reports extent IoU, precision, recall and F1, plus a 2-pixel tolerance variant. It also gives recall of EMS-damaged buildings and blocked-road km against EMS affected roads, and writes an agreement map. **EMS data is read only by `backend/eval/`, never by the pipeline.**

## Data rules compliance

| Allowed input | Where it is used |
|---|---|
| Sentinel-1 GRD | `pipeline/fetch.py`, via Sentinel Hub on CDSE |
| Sentinel-2 L2A | `pipeline/fetch.py` |
| Copernicus DEM GLO-30 | AWS Open Data COGs, plus Sentinel Hub orthorectification |
| Pre-event OSM | Geofabrik `nepal-260801` (state of 1 Aug 2026), or Overpass `[date:"<event−1 day>"]` |
| Training data | Sen1Floods11 |

- **Not used as inputs:** Copernicus EMS, UNOSAT or other damage maps, post-event OSM, and commercial imagery (e.g. Maxar/Planet open data).
- The ohsome API's v1 extraction endpoints now return 403, and v2 requires a key. The same pre-event snapshot therefore comes from Geofabrik's dated extract or Overpass attic queries.
- The display basemap tiles in the dashboard are not analysed.

## Limitations (see report for detail)

- **Revisit time.** Sentinel-1 revisits every 6–12 days, so the system maps aftermath. It cannot warn of a glacier collapse minutes ahead.
- **Steep terrain.** Radar shadow and layover hide parts of steep gorges. The dashboard reports the % of valley floor actually observed.
- **Debris versus wet ground.** Debris and wet soil can look alike in radar. Monsoon rain between acquisitions can cause false change.
- **Clouds.** Monsoon clouds usually make Sentinel-2 unusable.
- **Domain shift.** The model was trained on mostly lowland floods.
- **Bridges are inferred.** At 10–20 m pixels a bridge deck cannot be seen, so a bridge is marked "likely damaged" when it touches an affected channel.
- **Landslides missed.** Landslides on hillslopes outside the valley mask also cut roads, and they are not detected.
- **OSM gaps.** OSM completeness varies: trails, footbridges and new buildings are often missing. Roads outside the analysed area are assumed open.
- **Population is a proxy.** Estimates are mapped buildings × 4.4 (Nepal Census 2021 average household size).

## Attribution and citations

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL). Extracts: Geofabrik GmbH.
- Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- Bountos, N. I. et al. (2024). *Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping.* NeurIPS Datasets & Benchmarks. Cited as the recommended dataset; not used in this 12-hour build.
- Validation reference: European Union, Copernicus Emergency Management Service data (EMSR927).
