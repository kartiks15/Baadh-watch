# Baadhi Watch: the plain-language guide

This guide explains every part of the project in simple terms so you can present it in the 3-minute demo and answer judges' questions. The technical README is `README.md`; the 6-page report is `report/report.md`.

> *Baadhi* (बाढी) means *flood* in Nepali.

---

## 1. The 20-second version

On 26 August 2026 a glacier collapse sent a flood of water, mud and boulders down the Bhote Koshi–Trishuli valley in Nepal. Roads, bridges and phone lines failed, so rescuers could not tell which villages were hit or cut off.

**Baadhi Watch takes any area and a flood date, downloads free satellite data, and answers three questions automatically:**

1. **Where did the flood hit?** It produces a map of flood water and debris.
2. **What was damaged?** It lists buildings, roads and bridges hit.
3. **Who is cut off?** It lists villages that lost their road to the nearest town or hospital, ranked by how many people live there.

It then writes a one-page situation report in English and Nepali, and exports a priority list for helicopter and field teams.

On the real flood it matched the official Copernicus emergency maps with an **F1 score of 0.83**. **96% of the buildings Copernicus marked as damaged** fall inside our detected flood area.

---

## 2. The challenge and how it is scored

| Judging criterion | Weight | What we built for it |
|---|---|---|
| Flood and damage mapping quality | 30% | Radar + optical + AI fusion; compared with EMSR927 (F1 0.83) |
| AI component | 25% | U-Net flood segmentation model; plus a grounded EN/Nepali copilot |
| Cut-off settlement analysis | 20% | Road-network routing before vs after the flood |
| Report, Q&A, honesty about limitations | 15% | Report with a limitations section and an honest iteration log |
| Dashboard and situation report usability | 10% | Web dashboard, one-page sitrep, CSV/KML exports |

The data rules are strict, and breaking them means disqualification:
- **Allowed inputs:** Sentinel-1, Sentinel-2, the Copernicus DEM, OpenStreetMap as it was *before* the event, and the listed training data.
- **Never allowed as inputs:** Copernicus EMS or other damage maps, which may only be used to *check* results, and post-event OSM edits.

We follow both rules (see §6).

The system must also run on **any area and date the judges choose on the day**. That is why nothing is hard-coded to Trishuli.

---

## 3. The big picture

```
 You give: an area (box on the map) + a flood date
                     │
   ┌─────────────────┼──────────────────────────────────────────────┐
   │ 1. FIND IMAGES  │ Sentinel-1 radar passes before & after,       │
   │                 │ on the SAME orbit track                       │
   │ 2. DOWNLOAD     │ Radar (sees through clouds), optical (when    │
   │                 │ clear), elevation model, pre-event OSM map    │
   │ 3. TERRAIN      │ Where could floodwater physically reach?      │
   │ 4. DETECT       │ Three independent "votes" per pixel:          │
   │                 │  • AI water model  • radar change  • optical  │
   │ 5. DAMAGE       │ Which buildings / roads / bridges touch it?   │
   │ 6. CUT-OFF      │ Re-route every village to the nearest town or │
   │                 │ hospital with the broken roads removed        │
   │ 7. REPORT       │ Facts → situation report (EN + Nepali), Q&A,  │
   │                 │ dashboard, CSV/KML exports                    │
   └─────────────────┴──────────────────────────────────────────────┘
                     │
 You get: flood map · damaged assets · cut-off villages ranked by people · sitrep
```

The full Trishuli case study, about 52 × 58 km, runs **in about 9 minutes** on a laptop, starting from raw satellite data.

---

## 4. Every part explained

### 4.1 Finding the right satellite images (`backend/pipeline/scenes.py`)

**What it does:** it searches the Copernicus catalogue for Sentinel-1 radar images over the area. It picks the **first pass after the flood** and the **two most recent passes before it, on the same orbit track**.

**Why the same track matters:** a radar satellite looks at mountains from the side. On a different track it views the valley from another angle, so the images can't be compared pixel by pixel. On the same track the geometry is identical, so any difference is a real change on the ground.

**Trishuli:**
- Track 85 ascending: after 28 Aug, before 16 Aug and 4 Aug (12-day repeats).
- Track 19 descending: after 5 Sep, before 24 Aug and 12 Aug. It looks from the other side and fills in radar shadows.

### 4.2 Downloading the data (`backend/pipeline/fetch.py`)

| Data | Source | Why we need it |
|---|---|---|
| **Sentinel-1 radar** (VV and VH polarisation) | Sentinel Hub on the Copernicus Data Space (free account) | Works day and night and **sees through monsoon clouds** |
| **Sentinel-2 optical** | Same | Shows debris and stripped vegetation very clearly, *when the sky is clear* |
| **Copernicus DEM** (30 m elevation) | AWS open data | Terrain: slopes, rivers, how high each point sits above the river |
| **OpenStreetMap, pre-event** | Geofabrik snapshot from **1 Aug 2026** | Buildings, roads, bridges, villages, hospitals |

**Smart cloud handling (optical):** monsoon Nepal is cloudy almost every day. Instead of trusting the satellite's "cloud %" for a whole 100 km tile, we check every date *over our area only*, cheaply at low resolution. Then we keep the clearest dates and stitch them together pixel by pixel.

For Trishuli this found **12 Aug at 99.7% clear** (before) and **27 Aug** (after, the day after the flood).

**Everything is cached**, so re-running an area is instant and doesn't spend our free processing quota.

### 4.3 Terrain: where can a flood reach? (`backend/pipeline/terrain.py`)

From the elevation model we compute:

- **Slope:** how steep each spot is.
- **River network:** where water flows (D8 flow direction and flow accumulation), plus the mapped OSM rivers.
- **HAND (Height Above Nearest Drainage):** how many metres a spot sits above the river it drains into. A house 5 m above the river is in danger; one 300 m up the hillside is not.

We only look for flood evidence where a flood could physically reach. This removes most false alarms on steep slopes. The zone **adapts to the terrain automatically**:

| Terrain | Detected when | Allowed zone |
|---|---|---|
| Mountains (debris floods surge tens of metres high) | relief > 1,000 m | HAND < 60 m, slope < 35° |
| Lowlands (normal river floods) | otherwise | HAND < 15 m, slope < 10° |

### 4.4 Detecting the flood: three independent votes (`backend/pipeline/detect.py`)

Every pixel can get up to **three votes**:

1. **AI model vote:** the U-Net (see 4.5) says "water now, but not before".
2. **Radar change vote:** radar brightness changed a lot between before and after.
   - **Darker** means water or smooth wet mud.
   - **Brighter** means rough boulders and debris.
3. **Optical change vote:** in clear Sentinel-2 pixels, vegetation disappeared (NDVI drop) or the wet river channel moved (MNDWI change).

**The clever part: how much change counts as change?**

Radar images are noisy; even with no flood, two images differ by about ±2–4 dB. A fixed threshold either misses the flood or flags noise everywhere.

So we use **CFAR (constant false alarm rate)**. We compare the **two "before" images** with each other. Nothing happened between them, so their difference *is* the noise. The threshold is set where only 1% of that noise would trigger.

The threshold therefore adapts to every scene, and **no reference map is involved**. For Trishuli it came out at 1.6–2.1 dB.

**Combining votes:**
- A pixel with **at least one vote** is marked affected.
- Tiny specks are cleaned away (anything under 0.3 ha).
- The rest become map polygons labelled **water** or **debris**.
- Each polygon gets a **confidence** level: *high* when two or more independent sources agree.

### 4.5 The AI model: flood segmentation (`ml/train_sen1floods11.py`)

This is the **AI component we submit for judging**.

- **Model:** U-Net, a standard image-segmentation network, with a ResNet-34 backbone.
- **Input:** a Sentinel-1 radar image (VV + VH). **Output:** the probability of water for every pixel.
- **Training data:** **Sen1Floods11**, 446 hand-labelled flood images from 11 countries (the challenge's listed dataset).
- **Training:** 25 epochs on a laptop GTX 1650, about 7 minutes. We added "radiometric jitter" (random brightness shifts) so the model tolerates the calibration difference between training images and our Sentinel Hub images.
- **Runs on CPU in the app:** exported to ONNX, a portable model format.

**How good is it on scenes it never saw?**

| Test set | IoU (water) | F1 |
|---|---|---|
| Held-out test chips | 0.66 | 0.80 |
| **Bolivia (a whole flood event never seen in training)** | **0.66** | **0.79** |
| **Trishuli 2026 (unseen Himalayan scene), full system vs EMSR927** | **0.71** | **0.83** |

**Honest point (say it if asked):** in this event the model contributed fewer votes than radar and optical change. The flood was mostly **debris** (mud and boulders), not standing water, and the model was trained on water. That's why the system fuses three sources instead of trusting one.

### 4.6 Pre-event OpenStreetMap (`backend/pipeline/osm.py`, `osm_local.py`)

- We must use the map **as it was before the flood**. After a disaster, volunteers map the damage, and using that would be cheating.
- The challenge suggested the ohsome API, but its download endpoints were returning *403 Forbidden* (the new version needs a separate key). Public Overpass servers were also overloaded.
- **Our solution:** Geofabrik publishes **dated** snapshots of Nepal. `nepal-260801.osm.pbf` is OpenStreetMap exactly as it was on **1 Aug 2026**. We downloaded it once and built a fast local index: any area query takes about **4 seconds** with no dependence on public servers. That matters for the live judging.
- For the case study this gives **41,371 roads, 201,525 buildings, 727 villages and 263 towns/hospitals**.
- **Outside Nepal** (e.g. Chamoli in India, or Japan), the system uses the **ohsome API v2** from HeiGIT, the source the challenge recommends. It returns OpenStreetMap exactly as it was on the day before the flood, for any place and any date since 2007, and it needs a free API key (`OHSOME_API_KEY` in `.env`).
- **No key?** It falls back to public Overpass "attic" queries, which do the same job but are often overloaded.

### 4.7 What was damaged? (`backend/pipeline/impact.py`)

| Asset | Rule |
|---|---|
| **Building** | *in water* or *in debris* if it touches a flood polygon; *at risk* if within 30 m |
| **Road** | *blocked* if a road segment lies within 10 m of a flood polygon |
| **Bridge** | *likely damaged / washed out* if within 50 m of an affected channel |

Bridges are *inferred*: a bridge deck is about 5 m wide and can't be seen directly in 10–20 m satellite pixels.

### 4.8 Who is cut off? (`backend/pipeline/network.py`)

This is the most life-saving part.

1. Turn the pre-event road map into a **network graph**: junctions are points, road segments are links, and each link has a length.
2. **Targets** are hospitals and towns.
3. For every village, find the **shortest road route to the nearest target** (multi-source Dijkstra) on the **before** network.
4. Remove the blocked roads and damaged bridges, and do it again on the **after** network.
5. Each village gets a status:

| Status | Meaning |
|---|---|
| 🔴 **Cut off** | had a road route before, has **none** now |
| 🟠 **Long detour** | still connected, but the route is > 1.5× longer |
| 🟢 **Connected** | normal |
| ⚪ **No road before** | only reachable on foot even before the flood |

6. For every cut-off village we find the **first broken point** on its old route: how many km from the village, and whether it is a bridge.
7. **Estimated people** = buildings within 500 m × 4.4 (Nepal Census 2021 average household size). This is a proxy, not a census.
8. Villages are **ranked by estimated people**, giving rescuers a priority list. It is exported as **CSV** (for spreadsheets) and **KML** (opens in Google Earth or phone map apps for helicopter crews).

### 4.9 Situation report and copilot (`backend/pipeline/sitrep.py`)

**The problem with AI-written reports:** language models sometimes invent numbers, and in a disaster a made-up number could send a helicopter to the wrong place.

**Our solution: the AI never writes a number.**

1. The pipeline saves every number to `facts.json`, e.g. `settlements_cut_off = 37`.
2. The language model writes the report using **placeholders only**, e.g. "*{{settlements_cut_off}} villages are cut off*". It must also use placeholders for village names.
3. **Our code** fills in the real values. In Nepali it uses Devanagari digits (३७).
4. A **validator** rejects any answer containing a digit typed by the AI (in any script) or an unknown placeholder. The AI gets one retry.
5. For Nepali there is also a **script check**: no Cyrillic or other foreign letters, and no garbled words mixing English and Nepali letters.
6. If anything fails, or there's no internet or API key, the system uses a **hand-written report template** with the same numbers.

**Result:** the system cannot show a number that didn't come from its own maps. Six unit tests prove this (`backend/tests/test_core.py`).

- **The language model:** NVIDIA **Nemotron 3 Super** via OpenRouter (about $0.001–0.004 per report). It also works with Claude via the Anthropic API or OpenRouter.
- **What we learned:** Nemotron writes good English. Its Nepali mixed in Hindi words and garbled characters, so the script check rejects it and Nepali readers get the clean template. A weak model can lower the prose quality, but never the correctness.
- **Q&A:** the same safety rules apply. Example: *"Which villages should helicopters reach first?"* gives the top cut-off villages by population, says it has no medical-need data, and reminds the reader to confirm on the ground.

### 4.10 Bonus: flood-path tracer (in `terrain.py` + API `/trace`)

Click any point upstream, such as the glacier lake area. The system follows the **steepest downhill path** on the elevation model, along the river the flood would take. It lists the villages along it **in the order the water would reach them**, with:
- the distance downstream (km)
- the height above the river (m)
- the number of houses close to river level
- an *illustrative* arrival window at 3–8 m/s. This is not a forecast, and the app says so.

This is the beginning of an early-warning list for villages downstream.

### 4.11 The dashboard (`frontend/src/App.jsx`, `backend/app/main.py`)

**Backend:** a FastAPI web server that runs the pipeline, stores results and serves map layers. **Frontend:** a React app with MapLibre maps.

| Part of the screen | What it shows |
|---|---|
| **Left panel** | Draw an area or type its coordinates, pick a date, *Run analysis*; *Trishuli case study* button; live pipeline log; layer switches and legend; previous runs |
| **Map** | Flood (blue) and debris (brown) polygons; blocked roads (red); damaged bridges; villages coloured by status, sized by population; hospitals and towns; first break points; radar and optical before/after images; EMS agreement layer |
| **Overview tab** | Key numbers (cut off, buildings hit, hectares, km blocked, bridges); clickable priority table that zooms to each village; inputs used; **CSV/KML/GeoJSON downloads** |
| **Sitrep tab** | English / नेपाली toggle; *Template* or *AI-written*; *Print / save as PDF* |
| **Ask tab** | Questions in English or Nepali, answered from the map data only |
| **Flow path tab** | The bonus tracer |
| **Model tab** | AI model accuracy and the EMSR927 comparison |

### 4.12 Checking against the official maps (`backend/eval/evaluate_emsr.py`)

Copernicus EMS (activation **EMSR927**) mapped this flood by hand from sub-metre satellite images in four areas: Syapru Besi, Timure, Bidur and Phosretar. We download those maps into `validation/` and compare **only after** our system has finished. The pipeline code never reads that folder.

---

## 5. Results on the real flood

**Case study:** 84.93–85.45°E, 27.78–28.30°N (Rasuwagadhi to Phosretar), 26 Aug 2026, 20 m pixels, 526 s.

| What the system found | Number |
|---|---|
| Flood + debris area | **3,325 ha** (612 ha water, 2,712 ha debris); 1,109 ha confirmed by 2+ sources |
| Buildings hit | **4,020** (+2,087 within 30 m); ≈ 17,700 people live in them |
| Roads likely blocked | **152 km** (incl. 25 km primary road on the Pasang Lhamu highway) |
| Bridges likely damaged | **59 of 151** |
| Villages cut off | **37** (≈ **16,400 people**); 10 more on long detours |
| Villages with no road to any hospital | **38** |
| Top priority villages | Aamachhodingmo (≈1,487), Devighat (≈1,364), Betrawati (≈977), Bhainse (≈972), Adamghat (≈823) |

**Compared with the official EMSR927 maps:**

| Check | Result |
|---|---|
| Flood extent IoU / F1 | **0.71 / 0.83** |
| Precision / recall | 0.80 / 0.86 (with 2-pixel tolerance: 0.90 / 0.95) |
| EMS-damaged buildings inside our flood area | **4,503 of 4,685 (96%)** |
| Blocked roads matching EMS-damaged roads | 74.6 km of our 100 km inside the EMS areas (EMS total 91.8 km) |

**Simple meaning:**
- **Precision 0.80:** 80% of what we flagged, Copernicus also flagged.
- **Recall 0.86:** we found 86% of what Copernicus mapped.
- **IoU:** the overlap divided by the combined area (1.0 = perfect).
- **F1:** the balance of precision and recall.

### Our honesty story (judges like this)

Our **first** real run used a "lowland flood" rule that only looked within 25 m of river height. It was almost never wrong (precision 1.00) but found only **12%** of the flood.

Investigating, we learned that a **debris flood surges tens of metres up the valley walls**. We changed to the terrain-adaptive zone and recall jumped. Because that change was informed by the reference event, **we report both versions openly**. The radar thresholds were never tuned to the reference; they come from each scene's own noise.

---

## 6. How we follow the data rules

| Rule | How we comply |
|---|---|
| Only Sentinel-1/2, Copernicus DEM, pre-event OSM, listed training data as inputs | That is exactly what `fetch.py`, `osm.py` and `train_sen1floods11.py` use |
| No EMS/UNOSAT damage maps as input | EMS files live in `validation/`, read **only** by `backend/eval/` |
| OSM must be pre-event | Geofabrik snapshot from 1 Aug 2026 (or Overpass at event − 1 day) |
| Free Maxar/Planet imagery of this event exists | **Not used.** It is not on the allowed list |
| Required attributions | In the dashboard footer, every situation report, README and report |

---

## 7. Limitations: what the system cannot do

Being honest here is part of the score:

1. **No early warning.** Sentinel-1 passes every 6–12 days, so we map the *aftermath*. The first image here came 2 days after the flood. It cannot warn of a glacier collapse minutes ahead.
2. **Radar shadow.** In narrow gorges the radar can't see some slopes. We use two orbit directions and report how much was actually observed.
3. **Debris versus other change.** Rain, wet soil or shifting river bars can look like change. The three-vote system reduces this but doesn't eliminate it.
4. **Clouds.** Optical images help only where the sky is clear (31.7% of this area).
5. **Landslides on hillslopes** outside the flood zone also block roads and are not detected, so cut-off counts are a minimum.
6. **Bridges are inferred**, not seen.
7. **OSM gaps and errors.** Missing footpaths and buildings, mislabelled names (one village is tagged *Thulo* in Nepali but *Sano* in English), and health posts tagged as hospitals.
8. **Population is an estimate** (buildings × 4.4).
9. **A prototype**, not an operational tool. Everything needs ground confirmation.

---

## 8. Project folder map

```
backend/
  pipeline/            ← the whole analysis
    config.py          settings, thresholds, map grid
    scenes.py          find same-track before/after radar images
    fetch.py           download radar, optical (cloud screening), elevation
    terrain.py         slope, rivers, HAND, flow-path tracing
    model.py           runs the AI water model (ONNX)
    detect.py          three-vote fusion, CFAR thresholds, polygons
    osm.py / osm_local.py   pre-event OpenStreetMap (local Nepal index)
    impact.py          buildings and bridges hit
    network.py         road routing before vs after, cut-off villages
    sitrep.py          facts, EN/NE reports, copilot, validators
    run.py             runs everything in order + command-line tool
  app/main.py          web API for the dashboard
  eval/evaluate_emsr.py  comparison with EMSR927 (checking only)
  models/              trained AI model + its accuracy numbers
  tests/test_core.py   6 unit tests (no invented numbers, cut-off logic)
frontend/src/App.jsx   the dashboard
ml/                    model training script + Colab notebook + training log
report/                6-page report, evidence files, example sitreps
validation/            EMSR927 reference maps (not committed)
data/                  satellite cache, OSM index, run outputs (not committed)
```

---

## 9. How to run it

```bash
# terminal 1: the API (also serves the built dashboard)
cd backend
../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

Then open **http://localhost:8000**, click **Trishuli case study**, then **Run analysis**. Cached runs load instantly.

Command line (from `backend/`):

```bash
../.venv/Scripts/python.exe -m pipeline.run --bbox 84.93,27.78,85.45,28.30 --date 2026-08-26
```

Credentials go in `.env` (see `.env.example`): the Sentinel Hub ID and secret are required; OpenRouter or Anthropic keys are optional.

---

## 10. The 3-minute demo script

About 320 spoken words, roughly 2½ minutes at a calm pace, which leaves time for clicks. Record the dashboard full-screen with the case study already run (it loads instantly). **No images of victims**; show maps only.

| Time | Show on screen | Say |
|---|---|---|
| **0:00–0:20** | Title, then the dashboard map of the valley | "On 26 August 2026 a glacier collapse sent a debris flood down the Trishuli valley in Nepal. Roads, bridges and phones failed. Rescuers needed to know where the damage was and who was cut off. Baadhi Watch answers that from space, using only free satellite data." |
| **0:20–0:45** | Left panel: draw an area, pick the date, click *Run*; show the log | "You give it any area and a flood date. It finds Sentinel-1 radar images before and after, on the same orbit track, because radar sees through monsoon clouds. It adds Sentinel-2 optical images where the sky is clear, the Copernicus elevation model, and OpenStreetMap as it was before the flood." |
| **0:45–1:15** | Toggle *Radar change* and *Optical before/after*, then the flood layer | "Each pixel gets three independent votes: our AI water model, radar change, and optical change. The radar threshold calibrates itself on two pre-flood images, so it adapts to any valley. We only look where water can physically reach, using height above the river. Here: 3,300 hectares of flood and debris." |
| **1:15–1:45** | Zoom on red roads, broken bridges, red villages; click a village popup | "Then: what was damaged and who is cut off. We overlay pre-flood buildings, roads and bridges, remove the blocked roads, and re-route every village to the nearest town or hospital. 37 villages lost their road link, about 16,000 people. For each one we show where the road first breaks." |
| **1:45–2:10** | Overview tab: priority table, then CSV/KML download | "Rescuers get a priority list ranked by population, exportable to Google Earth for helicopter crews." |
| **2:10–2:30** | Sitrep tab: English, then नेपाली; Ask tab with one question | "One click gives a one-page situation report in English and Nepali. The AI copilot never types a number: it writes placeholders that our code fills from the map, and a validator rejects anything else. So it cannot invent figures." |
| **2:30–2:50** | Model tab, then the EMSR927 agreement layer (green/red/blue) | "Against the official Copernicus emergency maps we score an F1 of 0.83, and 96% of the buildings they marked damaged fall inside our flood area. Our AI model scores IoU 0.66 on a flood it never saw." |
| **2:50–3:00** | Limitations slide or report | "Limits: satellites pass every few days, so this maps the aftermath; it can't warn minutes ahead. Everything needs ground confirmation. Baadhi Watch: from space to the people who need help." |

**Tips:**
- Speak slowly; 3 minutes goes fast.
- Have the case study loaded beforehand so nothing has to compute on camera.
- If the AI report is slow, show the *Template* version. It has the same numbers.

---

## 11. Likely judge questions, with short answers

**"Why radar and not just optical?"** Monsoon clouds cover most of the valley most days; on the day after the flood only part of the area was clear. Radar sees through clouds, and optical adds detail where it's clear.

**"Why the same orbit track?"** Mountains look completely different from another viewing angle. Comparing images from the same track means differences are real ground changes.

**"How do you avoid false alarms?"**
1. Only look where water can reach (height above river, slope).
2. A self-calibrating threshold from two pre-event images (CFAR, about 1% false alarms).
3. Require clusters of at least 0.3 ha.
4. Show confidence by how many sources agree.

**"How do you know the AI doesn't make up numbers?"** It can't type numbers at all, only placeholders. Code inserts the values, a validator rejects any digit, and unit tests prove it.

**"Did you use the EMS maps to train or tune?"** No training or threshold tuning on EMS. One design choice was revised after checking: the flood zone height, because debris floods surge higher. We report both before and after openly.

**"Would it work somewhere else?"**
- The pipeline is generic: any area and date.
- The terrain regime switches automatically between mountain and lowland.
- Radar thresholds self-calibrate.
- OSM comes from local Nepal snapshots or, elsewhere, Overpass historical queries.

**"Why does the AI model contribute less than the other votes here?"** It was trained on water floods, and this was mainly a debris flood. That's why we fuse three evidence sources instead of relying on one model. Fine-tuning on debris data (e.g. Kuro Siwo) is future work.

**"How long does it take?"** About 9 minutes for a 52 × 58 km area from raw data on a laptop; re-runs are instant from the cache.

**"What would you do next?"**
- Radar *coherence* (phase data), which sees debris better.
- Training on Kuro Siwo.
- Footpath networks, because many Himalayan villages are only reachable on foot.
- Automatic runs whenever a new satellite pass arrives.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **SAR / radar (Sentinel-1)** | A satellite that sends microwaves and measures the echo. Works through clouds and at night |
| **VV / VH** | Two radar polarisations. VH is sensitive to vegetation, VV to surface roughness |
| **σ⁰ (sigma-nought), dB** | Radar brightness, on a logarithmic scale |
| **Sentinel-2** | Optical satellite (like a camera), useless under clouds |
| **NDVI / MNDWI** | Indices from optical bands: vegetation greenness / presence of water |
| **DEM** | Digital Elevation Model, the height of the ground everywhere |
| **HAND** | Height Above Nearest Drainage, metres above the river |
| **CFAR** | Constant False Alarm Rate, a threshold set from the noise level |
| **U-Net** | A neural network that labels every pixel of an image |
| **IoU / F1 / precision / recall** | Accuracy measures: overlap; balance; "how often right when it says flood"; "how much of the flood it found" |
| **OSM** | OpenStreetMap, the free, volunteer-made world map |
| **EMS / EMSR927** | Copernicus Emergency Management Service and its activation for this flood |
| **GLOF** | Glacial Lake Outburst Flood |
| **Dijkstra** | The classic shortest-path algorithm used for routing |
| **ONNX** | A portable format to run a trained AI model anywhere |
