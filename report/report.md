# Baadhi Watch: mapping flood damage and cut-off settlements from space

**Track B: Mapping Flood Damage from Space, Multimodal AI Hackathon 2026**
Case study: Bhote Koshi–Trishuli debris flood, Nepal, 26 August 2026 (Copernicus EMS activation EMSR927)

## 1. Problem and what the system does

On 26 August 2026 a high-altitude glacier collapse in northern Nepal sent a flood of water, mud and debris down the Bhote Koshi–Trishuli corridor. In such events roads, bridges and phone lines fail together, so rescuers cannot see where damage is worst or which villages are isolated.

Baadhi Watch takes **any bounding box and flood date**. Starting from raw, free satellite data, it produces:

1. **A flood and debris map** from Sentinel-1 radar change detection (same orbit track), a U-Net water model, and Sentinel-2 optical change where clear, constrained to the valley floor.
2. **A damage overlay**: pre-event OpenStreetMap buildings, roads and bridges classed as hit, at risk or unaffected.
3. **A cut-off analysis**: road connectivity before vs after from every settlement to the nearest town or hospital, ranked by estimated population, with the first broken point on each route.
4. **A map dashboard and a one-page situation report** in English and Nepali, a grounded Q&A copilot, a downstream flow-path tracer (bonus), and CSV/KML exports for field teams.

## 2. Data and compliance with the data rules

| Input | Source | Use |
|---|---|---|
| Sentinel-1 IW GRD (VV, VH) | CDSE catalogue + Sentinel Hub Process API | before/after radar, sigma0 orthorectified with Copernicus DEM 30 m, Lee 5×5 |
| Sentinel-2 L2A | Sentinel Hub | before/after optical (SCL cloud mask), only where clear |
| Copernicus DEM GLO-30 | AWS Open Data | slope, D8 flow, HAND, flow-path tracing |
| OpenStreetMap, pre-event | Geofabrik dated extract `nepal-260801` (OSM on 1 Aug 2026) or Overpass attic query at event − 1 day | buildings, roads, bridges, settlements, hospitals, rivers |
| Sen1Floods11 (hand-labelled) | public bucket | model training |

- **Not used as inputs:** Copernicus EMS, UNOSAT or other damage maps, post-event OSM edits, and commercial imagery. EMSR927 vectors are read only by a separate evaluation module (`backend/eval/`).
- **ohsome substitute.** The ohsome v1 extraction endpoints returned HTTP 403 during the build, and v2 requires an API key. The same pre-event snapshot is therefore taken from Geofabrik's dated extract, which is equivalent and keeps live runs independent of public servers.

## 3. Method

**Scene selection.** For the given date, the CDSE catalogue is searched for Sentinel-1 GRD passes. Each post-event pass is paired with the one or two most recent pre-event passes **on the same relative orbit**, so the viewing geometry is identical and pixel-wise comparison is valid. Up to two tracks are fused, preferring opposite look directions so that one fills the other's radar shadow.

For Trishuli, the pairs are track 85 ascending (28 Aug vs 16 and 4 Aug) and track 121 descending (31 Aug vs 19 and 7 Aug).

**Flood-reachable zone.** HAND (height above nearest drainage) is computed from the Copernicus DEM. Drainage is defined as flow accumulation > 5 km² plus pre-event OSM rivers. The zone depends on relief:

- **mountain regime** (DEM relief > 1 km): HAND < 60 m and slope < 35°, because debris floods surge far above normal flood level;
- **lowland regime**: HAND < 15 m and slope < 10°.

In both regimes, pixels with a local incidence angle outside 15°–75° are excluded, which removes layover and radar shadow.

**Evidence and fusion.** Each pixel can receive up to three independent votes:

1. *Model.* The U-Net predicts water after the event but not before.
2. *SAR change (CFAR).* The 7×7-averaged sigma0 change against the pre-event median exceeds a scene-adaptive threshold. The threshold is the 99th percentile of the change between the **two pre-event images** in the same zone: how much the ground "changes" when nothing happened. It needs no reference data and transfers to any area and date. Darkening indicates water or wet mud; brightening indicates boulders and debris.
3. *Optical change.* NDVI loss > 0.25 (vegetation scoured or buried) or |ΔMNDWI| > 0.3 (the wet channel moved), on pixels clear in both composites. Every Sentinel-2 date in the window is screened for cloud over the area using its SCL layer at 100 m, which costs almost nothing. The three clearest dates are composited per pixel, closest to the event first.

Pixels with at least one vote are cleaned morphologically (minimum 0.3 ha) and vectorised. Each polygon is classed as **water** or **debris/changed ground** and carries a confidence level: high when ≥25% of its area has two or more agreeing sources. No detection threshold was fitted to EMSR927. The one post-check design change, the zone, is described in §5.

**Damage.** A building is marked *in water* or *in debris* if it intersects an affected polygon, and *at risk* if it lies within 30 m of one. A road segment is *blocked* if it lies within 10 m of an affected polygon. A road bridge within 50 m of an affected channel is *likely damaged / washed out*; its deck is too small to observe directly at 10–20 m.

**Cut-off analysis.**

1. A graph is built from pre-event OSM roads, including tracks, with OSM vertices as nodes so that junction topology is exact. The road network is fetched 15 km beyond the analysis area so that routes out of the valley exist.
2. Targets are hospitals plus towns and cities.
3. A multi-source Dijkstra runs on the pre-event graph and on the post-event graph (blocked edges and damaged bridges removed).
4. Each named settlement is classed as:
   - **cut off**: reachable before, unreachable after
   - **long detour**: route > 1.5× and > 1 km longer
   - **connected**
   - **no road before the event**
5. For cut-off settlements the first blocked edge on the old route is reported, with whether it is a bridge.
6. Population is estimated as buildings within 500 m × 4.4 (Nepal Census 2021 average household size).

**Bonus: flow-path tracing.** From any clicked point the steepest-descent D8 path on the DEM is followed downstream. Settlements within 1 km of the channel and ≤ 30 m above it are listed in arrival order, with distance and an *illustrative* arrival window at 3–8 m/s front speed.

## 4. AI components

### 4.1 Flood segmentation model (submitted AI component)

- **Model.** U-Net with a ResNet-34 ImageNet encoder. Input is 2-channel Sentinel-1 VV/VH sigma0 in dB.
- **Training.** Sen1Floods11 hand-labelled chips with 256 px random crops, flips and rotations, and per-channel radiometric jitter. Jitter makes the model robust to calibration differences between the training data (GEE-processed) and our Sentinel Hub inputs. Loss is class-weighted cross-entropy plus Dice. 25 epochs on a GTX 1650 took about 7 minutes.
- **Inference.** Exported to ONNX and run on CPU with overlapping tiles.

| Evaluation split (never trained on) | IoU (water) | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation chips | 0.647 | 0.785 | 0.746 | 0.829 |
| Test chips | 0.663 | 0.797 | 0.778 | 0.818 |
| Bolivia (held-out *event*) | 0.657 | 0.793 | 0.772 | 0.816 |
| **Trishuli 2026 (unseen Himalayan scene), fused system vs EMSR927** | **0.71** | **0.83** | 0.80 | 0.86 |

### 4.2 Grounded situation-report copilot (supporting feature)

- All numbers live in `facts.json`, produced by the pipeline. The same facts drive the Q&A copilot ("Which villages should helicopters reach first?").
- The language model may reference numbers *and settlement names* only as `{{key}}` placeholders. It is reached through the Anthropic API or OpenRouter; our demo uses NVIDIA Nemotron 3 Super on OpenRouter, about $0.001–0.004 per report. Code substitutes the values, using Devanagari digits for quantities and dates in Nepali.
- Nepali output must also pass a script check: no foreign scripts, and no words mixing Latin and Devanagari. In testing, Nemotron's Nepali mixed in Hindi vocabulary and Cyrillic fragments, so it failed this check. Rescuers then receive the hand-written Nepali template, so a weak model can lower prose quality but never correctness.
- Any output containing a raw digit (ASCII or Devanagari) or an unknown key is rejected and retried, then replaced by a deterministic template. This is unit-tested.
- The system therefore **cannot** show a number that did not come from its maps, and it still works offline without an API key.

## 5. Case study: Trishuli, 26 August 2026, compared with EMSR927

**Run.** The bbox is 84.93–85.45°E, 27.78–28.30°N, from Rasuwagadhi to Phosretar. The grid is 20 m, 2,600 × 2,920 px. The run took 526 s from raw data on a laptop.

**Inputs.**
- *Radar:* Sentinel-1D track 85 ascending, 28 Aug against 16 and 4 Aug. A second track, 19 descending, compares 5 Sep against 24 and 12 Aug.
- *Optical:* Sentinel-2 composites. Before: 12 Aug, 4 Aug, 23 Jul. After: 27 Aug. 31.7% of the area was clear in both.
- *Terrain:* the regime was detected automatically as mountain (relief 3,652 m).
- *Radar change thresholds:* calibrated per track from each scene's own no-event pair. They came out at 1.8/2.1 dB (VV/VH) and 1.6/1.7 dB.

**What the system reports.**
- **Flood and debris:** 3,325 ha affected (612 ha water, 2,712 ha debris or changed ground). 1,109 ha are confirmed by two or more independent sources.
- **Buildings:** 4,020 hit (407 in water, 3,613 in debris), plus 2,087 within 30 m.
- **Roads and bridges:** 152 km likely blocked across the whole area (100 km of it inside the EMS areas), including 25 km of primary and 4 km of trunk road on the Pasang Lhamu highway corridor. 59 of 151 road bridges are likely damaged.
- **Settlements:**
  - **37 cut off** (≈ 16,400 people), including Aamachhodingmo, Devighat, Betrawati, Bhainse and Adamghat.
  - 10 more face long detours.
  - 38 lost road access to every hospital.

**Against EMSR927** (all four EMS areas: Syapru Besi, Timure, Bidur, Phosretar). EMS mapped the whole debris-flood footprint as "mass movement".

| Metric | Value |
|---|---|
| Extent IoU / F1 | **0.71 / 0.83** |
| Extent precision / recall | 0.80 / 0.86 |
| … with 2-pixel tolerance (EMS uses sub-metre imagery, we use 20 m pixels) | 0.90 / 0.95 |
| EMS "damaged / destroyed / possibly damaged" buildings inside our affected area | **4,503 of 4,685 (recall 0.96)** |
| Our blocked roads (inside EMS areas) coinciding with EMS-affected roads | 74.6 km of our 100.0 km; EMS total 91.8 km |

**Honest account of iteration.** The first real run (v0.2) used a lowland-style valley mask: HAND < 25 m, slope < 20°. On the upper reach it reached **precision 1.00 but recall 0.12** (IoU 0.12). Inspection showed that this mask covered only 29% of the debris footprint, because a debris flood surges tens of metres above normal flood level.

We replaced it with a terrain-adaptive zone, selected automatically by relief:
- mountains: HAND < 60 m, slope < 35°
- lowlands: HAND < 15 m, slope < 10°

On the upper reach this raised IoU to 0.47 and building recall to 0.88. The change was informed by this one reference event, so we report both versions. No detection threshold was fitted to EMS: radar thresholds come from each scene's pre-event noise, and the optical thresholds are fixed.

**Where we disagree with EMS, and why.**
- **Our extra area** (red on the agreement map) is mostly bank vegetation and slopes stripped by the flood, which EMS often did not delineate, plus monsoon-season change and some landslides on lower slopes.
- **Our misses** (blue) are mostly debris inside narrow gorge sections where 20 m pixels mix rock walls with the channel, and areas cloudy on 27 Aug that rely on radar alone. Radar intensity barely changes when a boulder bed is buried by more boulders.
- **The U-Net contributed few votes** (8k pixels against 45k for radar change and 57k for optical change). This is a debris event, not standing water, so the fused system, not the water model alone, carries the result.
- **Upper reach (Syapru Besi–Timure, 10 m):** IoU 0.47, precision 0.98, building recall 0.88, roads 11.3 of 12.7 km matching. The narrow gorge is the hardest terrain for both sensors.

Evidence files: `report/evidence/*.json`; agreement map `data/runs/3014ddb4f134/emsr927_agreement.png`.

## 6. Limitations: what the system cannot do

- **No early warning.** Sentinel-1 revisits a valley every 6–12 days. The system maps the aftermath and cannot warn of a glacier collapse minutes ahead; the first post-event image here came two days after the flood.
- **Radar geometry.** In narrow gorges large parts of the valley floor are in radar shadow or layover for a given track. We report the observed fraction, and anything unobserved is unknown, not undamaged.
- **Debris versus other change.** Wet soil, monsoon rain, harvests and river-bar shifts between acquisitions also change backscatter. The valley mask and multi-source voting reduce false alarms but do not remove them. Debris on already bare gravel bars is hard to see.
- **Domain shift.** The water model learned from mostly lowland floods. Turbid, fast, debris-laden Himalayan rivers look different, so the model is one vote among three, not the sole decision.
- **Landslides.** Landslides on hillslopes, outside the HAND mask, block many Nepali roads and are not detected, so cut-off counts are a lower bound.
- **Bridges and roads are inferred.** Damage is inferred from proximity to affected ground; a bridge can survive a flood that reaches it, or fail from scour we cannot see.
- **OSM completeness and errors.** Missing trails, footbridges and new buildings bias damage and population counts. Tagging errors pass straight through: one node named ठुलो भार्खु (Thulo Bharkhu) carries `name:en=Sano Bharkhu`, and many health posts are tagged `amenity=hospital`. Many villages are reached only by footpaths, which are excluded, and a village with "no road before the event" is not necessarily isolated. Roads outside the fetched network are assumed open.
- **Population is a proxy.** Buildings × 4.4 is not a census and ignores seasonal and tourist populations.
- **Optical data.** Monsoon cloud usually removes Sentinel-2. Thresholds are untuned, and accuracy will vary by region.
- **Prototype, not an operational tool.** All outputs need ground confirmation.

## 7. Attribution and citations

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL); extracts by Geofabrik GmbH.
- Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* CVPR Workshops.
- Bountos, N. I., et al. (2024). *Kuro Siwo: 33 billion m² under the water.* NeurIPS Datasets and Benchmarks (recommended dataset; future work).
- Reference maps for checking: European Union, Copernicus Emergency Management Service data (EMSR927).
