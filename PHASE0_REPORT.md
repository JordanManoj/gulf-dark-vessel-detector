# Phase 0: Data-Access Report

**Date:** 22 Oct 2026 · **Status:** research done, building can start; four account sign-ups needed (see bottom).

## 1. xView3-SAR (training + labelled evaluation)

| Item | Finding | Source |
|---|---|---|
| Access | Free, **registration required** at iuu.xview.us ("Register/Login to Download Data") | iuu.xview.us |
| Size | 991 Sentinel-1 IW GRD scenes, avg. 29,400 × 24,400 px, ~1,400 gigapixels total | NeurIPS 2022 paper |
| Splits | train 554 · validation 50 · public 150 · holdout 237 (holdout **not released**) | paper §3 |
| Regions | Adriatic, Bay of Biscay, Gulf of Guinea, Iceland, Norway (North Sea). **No Persian Gulf.** | whitepaper Table 1 |
| Files per scene | `VV_dB.tif`, `VH_dB.tif` (10 m, UTM, half-precision ≈1.2 GB each), `bathymetry.tif`, `owiWindSpeed.tif`, `owiWindDirection.tif`, `owiWindQuality.tif`, `owiMask.tif` | reference dataloader, paper |
| Processing | SNAP: orbit correction, noise removal, radiometric calibration, terrain correction, dB | paper §3 step 2 |
| Labels (CSV) | `scene_id, detect_lat, detect_lon, detect_scene_row, detect_scene_column, is_vessel, is_fishing, vessel_length_m, source, confidence, distance_from_shore_km` | whitepaper, official `metric.py` |
| AIS link | `source` ∈ {AIS, AIS/manual, manual}. **Train split = AIS labels only**; other splits mix AIS + manual | paper |
| Metric | Hungarian matching, 200 m tolerance; aggregate = F1_det × (1 + length_acc + F1_vessel + F1_fishing + F1_shore) / 5; LOW-confidence labels not scored | official `metric.py` |
| Leaderboard (holdout) | 1st 0.618 aggregate (F1 detection 0.770, F1 near-shore 0.531); reference Faster R-CNN 0.190 | paper Table 1 |

**Plan changes from this:**
- **Train on the validation split, evaluate on the public split** (decision D01). The train split has no manual labels, so the vessels with no AIS, the very ones we care about, are missing from it.
- "Dark" in xView3 = vessel with `source == manual` (D02).
- Stream scenes and keep only chips: one scene is about 3 GB, the laptop has about 92 GB free (D05).
- **Domain shift is guaranteed:** no Gulf scenes in training at all. The Gulf run is genuinely out-of-distribution.

## 2. Sentinel-1 over the Gulf (Copernicus Data Space Ecosystem)

**Verified live on 22 Oct 2026** with an anonymous OData catalogue query, `productType = IW_GRDH_1S`, 1 Jul to 21 Oct 2026:

| AOI | Scenes found | Example | Size |
|---|---|---|---|
| Khalifa Port (`aoi/khalifa_port.geojson`) | **38** | `S1D_IW_GRDH_1SDV_20261021T021455_…` | ~1.7 GB, online |
| Strait of Hormuz (`aoi/strait_of_hormuz.geojson`) | **102** | `S1D_IW_GRDH_1SDV_20261021T021430_…` | ~1.7 GB, online |

- All are **1SDV = dual-pol VV+VH**, the same polarisations as xView3. Sentinel-1C and 1D are both acquiring, so revisits are frequent (about every 3 days).
- Search is anonymous; **download/processing needs a free CDSE account**.
- **Plan change:** instead of downloading 1.7 GB SAFE files and installing SNAP, use the **Sentinel Hub Process API** on CDSE. It returns thermal-noise-removed, calibrated σ⁰, orthorectified at 10 m, cropped to the AOI. That's the same quantity and pixel size as xView3 (D04). Free quota: 10,000 processing units/month.

## 3. AIS for the Gulf

| Source | Status | Notes |
|---|---|---|
| **Global Fishing Watch API** | ✅ usable | Free **for non-commercial use**, token required. `public-global-sar-presence` = GFW's own S1 detections with a `matched` (to AIS) filter, 2017 → 5 days ago. `public-global-presence` = AIS vessel presence (1 position/vessel/hour). **Gridded only**: no per-ship tracks via the API. |
| AISHub | ❌ | Requires contributing your own AIS receiver feed |
| MarineTraffic / Spire / exactEarth | ❌ | Commercial |
| NOAA / Danish Maritime Authority | ❌ | Free but cover US / Danish waters only |

**Plan change (matches the spec's fallback):** full, per-ship dark-vessel scoring happens on **xView3**. On the Gulf: detection + **cell-level agreement** with GFW (our detection counts vs GFW SAR detections and AIS presence per grid cell per day), plus a "likely dark" flag where we detect vessels in cells with no AIS presence (D13).

## 4. Compute

- Laptop: GTX 1650 (4 GB), but the installed PyTorch is the **CPU build** → use the laptop for preprocessing, CFAR, evaluation and the demo.
- **Training on Kaggle** (free T4/P100, 30 h/week): upload the chip dataset (a few GB) as a Kaggle dataset.

## 5. What the user (Jordi) needs to do

1. **Register at https://iuu.xview.us** and download access for the validation + public splits (labels + scenes).
2. **Create a CDSE account** at https://dataspace.copernicus.eu, then in the Sentinel Hub dashboard create an **OAuth client** (client id + secret) and put them in `.env` (see `data/README.md`).
3. **Get a GFW API token** at https://globalfishingwatch.org/our-apis/tokens (non-commercial/research use).
4. **Kaggle account** with phone verification (needed for GPU).
