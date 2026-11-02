# Data

Raw data is **not** committed (OSSDD shards ~0.5 GB each, an xView3 scene ≈ 3 GB). This file explains how to get everything.
Downloads land in `data/raw/`, `data/interim/` and `data/processed/`, which are git-ignored.

## 0. Credentials (`.env` in the repo root, git-ignored)

```
CDSE_CLIENT_ID=...        # Copernicus Data Space → Sentinel Hub dashboard → User settings → OAuth clients
CDSE_CLIENT_SECRET=...
GFW_API_TOKEN=...         # https://globalfishingwatch.org/our-apis/tokens (non-commercial use)
```

## 1. OSSDD: training / validation / test (used for all model results)

Open SAR Ship Detection Dataset, Hugging Face `sylviaHoch/OpenSARShip-Ship-Detection-Dataset` (CC BY-NC-SA 4.0, no login).
Used in this repo: train shards 000000–000004, val 000000, test 000000 (~3.5 GB).

```powershell
$B = "https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset/resolve/main/webdataset"
mkdir data/raw/ossdd/webdataset/train, data/raw/ossdd/webdataset/val, data/raw/ossdd/webdataset/test -Force
0..4 | % { curl.exe -L -o ("data/raw/ossdd/webdataset/train/train-{0:D6}.tar" -f $_) ("$B/train/train-{0:D6}.tar" -f $_) }
curl.exe -L -o data/raw/ossdd/webdataset/val/val-000000.tar  "$B/val/val-000000.tar"
curl.exe -L -o data/raw/ossdd/webdataset/test/test-000000.tar "$B/test/test-000000.tar"
```
Full dataset (14.4 GB, all splits): https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset/resolve/main/ossdd.zip

Then:
```
python -m src.extract_ossdd                                          # browsable folders + PNG previews
python -m src.ossdd_to_yolo --shards data/raw/ossdd/webdataset --out data/processed/ossdd
```
Output: `data/processed/ossdd/{images,labels}/{train,val,test}`, `{val,test}_arrays/*.npz` (VV/VH dB for CFAR), `{val,test}_gt.csv`, `{val,test}_chips.csv` (land fraction), `data.yaml`, `stats.json`.

## 2. xView3-SAR labels (per-ship AIS vs manual labels; dark-flag scoring is the next step)

1. Register at https://iuu.xview.us.
2. Download `validation.csv`, `public.csv`, `ESA_xView3_sceneName_mapping.csv` into `data/raw/xview3/labels/`.
   SHA1: validation `ad89b776…5cd1`, public `8ed2431d…e16f`, mapping `09357dfd…848d`.
3. Scenes (≈ 3 GB each; validation split 123 GB) go to `data/raw/xview3/scenes/<scene_id>/` and are chipped one at a time:
   `python -m src.make_chips --labels data/raw/xview3/labels/validation.csv --scenes data/raw/xview3/scenes --out data/processed/chips --delete-raw`

Why validation and not train: `DECISIONS.md` D01.

## 3. Sentinel-1 over the Gulf (Copernicus Data Space)

Search (no account needed):
```
python -m src.gulf_fetch search --aoi aoi/khalifa_port.geojson --start 2026-10-01 --end 2026-10-22
```
Fetch calibrated σ⁰ for an AOI (needs `CDSE_CLIENT_ID/SECRET`):
```
python -m src.gulf_fetch fetch --aoi aoi/khalifa_port.geojson --date 2026-10-15 --out data/raw/gulf
```
Output: `data/raw/gulf/<aoi>_<date>.tif`: 2 bands (VV dB, VH dB), EPSG:32640 (UTM 40N), 10 m, ≤ 2500 × 2500 px, no-data = −9999.
AOIs: `aoi/khalifa_port.geojson`, `aoi/hormuz_larak_anchorage.geojson` (main Hormuz AOI), `aoi/strait_of_hormuz.geojson` (open strait, low traffic).

## 4. Land masks (OpenStreetMap)
```
python -m src.land_mask --aoi aoi/khalifa_port.geojson --out data/interim/khalifa_port_land.geojson
```

## 5. AIS context for the Gulf (Global Fishing Watch)
Needs `GFW_API_TOKEN`; ~4-day (AIS) / ~5-day (SAR) latency, so use scenes at least 5 days old.
```
python -m src.gfw_ais --aoi aoi/khalifa_port.geojson --date 2026-10-15 --hour 2 --out data/raw/gfw
```
Output: `<aoi>_<date>_ais_presence.csv` (hourly AIS per vessel per 0.01° cell), `_sar_matched.csv`, `_sar_unmatched.csv`.

## Folder layout
```
data/
├── raw/
│   ├── ossdd/webdataset/{train,val,test}/*.tar  (+ extracted/)
│   ├── xview3/labels/*.csv
│   ├── gulf/*.tif
│   └── gfw/*.csv
├── interim/        land masks (GeoJSON)
└── processed/
    └── ossdd/      YOLO chips, arrays, ground truth
```
