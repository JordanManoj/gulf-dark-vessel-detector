# Decision Log

Every meaningful choice in this project: what was decided, what else was considered, why, and what happened.
Sources for the facts cited here are listed in [`docs/RESEARCH.md`](docs/RESEARCH.md).
"Result" stays `pending` until a run in this repo produces a number; no result is ever filled in from memory.

---

### D01 · 2026-10-22 · Training data: xView3-SAR, trained on the **validation** split, not the train split
- **Decision:** Fine-tune on scenes from xView3's *validation* partition (50 scenes); evaluate on the *public* partition (150 scenes, labels released). The 554-scene *train* partition is not used for training.
- **Alternatives:** (a) xView3 train split (largest); (b) SSDD / HRSID / OpenSARShip chip datasets; (c) SARFish (xView3 + SLC).
- **Reason:** The xView3 paper states the train split contains **only automated GFW (AIS-derived) labels**. Vessels without AIS are, by construction, the ones that are missing there, so a model trained on it learns to ignore exactly the "dark" vessels we want. The validation/public splits combine AIS and **manual** labels. The 1st-place solution also dropped the train split for the same reason (noisier). SSDD/HRSID are small pre-cut chips with no AIS link, so no dark-vessel evaluation is possible.
- **Result:** pending.

### D02 · 2026-10-22 · Definition of "dark" in xView3
- **Decision:** A ground-truth object is **dark** if `is_vessel == True` and `source == "manual"` (seen by an analyst, not correlated with any AIS track). AIS-correlated = `source in {"ais", "ais/manual"}`.
- **Alternatives:** Use GFW's per-detection `matched` flag (not available in xView3 CSVs); define by confidence level.
- **Reason:** The `source` column is the only AIS link in the labels (xView3 whitepaper: "source of detection (AIS, manual label, or both)").
- **Caveat (to state in interview):** "manual only" also includes AIS-carrying vessels that the AIS-to-SAR matcher failed to correlate, so this is an upper bound on true dark vessels.
- **Result:** pending.

### D03 · 2026-10-22 · Dark-vessel flag at inference = "detected but no AIS within 200 m"
- **Decision:** After detection, match each detection to AIS positions with Hungarian assignment and a 200 m gate; unmatched detections are flagged dark (drawn red).
- **Alternatives:** Learn "dark" directly as a class.
- **Reason:** "Dark" is not a visual property of the ship; it is a property of the *AIS record*. Learning it from pixels would teach the model geography/ship type shortcuts. Keeping detection and AIS matching separate mirrors how operational systems (GFW, Stella-style fusion) work and lets us evaluate each part on its own.
- **Result:** pending.

### D04 · 2026-10-22 · Gulf imagery via CDSE Sentinel Hub Process API (σ⁰, orthorectified, 10 m)
- **Decision:** Fetch VV/VH for each AOI from the Copernicus Data Space Sentinel Hub Process API with `backCoeff=SIGMA0_ELLIPSOID`, `orthorectify=true`, 10 m pixels, then convert to dB.
- **Alternatives:** (a) download the full 1.7 GB `.SAFE` GRD via OData and process with ESA SNAP (exact xView3 pipeline); (b) Google Earth Engine `COPERNICUS/S1_GRD`.
- **Reason:** xView3 imagery was produced with the Sentinel-1 Toolbox: noise removal, radiometric calibration, terrain correction, dB, 10 m. Sentinel Hub applies thermal-noise removal + calibration + orthorectification server-side, so the result is the same physical quantity at the same pixel size, for a ~25 km crop instead of a 1.7 GB scene. Free tier: 10,000 PU/month, far above what ~20 AOI requests need. GEE would also work but adds a second account and isn't part of the Copernicus ecosystem the spec names.
- **Risk:** small radiometric differences vs SNAP (different DEM, resampling). That is part of the domain-shift discussion, not hidden.
- **Result:** pending.

### D05 · 2026-10-22 · Disk strategy: stream xView3 scenes, keep only chips
- **Decision:** Download one scene at a time, cut training chips, delete the raw GeoTIFFs.
- **Reason:** One scene ≈ 2 × 1.2 GB (VV+VH, half precision) plus ancillary rasters; the laptop has ~92 GB free. Keeping 50 scenes is impossible, chips are a few % of that.
- **Result:** pending.

### D06 · 2026-10-22 · Input representation: 3 channels = VV dB, VH dB, VV−VH dB, fixed global scaling
- **Decision:** Clip VV and VH to [−50, 20] dB and the ratio to a fixed range, linearly map to [0, 255], stack as a 3-channel image.
- **Alternatives:** (a) per-chip min-max (xView3 reference repo); (b) sigmoid mapping (1st place); (c) add bathymetry/wind channels.
- **Reason:** Pre-trained YOLO weights expect 3 channels, so the third channel should carry information rather than a duplicate. VV−VH (cross-pol ratio in dB) is a standard, cheap ship/sea discriminator. **Fixed** scaling keeps absolute backscatter meaningful across chips and across regions, which per-chip min-max destroys (pointed out explicitly in the 4th-place xView3 write-up, which uses the same [−50, 20] clip). The 1st place found bathymetry/wind did not help, and they don't exist the same way for Gulf scenes.
- **Result:** pending.

### D07 · 2026-10-22 · Tiling at native 10 m resolution: 640 px tiles, 128 px overlap at inference
- **Decision:** Train on 640×640 chips cut at native resolution (no resizing); infer with a 640 sliding window and 128 px overlap, merging duplicates in scene coordinates.
- **Alternatives:** Resize large tiles (e.g. 2048 → 640); no overlap.
- **Reason:** Many vessels are only 2–10 px at 10 m spacing. Downscaling would shrink them below one pixel at YOLO's finest stride (8). Overlap ≥ the largest object (~40 px for a 400 m ship) guarantees every ship is fully inside at least one tile; the 1st place used the same idea (overlapping tiles + accumulation) to remove border artefacts.
- **Result:** pending.

### D07b · 2026-10-22 · Tile merging by "core + halo" instead of cross-tile NMS
- **Decision:** Each inference window = a 512 px **core** plus a 64 px **halo** on every side (640 px total, i.e. 128 px overlap). A detection is kept only if its centre falls in the window's core. The same scheme runs the CFAR baseline over full xView3 scenes in memory-sized pieces.
- **Alternatives:** Overlapping tiles + NMS / distance-based de-duplication across tiles (e.g. SAHI).
- **Reason:** Cores tile the scene without gaps or overlap, so every point belongs to exactly one core: no duplicates by construction and no merge threshold to tune. Any ship ≤ 64 px half-size centred in a core is fully visible in that window. A 128 px margin also covers the CFAR background window (81 px).
- **Result:** pending.

### D08 · 2026-10-22 · Point labels → fixed 10×10 px boxes
- **Decision:** xView3 gives points; convert each to a 10×10 px (100 m) box centred on the point.
- **Alternatives:** Box size from `vessel_length_m`.
- **Reason:** Length is only known for AIS-matched vessels; manual (dark) labels have none. Length-based boxes would give dark and AIS vessels systematically different box shapes, i.e. a label leak and an inconsistent regression target. Evaluation is point-based (centre distance ≤ 200 m) anyway, as in the official xView3 metric. Same approach as the official xView3 reference implementation.
- **Result:** pending.

### D09 · 2026-10-22 · Baseline: two-parameter CFAR on VH, with land mask
- **Decision:** Two-parameter (mean/std) CFAR with target/guard/background windows, computed on VH dB with box filters (integral images), followed by connected-component grouping.
- **Alternatives:** (a) global threshold; (b) CA-CFAR on linear intensity; (c) OS-CFAR; (d) K-distribution CFAR.
- **Reason:** It is the classic operational detector (ESA SNAP's Adaptive Thresholding operator, and the automated detector used to build xView3 itself), so it is the fair "what would you do without ML" baseline. VH is used because the xView3 whitepaper notes it separates ships from sea clutter better than VV. Box-filter implementation runs on a full 25 km AOI in seconds; OS-CFAR (sorting every window) is far more robust in dense harbours but much slower, so it's documented as a known limitation rather than implemented first. Window sizes follow the SNAP rule: target ≈ smallest object, guard ≈ largest object, background large enough for stable statistics.
- **Result:** pending.

### D10 · 2026-10-22 · Detector: Ultralytics YOLO, controlled comparison YOLOv8s vs YOLO26s
- **Decision:** Fine-tune COCO-pretrained **YOLOv8s** (the model named in the plan) and **YOLO26s** with identical data, chips, epochs and augmentations; keep whichever wins on validation F1 and report both.
- **Alternatives:** YOLO11s; CenterNet/CircleNet point detector (1st place); Faster R-CNN (xView3 reference, 4th place used a standard detector pipeline); training from scratch.
- **Reason:** Pre-trained weights (D11) and an anchor-free head are the main reasons YOLO fits: no anchor priors tuned to COCO object sizes, which matter when objects are a few pixels. YOLO26 (Jan 2026) adds Small-Target-Aware Label Assignment (STAL) + ProgLoss and is NMS-free, all directly relevant to tiny, dense vessels, and is slightly better than v8s on COCO at fewer FLOPs (48.6 vs 44.9 mAP). The spec asked for YOLOv8, so rather than switching on a vendor claim, a like-for-like run decides. "s" not "n": free Kaggle T4/P100 can train "s" at 640 comfortably, and small-object recall benefits from the wider network. CircleNet would likely score higher (it won xView3) but is a 12-model ensemble on 2×3090s, out of scope for free GPUs in one week; it's the "next step".
- **Result:** pending.

### D11 · 2026-10-22 · Build vs reuse
- **Decision:** Reuse pre-trained YOLO weights (COCO) and fine-tune; reuse the xView3 metric definition; build only the SAR-specific glue (preprocessing, tiling, CFAR baseline, AIS matching, evaluation breakdowns).
- **Reason:** The interviewer asked last round why I didn't use a pre-trained model. Even though SAR looks nothing like COCO photos, low-level features (edges, blobs, local contrast) transfer and fine-tuning converges much faster than random init on ~50 scenes. CFAR is implemented by hand because it's ~40 lines, it's the baseline, and I need to explain every line of it.
- **Result:** pending.

### D12 · 2026-10-22 · Metrics: xView3-style point matching
- **Decision:** Hungarian matching of predicted vs true points, true positive if ≤ 200 m; report precision, recall, F1 overall, **near-shore (≤ 2 km) vs open sea**, and **by vessel size** (length < 40 m / 40–120 m / > 120 m / unknown). Ground truth uses HIGH + MEDIUM confidence only; LOW labels are ignored (neither TP nor FN), as in the official scorer. Dark-vessel flag scored separately: precision/recall of "dark" among matched detections.
- **Alternatives:** COCO mAP from Ultralytics; official `metric.py` as-is.
- **Reason:** mAP on IoU of artificial 10×10 boxes is not meaningful (D08). The official xView3 metric is the community standard on this dataset, so numbers are comparable with the leaderboard. I reimplement the core matching (≈30 lines) so I can add the size/shore breakdowns, and verify it against the official script on the same predictions.
- **Result:** pending.

### D14 · 2026-10-22 · Two classes: `vessel` and `fixed_structure`; label-noise handling
- **Decision:** Train YOLO with 2 classes: `vessel` (`is_vessel == True`) and `fixed_structure` (`is_vessel == False`: platforms, turbines, buoys). Labels with unknown `is_vessel` are treated as `vessel`. LOW-confidence labels are **kept for training** but **excluded from scoring**.
- **Alternatives:** Single "object" class; drop unknowns; drop LOW labels everywhere.
- **Reason:** Fixed structures never broadcast AIS, so with one class **every oil platform would be flagged as a dark vessel**. The Gulf (Abu Dhabi offshore fields, Hormuz) is full of them. Only `vessel` detections go into the dark-flag step. Dropping unknown/LOW objects from training would leave real bright objects unlabelled, teaching the model to ignore them. Scoring ignores LOW labels to match the official xView3 scorer.
- **Result:** pending.

### D13 · 2026-10-22 · Gulf AIS cross-check: Global Fishing Watch API (gridded)
- **Decision:** Use the GFW 4Wings API: `public-global-sar-presence` (GFW's own SAR detections with `matched=true/false` to AIS) and `public-global-presence` (AIS vessel presence) over each AOI and scene date.
- **Alternatives:** AISHub (requires contributing a receiver), MarineTraffic/Spire (commercial).
- **Reason:** It's the only free, documented source of AIS-matched information for the Gulf. Limitation: the API returns gridded counts, not individual tracks, so Gulf "AIS agreement" is per grid cell and per day, not per ship. This is exactly the spec's fallback: full dark-vessel scoring on xView3; Gulf is detection + cell-level AIS agreement.
- **Result:** pending.

### D15 · 2026-10-22 · Lesson from a failing test: CFAR's PFA must be tuned on data
- **What happened:** The first synthetic CFAR test found 0/3 ships. The synthetic sea used single-look (exponential) speckle, which in dB has a std of ≈5.6 dB; a −5 dB ship on −25 dB sea was only ≈4σ above the background, below the threshold of 4.75σ implied by PFA = 1e-6 under a Gaussian assumption.
- **Fix:** Sentinel-1 GRDH is multi-looked (ENL ≈ 4.4), so the test now uses Gamma(4.4) speckle; all ships are found.
- **Decision:** Treat `pfa` as a **tuned hyperparameter** (sweep on xView3 validation scenes, pick the best F1), not as a value trusted from theory. The Gaussian-in-dB model is only an approximation of real clutter tails.
- **Result:** test passes (`pytest tests`); real-data sweep pending.

### D16 · 2026-10-22 · SAR-specific augmentation (changes to Ultralytics defaults)
- **Decision:** `hsv_h=0, hsv_s=0, hsv_v=0.1, scale=0.2, flipud=0.5, mixup=0, degrees=0`, keep mosaic with `close_mosaic=10` (`configs/train.yaml`).
- **Alternatives:** Ultralytics defaults (`hsv_s=0.7, hsv_v=0.4, scale=0.5, flipud=0`).
- **Reason:** Defaults are tuned for RGB photos. Our three channels are physical measurements (VV, VH, VV−VH in dB), so hue/saturation shifts change their relationship meaninglessly. ±50 % scaling would shrink few-pixel ships below the stride-8 grid. SAR has no canonical "up", so vertical flips are free extra data. Mild brightness jitter is kept on purpose: it mimics calibration / processing differences, which is exactly the Gulf domain shift (D04). The 1st-place xView3 solution likewise limited augmentation to flips, small scale, mild brightness/contrast and noise.
- **Result:** pending.

### D17 · 2026-10-22 · Gulf land mask from OpenStreetMap coastlines
- **Decision:** Build land polygons per AOI from OSM `natural=coastline` ways (Overpass API), using OSM's convention that land lies to the left of the way. Rasterise onto each scene's grid, buffer by 100 m. xView3 scenes use their own `owiMask` raster.
- **Alternatives:** Natural Earth 1:10m land (≈1 km detail, misses port basins); GSHHG (older, coarse at port scale); ESA WorldCover 10 m (another large download + account).
- **Reason:** Khalifa Port is reclaimed land with dredged basins; only a metre-level, current coastline captures it. OSM is free (ODbL) and needs no account.
- **Obstacles:** public Overpass servers returned HTTP 502 repeatedly → added retries + mirror fallback. The first land/sea classification looped over every segment × face and took > 8 min → rewrote with a shapely STRtree spatial index (seconds).
- **Result:** Khalifa Port AOI = 25 land polygons, 23.1 % land, visually checked (`results/figures/khalifa_land_mask.png`). **Strait of Hormuz AOI = 0 % land**: no coastline crosses the box, so it is a pure open-water AOI. Kept on purpose: Khalifa tests near-shore/port clutter, Hormuz tests open-sea traffic lanes, mirroring the near-shore vs open-sea split used on xView3.

### D18 · 2026-10-22 · xView3 public-split labels may not be released → fallback split
- **Finding:** SARFish (built on xView3) lists the public partition as *without labels*; the xView3 paper says "we release all data but the holdout set". Ambiguous until we can see the download page.
- **Decision:** If `public.csv` with labels is downloadable → train on validation, test on public (D01). If not → split the 50 validation scenes **by scene** into train / val / test (≈ 30 / 8 / 12), stratified by region.
- **Also found:** validation split = **123.1 GB** for 50 scenes (IEEE GRSS EOD dataset page) → confirms streaming + chipping (D05).
- **Result:** pending (needs registration).

### D19 · 2026-10-22 · Training data available *now*: Open SAR Ship Detection Dataset (OSSDD)
- **Decision:** While xView3 / CDSE / GFW accounts are pending, train and evaluate the detector on **OSSDD** (Hugging Face `sylviaHoch/OpenSARShip-Ship-Detection-Dataset`, Hammer et al. 2026, arXiv:2608.01963, CC BY-NC-SA 4.0). Full dataset downloaded as `ossdd.zip` (14.4 GB); WebDataset shards used for a first run.
- **Alternatives:** wait for xView3 registration; SSDD / HRSID (smaller, mostly not Sentinel-1 VV+VH); SARFish (needs xView3 labels anyway).
- **Reason:** It's the only open, ungated, **Sentinel-1 GRD dual-pol (VV+VH)** ship dataset with boxes: same sensor and polarisations as xView3 and the Gulf. Non-commercial licence fits a portfolio project. Limits: no AIS/"manual" source labels (no dark-flag scoring), no georeferencing, no land mask (only per-chip `land_fraction`), single class (no fixed structures).
- **Inspection (val-000000.tar):** 200 chips, 142 with ships, 327 ships, up to 39 per chip, median box 13 × 12 px, 45 chips contain land. Pixel values are **raw GRD amplitude DN** (sea VV ≈ 82, VH ≈ 33), not calibrated dB.
- **Approximate calibration:** σ⁰ = DN² / A² (Sentinel-1 Product Specification). The per-product LUT isn't included, so a constant A ≈ 550 (typical IW GRDH σ⁰ LUT magnitude) is used → sea ≈ −16.5 dB (VV) / −24.4 dB (VH), realistic values. Expected error ≈ 1–2 dB; covered by the fixed clip range and the brightness augmentation (D16). Logged as a domain-shift source vs xView3/Gulf data.
- **Result:** pending.

### D20 · 2026-10-22 · OSSDD chip geometry: crop, don't resize; real boxes
- **Decision:** 700 px train chips → centre-crop to 640 (boxes clipped, dropped if < 50 % survives); 512 px val/test chips → zero-pad to 640. Use OSSDD's real axis-aligned boxes.
- **Reason:** Same rule as D07: never resample, ships are only ~13 px. OSSDD has real ship extents, unlike xView3 points, so there's no reason to use synthetic boxes; evaluation stays point-based (box centres) so numbers are comparable.
- **Result:** pending.

### D21 · 2026-10-22 · Fair baseline-vs-model protocol on OSSDD
- **Decision:** Tune CFAR's PFA and YOLO's confidence threshold on **val** only (max F1), report **test** once. Point matching ≤ 200 m (20 px at 10 m). Near-shore proxy = chips with `land_fraction > 0`.
- **Reason:** Tuning on test would inflate numbers. CFAR runs without a land mask here (OSSDD has none), which hurts it on land chips, so the **open-water** subset is the fair head-to-head and land chips show the value of learned context.
- **Result:** pending.

### D22 · 2026-10-22 · Local GPU training (GTX 1650, 4 GB) in a project venv
- **Decision:** Project-local `.venv` (system packages visible) with the CUDA 12.6 PyTorch build; system Python untouched.
- **Obstacle:** first install silently kept the CPU build (pip saw torch already satisfied via system site-packages) → reinstalled with `--ignore-installed`.
- **Result:** pending.

### D23 · 2026-10-23 · Local training settings forced by the GTX 1650
- **Finding 1 (hardware):** Ultralytics' `check_amp` hard-codes GTX 16-series cards (1660/1650/…) as unsafe for mixed precision (FP16 NaN losses), so training runs in **FP32**: ~2× memory, slower. Batch 8 reported 7.9 GB on a 4 GB card → Windows spilled into shared system RAM (4.4 s/it). Batch 4 still spilled (5.9 GB, 3.1 s/it) and was *slower per epoch* (149 it × 3.1 s ≈ 7.7 min vs 75 it × 4.4 s ≈ 5.5 min).
- **Finding 2 (my bug):** the `--workers 0` override was ignored (`if getattr(a, k)` treats 0 as unset), so 4 data-loader processes ran, each a full Python process, which is what pushed system RAM to the limit and got earlier runs killed. Fixed (`is not None`).
- **Decision:** batch 8, 2 loader workers, **40 epochs, patience 10** (was 60/15) to fit the laptop; same settings for both models so the comparison stays fair. The first, aborted batch-8 run (6 epochs, val mAP50 0.46 → 0.54) is kept in `runs/yolov8s_batch8_aborted/` for reference.
- **Alternatives:** Kaggle T4 (16 GB, FP16; needs account); nano models (≈3× faster but weaker on small objects, contradicts D10).
- **Result:** pending.

### R01 · 2026-10-23 · First real results: CFAR vs fine-tuned YOLOv8s on OSSDD test (200 chips, 1,171 ships)
Commands: `python -m src.eval_chips --name cfar` · `scripts/run_training.ps1` (train + `src.eval_chips --weights runs/yolov8s/weights/best.pt`). Files: `results/metrics_cfar.json`, `results/metrics_yolov8s.json`, `results/figures/metrics_comparison.png`, `results/failures/`.

| | Precision | Recall | F1 | FP | F1 land chips | F1 open water |
|---|---|---|---|---|---|---|
| CFAR (PFA 1e-10, tuned on val) | 0.611 | 0.954 | 0.745 | 712 | 0.546 | 0.818 |
| YOLOv8s (conf 0.15, tuned on val) | 0.872 | 0.939 | 0.904 | 162 | 0.790 | 0.940 |

- YOLOv8s: early-stopped at epoch 26 (best 21, val mAP50 0.698), 1.8 h on the GTX 1650.
- **What the failure images show:** (1) CFAR's false alarms are largely **fragmentation**: one ship split into 2 detections by bright scatterers / sidelobes, plus harbour structures on land chips (no land mask in OSSDD). (2) YOLO's misses are faint ships and **motion-smeared ships with wakes**. (3) Several "false alarms" shared by both detectors look like **real unlabelled ships**, so precision is likely under-estimated by label noise (same issue the xView3 winners reported).
- **Caveat:** size breakdown uses label-box extent × 10 m, which includes sidelobe blur, so it's not true vessel length.

### D24 · 2026-10-23 · YOLO26s cannot be trained on the 4 GB GTX 1650 → deferred to a 16 GB GPU
- **What happened:** overnight run failed silently; re-run with logging (`results/train_yolo26s.log`) shows CUDA OOM in the **first batch**: a single allocation of **8.86 GB** on a 4 GB card, even at batch 4. A 1-epoch test on 2 % of the data had passed, because it never hit a crowded batch.
- **Cause (as far as the log shows):** FP32 forced on GTX 16-series (D23) + YOLO26's label-assignment tensors growing with (anchors × targets); OSSDD port chips hold up to 46 ships and mosaic combines 4 chips → ~150+ targets per image.
- **Rejected workaround:** disabling mosaic / capping targets only for YOLO26 would make the YOLOv8s-vs-YOLO26s comparison unfair (D10 requires identical settings).
- **Decision:** report YOLOv8s as the model; run the YOLO26s comparison on a Kaggle T4 (16 GB, FP16) with the same config if time allows. Otherwise state it as a limitation.
- **Result:** pending (needs Kaggle).

### D25 · 2026-10-23 · Gulf domain shift: noise-floor harmonisation + CFAR speckle guard
- **What happened (first Gulf run, Khalifa Port, S1D 21 Oct 2026 02:14 UTC, fetched via Sentinel Hub):** CFAR found **0** objects and YOLOv8s **7**, missing an obvious bright ship.
- **Diagnosis:** Sentinel Hub subtracts thermal noise and clips negatives to 0 → **37.5 % of VH pixels are exactly zero** (calm Gulf sea is below the noise level). Sea medians: Gulf VV −22.7 / VH −36 dB vs OSSDD VV −18.7 / VH −25 dB (OSSDD = raw DN, noise not removed). Zeros (→ −60 dB) blow up CFAR's background std; YOLO sees a darker-than-ever sea.
- **Fix 1, noise floor:** clip Gulf VV/VH at the OSSDD **1st percentile** (VV −29.2 dB, VH −31.9 dB, measured on val chips), i.e. "below noise" looks the same as in training. Only affects sub-noise sea; ships/land are far brighter. Option `--noise-floor` in `src.predict`.
- **Fix 2, CFAR speckle guard:** flooring creates flat patches with σ ≈ 0 → 1,877 CFAR hits. Background σ is now bounded below by the speckle std of multi-look GRD, 4.34·√ψ₁(4.4) = **2.19 dB** (`min_sigma_db=2.2`). Re-checked on OSSDD: CFAR test F1 0.745 → 0.746, so the baseline comparison is unaffected.
- **Not done (on purpose):** no tuning on the Gulf scene (no labels). Floors come from training data, the guard from speckle theory.
- **Result:** Khalifa Port: YOLOv8s 15 detections (several clearly ships incl. the one missed before; 3 near-duplicate pairs), CFAR 43 (many on coastline / breakwater edges). Figures: `results/figures/gulf_khalifa_cfar_vs_yolo.png`, `presentation/figures/demo_gulf_khalifa_yolov8s.png`. Qualitative only; AIS cross-check pending (GFW token).

### D26 · 2026-10-23 · Gulf dark-vessel flag with GFW gridded AIS; duplicate merging
- **Decision:** For Gulf scenes, a vessel detection is AIS-matched if GFW AIS presence (`public-global-presence`, HIGH = 0.01° cells, hourly, grouped by vessel) has any vessel in its cell or the 8 neighbouring cells within ±1 h of acquisition; otherwise it's a **dark candidate**. Also report agreement with GFW's own S1 detections (`public-global-sar-presence`, matched/unmatched).
- **Why this date:** GFW AIS has ~4-day and SAR ~5-day latency, so the 21 Oct scene was too new → used the 15 Oct 02:14 UTC pass.
- **Duplicate merging:** YOLO left near-duplicate detections 1–4 px apart (NMS at IoU 0.7 on ~13 px boxes doesn't remove them) → greedy merge within 50 m, highest score kept (`merge_close` in `src.predict`).
- **Limitation (state it):** cell-level gating with ±1 cell (~3 km window) is lenient in busy ports: a dark ship next to an AIS ship counts as matched, so dark counts are a **lower bound**; dark flags are investigation leads, not proof. Per-ship matching needs exact AIS positions (commercial) or xView3 labels.
- **Result (Khalifa Port, 15 Oct 2026):** 13 YOLOv8s detections → 12 AIS-matched, **1 dark candidate**; 5/13 within ±1 cell of a GFW SAR detection (GFW: 6 matched + 3 unmatched). Files: `results/gulf/khalifa_port_2026-10-15_yolov8s_floor_dark*.{csv,json}`, `results/figures/khalifa_port_2026-10-15_yolov8s_floor_dark.png`, `presentation/figures/demo_gulf_khalifa_dark.png`.

### D27 · 2026-10-23 · Hormuz AOI refined from AIS density; Larak anchorage results
- **Finding:** the original Hormuz box (open strait, 26.4–26.62 N / 56.2–56.45 E) had full S1 coverage on 15 Oct but almost no traffic: YOLOv8s 2, CFAR 5, GFW SAR 2 (both AIS-matched), 11 AIS positions all day. GFW AIS density over 55.8–57.0 E / 25.9–27.0 N (`results/figures/hormuz_ais_density_2026-10-15.png`) shows traffic near Khasab / west Musandam and at the **Larak–Qeshm anchorage**, and **no continuous traffic-lane band in AIS at all**.
- **Decision:** keep the open-strait box as a low-traffic contrast; add `aoi/hormuz_larak_anchorage.geojson` (56.25–56.50 E, 26.80–27.02 N) as the main Hormuz AOI, chosen from data, not guessed. Land mask: OSM, 2 polygons, 10.1 % land.
- **Obstacles:** the projected UTM box was 2,504 px > Process API limit 2,500 → clamp to 25 km around the centre (`gulf_fetch.py`). 16 Oct pass covered only 57 % of the box → used the 15 Oct 02:14 pass (same orbit as the Khalifa scene). One background run killed for low RAM → reran steps in the foreground.
- **Result (15 Oct 2026, 02:14 UTC):** YOLOv8s **40** detections → **24 AIS-matched, 16 dark candidates (40 %)**; CFAR 58. GFW's own SAR on the same pass: 13 matched + **17 unmatched (57 %)**; **30 / 40** of our detections are within ±1 cell of a GFW detection, so there's independent agreement. Contrast: Khalifa Port 1 / 13 dark. Visible miss: one large bright object (top right). Natural slicks/eddies not flagged. Files: `results/gulf/hormuz_larak_anchorage_2026-10-15_*`, `results/figures/hormuz_larak_anchorage_2026-10-15_yolov8s_floor_dark.png`, `presentation/figures/demo_gulf_hormuz_dark.png`.
