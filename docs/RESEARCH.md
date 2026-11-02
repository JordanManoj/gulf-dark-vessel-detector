# Research Notes: Dark Vessel Detector (Gulf)

For each technique, dataset, model and metric used: **what it is → why it fits → alternatives → sources → pitfalls**.
Decisions that follow from this research are in [`../DECISIONS.md`](../DECISIONS.md) (referenced as D01…D27). Last updated 23 Oct 2026, after the build: section 0 records what was actually chosen, sections 9–15 cover what was researched during the build.

---

## 0. Decision matrix: what we chose, what we rejected, and why

One row per component. "Status" says whether the alternative was tested in this repo or rejected from research only.

| Component | Chosen | Alternatives considered | Why chosen | Why not the alternatives | Status / evidence |
|---|---|---|---|---|---|
| **Imagery (training)** | Sentinel-1 IW GRD, VV+VH | Optical (Sentinel-2), commercial SAR (ICEYE, Capella) | Free, global, day/night, through cloud; same sensor as the Gulf scenes | Optical fails at night and under cloud; commercial SAR not free | Research (§1–2) |
| **Training dataset** | **OSSDD** (Open SAR Ship Detection Dataset, Hugging Face): 595 / 200 / 200 chips | xView3-SAR; SSDD; HRSID; OpenSARShip; SARFish | Only open, ungated Sentinel-1 **dual-pol** set with boxes; downloadable without accounts | xView3: registration, 123 GB validation split, train split blind to dark ships (AIS-only labels). SSDD/HRSID: mostly other sensors / single-pol, small chips. SARFish: needs xView3 labels anyway | Used (§9, D19) |
| **Dark-vessel labels** | xView3 `validation.csv` / `public.csv` (downloaded, SHA1-verified) | Build own AIS-SAR matches; GFW per-detection data | Only per-ship AIS-vs-manual labels available | Own matching has no ground truth | Labels downloaded; imagery pending (D18) |
| **Gulf imagery access** | CDSE **Sentinel Hub Process API**: σ⁰, thermal-noise removed, orthorectified, 10 m, cropped to AOI | Full SAFE download (1.7 GB) + ESA SNAP; Google Earth Engine; Microsoft Planetary Computer | Same physical quantity as xView3 for a 25 km crop; free quota 10,000 PU/month | SNAP: heavy install + 1.7 GB per scene; GEE: second ecosystem/account; Planetary Computer RTC is γ⁰ (terrain-flattened), not σ⁰ | Used (D04) |
| **AIS source (Gulf)** | **Global Fishing Watch** 4Wings API (AIS presence + GFW's own SAR detections) | MarineTraffic, Spire, exactEarth; AISHub; NOAA / Danish AIS | Only free AIS covering the Gulf; also gives an independent SAR reference | Commercial; AISHub needs own receiver; NOAA/DMA don't cover the Gulf. Limitation: gridded (~1 km, hourly) | Used (D13, D26) |
| **Pre-training** | **COCO** weights (Ultralytics YOLOv8s) | ImageNet backbone; SAR self-supervised backbones (SSL4EO-S12, CROMA); from scratch | Complete detector incl. a trained head; low-level features transfer; 1.8 h to F1 0.90 | ImageNet: classification only, head from scratch. SAR SSL: strongest option but backbones, not detectors, so need a custom head (out of scope for one week). Scratch: ~600 chips → underfits | COCO used; others = research (§10) |
| **Detector** | **YOLOv8s** | YOLO26s; YOLO11s; YOLOv8-P2; CircleNet/CenterNet (xView3 winner); Faster R-CNN; DETR | Anchor-free, fast, fits 4 GB GPU, named in the plan | YOLO26s: **tested, out of memory** (8.9 GB single allocation). CircleNet: 12-model ensemble on 2×3090. Faster R-CNN: xView3 reference scored 0.19. DETR: data-hungry, weak on small objects | v8s used; 26s failed (D24) |
| **Baseline** | Two-parameter **CFAR** on VH + speckle guard | Global threshold; CA-CFAR; OS-CFAR; K-distribution CFAR | Operational standard (SNAP, GFW); O(1) per pixel via box filters | Global threshold ignores sea state; OS-CFAR much slower; K-dist fragile | Used (§3, D09, D25) |
| **Input encoding** | VV, VH, VV−VH in dB; **fixed** clip [−50, 20] dB | Per-chip min-max; sigmoid (1st place); + bathymetry/wind | Keeps absolute brightness across chips and regions | Per-chip destroys absolute intensity; ancillary layers didn't help the winner and don't exist for Gulf crops | Used (D06) |
| **Tiling** | 640 px tiles at native 10 m, **core + halo** (512 + 64) | Resize big tiles; overlap + NMS (SAHI) | Ships ~13 px: never resize; cores partition the scene, so no duplicates | Resizing loses small ships; cross-tile NMS needs a tuned threshold | Used (D07, D07b) |
| **Augmentation** | No hue/sat, scale ±20 %, flips both ways, mild brightness, no mixup | Ultralytics defaults | Channels are physics, not colours; protect tiny ships | Defaults (scale ±50 %, hsv_s 0.7) made for RGB photos | Used (D16) |
| **Land mask** | **OpenStreetMap** coastlines (Overpass) → polygons | Natural Earth 1:10m; GSHHG; ESA WorldCover | Metre-level, current (Khalifa's reclaimed land) | Natural Earth ~1 km detail; GSHHG older; WorldCover large download + account | Used (D17) |
| **Domain-shift fix** | Noise floor at training 1st percentile + CFAR σ ≥ speckle (2.2 dB) | Tune thresholds on Gulf; retrain on Gulf; leave as is | Physically motivated, no Gulf labels needed, didn't change test F1 (0.745 → 0.746) | Tuning on Gulf without labels = overfitting to my eye; retraining needs labels | Used (§13, D25) |
| **Dark-flag logic** | Detection + AIS presence within ±1 cell and ±1 h | Learn "dark" as a class | Dark is a property of the AIS record, not the pixels | A learned class would learn region/ship-type shortcuts | Used (D03, D26) |
| **Metric** | xView3-style point matching (Hungarian, ≤ 200 m), P/R/F1 + land/open-water + size | COCO mAP | Matches operational need and the xView3 leaderboard | IoU on marker boxes is meaningless | Used (§7, D12) |
| **Compute** | Laptop GTX 1650 (4 GB) in a project venv | Kaggle T4; Colab | Available immediately, no account | FP32 only on GTX 16-series → slow; YOLO26s doesn't fit → Kaggle next | Used (D22, D23) |

---

## 1. SAR imaging basics

**What it is.** Synthetic Aperture Radar is an *active* sensor: the satellite transmits microwave pulses and records the echo. Each pixel's brightness is **backscatter** (σ⁰, "sigma-nought"), the fraction of energy returned to the sensor per unit area. Because it brings its own illumination and C-band (≈5.5 cm) passes through cloud, SAR works **day and night in any weather**, which optical satellites can't.

**Why ships show up.** Calm-to-moderate sea acts like a mirror and reflects energy *away* from the sensor (dark). A steel hull and superstructure form **corner reflectors** (dihedral/trihedral angles) that bounce energy straight back (bright). So a ship is a bright point on a dark background.

**Polarisation.** Sentinel-1 transmits vertically (V) and receives both V and H:
- **VV** (co-pol): strong sea return, good for sea-surface features (wind, slicks); more sea clutter.
- **VH** (cross-pol): sea returns very little because cross-pol needs depolarising multiple bounces, but ships produce plenty. **Better ship/clutter contrast**, which is why the CFAR baseline uses VH (D09). The xView3 whitepaper says the same.
- The **VV−VH ratio (dB)** is a cheap extra feature, used as our 3rd channel (D06).

**Speckle.** Each resolution cell holds many random scatterers whose echoes interfere, so even a uniform sea looks "grainy" (multiplicative noise, exponentially distributed intensity in single-look data). Consequences: single bright pixels are often noise; detectors must use **local statistics** (CFAR) or learned context (CNN) rather than one fixed threshold.

**Pitfalls for ship detection:** azimuth ambiguities ("ghost" ships displaced along-track from bright targets), sidelobes/cross-shaped diffraction spikes around very bright ships, wind-roughened sea raising clutter, fixed infrastructure (oil platforms, wind turbines, buoys) looking exactly like ships, and land/harbour clutter.

**Sources**
- ESA Sentinel-1 SAR User Guide: https://sentiwiki.copernicus.eu/web/s1-mission
- xView3 whitepaper, "Developing ML algorithms to combat IUU fishing" (2021): https://iuu.xview.us/xview3_dataset_whitepaper.pdf (VH vs VV for ship detection)

---

## 2. Sentinel-1 GRD products

**What it is.** Sentinel-1 is ESA's C-band SAR constellation. **IW (Interferometric Wide swath)** is the default mode over land and coastal seas: 250 km swath, ~20 × 22 m resolution. **GRD (Ground Range Detected)** is the Level-1 product: amplitude only (phase discarded), multi-looked, projected to ground range, **10 m pixel spacing** (GRDH). File naming example: `S1D_IW_GRDH_1SDV_…`, where `1SDV` means dual polarisation VV+VH.

**Why it suits the problem.** Free and open, global coastal coverage, both polarisations, and it's exactly what xView3 was built from, so the training and Gulf data are the same sensor.

**What we verified (22 Oct 2026, anonymous CDSE OData query).** 38 IW GRDH scenes over Khalifa Port and 102 over Hormuz between 1 Jul and 21 Oct 2026, all `1SDV`, ~1.7 GB each. Sentinel-1C and **1D** are both operational, giving a ~3-day revisit over Khalifa Port.

**Processing chain to analysis-ready σ⁰ (dB).** Apply orbit file → thermal noise removal → radiometric calibration (DN → σ⁰) → terrain correction (geocode with a DEM) → 10·log₁₀ to dB. xView3 used exactly this chain in ESA's Sentinel-1 Toolbox (SNAP). We get the equivalent from the **CDSE Sentinel Hub Process API** (`backCoeff=SIGMA0_ELLIPSOID`, `orthorectify=true`, thermal-noise removal always applied) for just the AOI (D04).

**Alternatives:** full SAFE download + SNAP (exact but 1.7 GB/scene and a heavy desktop tool); Google Earth Engine `COPERNICUS/S1_GRD` (already σ⁰ dB, different account/ecosystem); ASF Vertex.

**Pitfalls:** GRD resolution (~20 m) is coarser than pixel spacing (10 m), so a 15 m fishing boat is only 1–2 *real* resolution cells; border noise at swath edges; incidence-angle dependence of sea clutter (near range brighter); different DEM / resampling between SNAP and Sentinel Hub introduces small radiometric differences (part of domain shift).

**Sources**
- CDSE OData API: https://documentation.dataspace.copernicus.eu/APIs/OData.html
- CDSE Sentinel Hub S1 GRD (bands, `backCoeff`, orthorectification, thermal noise): https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html
- CDSE quotas (Sentinel Hub: 10,000 PU/month, 10,000 requests/month, 300 PU/min): https://documentation.dataspace.copernicus.eu/Quotas.html

---

## 3. CFAR detection (the baseline)

**What it is.** *Constant False Alarm Rate* detection sets a **local** threshold for each pixel from the statistics of the surrounding sea, so the false-alarm rate stays constant whether the sea is calm or rough. Three nested windows slide over the image:
- **target window**: the pixel(s) under test, ≈ smallest object;
- **guard ring**: excluded so a large ship doesn't contaminate its own background, ≈ largest object;
- **background ring**: used to estimate clutter statistics, big enough for stable estimates.

**Two-parameter CFAR** (what ESA SNAP's *Adaptive Thresholding* operator implements) flags pixel *x* if `x > μ_b + t·σ_b`, where μ_b, σ_b are the background mean and std and *t* comes from the chosen probability of false alarm (PFA) under a Gaussian assumption. On dB data the sea is roughly log-normal, so a Gaussian on dB values is a reasonable approximation.

**Variants and why we chose two-parameter (D09):**
| Variant | Background estimate | Strength | Weakness |
|---|---|---|---|
| CA-CFAR | mean of reference cells | optimal in homogeneous clutter | fails at **clutter edges** (coastline) and with **multiple targets** (busy anchorages) |
| GO/SO-CFAR | greatest/smallest of half-windows | better at edges | poor with close targets |
| **OS-CFAR** | k-th order statistic (e.g. 0.75·N) | robust to interfering targets (Blake 1988) | needs sorting per window, much slower on 25 km × 25 km |
| Distribution-based (K, Gamma, G⁰) | fitted clutter model | accurate tails in heavy clutter | parameter estimation per window, slow, fragile |
| **Two-parameter (ours)** | mean + std of background ring | the operational standard; computable with integral images in O(1)/pixel | Gaussian assumption; still hurt by multiple targets |

**Why it's the right baseline:** it's the "no-ML" industry approach (SNAP; GFW's automated detector, which seeded xView3's labels, is CFAR-based), so the ML model must beat it to justify itself.

**Pitfalls:** dense ports (multiple-target masking), coastline edges without a land mask, bright fixed structures, azimuth ghosts; PFA must be tuned on held-out data, not on the test set.

**Sources**
- ESA SNAP Object Detection / Adaptive Thresholding help (two-parameter CFAR, window-size rule): https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.feature.extraction.ui/operators/ObjectDetection.html
- Blake, S. (1988). *OS-CFAR theory for multiple targets and nonuniform clutter.* IEEE Trans. AES 24(6). https://ui.adsabs.harvard.edu/abs/1988ITAES..24..785B/abstract
- *Modified rank-sum nonparametric CFAR to combat clutter edge* (EURASIP JASP, 2023), summary of CA/GO/SO/OS behaviour: https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-023-01032-z
- Correlation-based joint CFAR with truncated statistics (Sensors 2017), the multiple-target problem in SAR: https://pmc.ncbi.nlm.nih.gov/articles/PMC5419799

---

## 4. YOLO detectors (YOLOv8 and YOLO26)

**What it is.** Single-stage detectors from Ultralytics: one network pass predicts boxes + classes on a grid at three scales (strides 8, 16, 32), no separate region-proposal stage.
- **YOLOv8 (2023):** CSP-style backbone with **C2f** blocks, PAN-FPN neck, **anchor-free decoupled head** (separate classification and box branches), **Task-Aligned Assigner** for matching predictions to ground truth, losses = BCE (cls) + CIoU + **Distribution Focal Loss** (box), NMS at inference.
- **YOLO11 (2024):** C3k2 blocks + C2PSA attention; 39.5 vs 37.3 COCO mAP (n) with fewer params than v8.
- **YOLO26 (Jan 2026):** **NMS-free** end-to-end head (one-to-one assignment, idea from YOLOv10), **DFL removed**, **ProgLoss + STAL (Small-Target-Aware Label Assignment)**, **MuSGD** optimiser. COCO mAP 48.6 (26s) vs 47.0 (11s) vs 44.9 (v8s), with 20.9 vs 21.6 vs 28.6 GFLOPs.

**Why YOLO suits this problem:**
1. **Pre-trained weights** (COCO): fine-tuning beats training from scratch on ~50 scenes (D11).
2. **Anchor-free**: no anchor boxes tuned to COCO object sizes; anchors that are too large are a classic cause of missed tiny objects.
3. Fast inference: a 25 km AOI is ~25 tiles of 640 px.
4. Mature tooling (augmentation, export, metrics), so the week goes into SAR-specific work.

**Does small-object handling, anchor-free design or tile overlap matter here? (question asked by the plan)**
- **Small objects: yes, a lot.** At 10 m spacing most vessels are 2–10 px. The finest YOLO head is stride 8, so a 3 px ship is less than half a cell. Mitigations, in order of cost: (a) never downscale, tile at native resolution (D07); (b) YOLO26's STAL assigns small targets to more positive anchor points; (c) Ultralytics ships a **P2 head** config (`yolov8-p2.yaml`, stride 4), the next experiment if small-vessel recall is low.
- **Anchor-free: yes, but it's already the default** in v8/11/26, so it's not a design choice we have to make, just a reason the family fits.
- **Tile overlap: yes, at inference.** Without overlap, ships cut by a tile border are split or missed. Overlap ≥ largest ship (~40 px) makes every ship whole in at least one tile; duplicates are merged in scene coordinates (D07).

**Alternatives considered:** CenterNet-style point detectors (CircleNet, the xView3 winner, natural fit for point labels); Faster R-CNN (xView3 reference, 0.19 aggregate); DETR family (slow to converge on small data); SAR-specific YOLO variants (SAR-LtYOLOv8, AC-YOLO), which report a few AP points over vanilla YOLO on SSDD/HRSID but have no maintained code or pre-trained weights for our setting.

**Pitfalls:** COCO pre-training is on RGB photos (different statistics, so domain gap at the input); mAP from Ultralytics is IoU-based and misleading with artificial 10 px boxes (D08, D12); default augmentations like HSV jitter are meaningless on SAR and mosaic can paste land into sea, so they're reviewed in the training config.

**Outcome in this project.** YOLOv8s was fine-tuned on OSSDD and reached test F1 0.90 (CFAR 0.75). The planned YOLO26s comparison **could not run**: on the 4 GB GTX 1650, which trains in FP32 because Ultralytics disables mixed precision on GTX 16-series, YOLO26s requested 8.9 GB in a single allocation in its first batch. Its settings were not changed to force a fit, because that would break the like-for-like comparison. It moves to a 16 GB GPU (D23, D24).

**Sources**
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/models/yolov8/
- YOLOv8 vs YOLO11 (official table): https://docs.ultralytics.com/compare/yolov8-vs-yolo11
- YOLO11 vs YOLO26 (STAL, ProgLoss, NMS-free, MuSGD): https://docs.ultralytics.com/compare/yolo11-vs-yolo26
- Sapkota & Karkee, review of YOLO26 architecture incl. STAL (v3, Mar 2026), arXiv:2510.09653: https://arxiv.org/abs/2510.09653
- Wang et al. (2024), *YOLOv10: Real-Time End-to-End Object Detection* (origin of NMS-free training), arXiv:2405.14458
- *SAR-LtYOLOv8* (CSSE 2024), YOLOv8 on HRSID/SSDD: https://www.techscience.com/csse/v48n6/58696/html
- *AC-YOLO* (PLOS ONE 2025), YOLO11 for SAR ships: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327362

---

## 5. The xView3 challenge and winning approaches

**Dataset (xView3-SAR).** Paolo, Lin, Gupta, Goodman, Patel, Kuster, Kroodsma, Dunnmon. *xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery.* NeurIPS 2022 Datasets & Benchmarks. 991 Sentinel-1 IW GRD scenes (2020), avg. 29,400 × 24,400 px, 243,018 labelled objects over 43.2 M km²; VV/VH + bathymetry + wind rasters. Splits: train 554 / validation 50 / public 150 / holdout 237. Regions: Adriatic, Bay of Biscay, Gulf of Guinea, Iceland, Norway; **no Persian Gulf**.

**How labels were made.** GFW's CFAR detector → AIS-to-SAR matching → vessel-type classification from AIS → manual labelling by analysts on 437 scenes → combined with confidence HIGH/MEDIUM/LOW. Label sources: 39.1 % both, 33.4 % manual only, 27.5 % AIS only. **The train split has AIS-derived labels only.**

**Results on holdout (paper Table 1):**

| Rank | Team | Aggregate | F1 detection | F1 near-shore | F1 vessel | F1 fishing | Length (PE) |
|---|---|---|---|---|---|---|---|
| 1 | BloodAxe | 0.618 | 0.770 | 0.531 | 0.939 | 0.843 | 0.697 |
| 2 | selim_sef | 0.605 | 0.763 | 0.477 | 0.935 | 0.808 | 0.744 |
| 3 | Tumen | 0.581 | 0.740 | 0.421 | 0.948 | 0.828 | 0.729 |
| 4 | Skylight (AI2) | 0.578 | 0.732 | 0.467 | 0.929 | 0.823 | 0.725 |
| — | reference Faster R-CNN | 0.190 | 0.430 | 0.129 | 0.689 | 0.395 | — |

**What I took from the winners:**
- **1st (BloodAxe, CircleNet):** CenterNet-like point detector, EfficientNet encoder + U-Net decoder at stride 2, 12-model ensemble. Used **only VV/VH** (bathymetry/wind didn't help), **sigmoid normalisation**, 1024 px training crops centred on objects + 10 % random, **2048 px inference tiles with overlap**, label smoothing for noisy labels, and **did not train on the train split** because it was much noisier. Also noted train/val/test share ~42 locations (leaky CV). → Our D01, D06, D07.
- **4th (Skylight/AI2):** tried temporal tricks but found "a standard object detection pipeline performed best"; **fixed global normalisation, clip VV/VH to [−50, 20] dB**, because per-window min-max destroys absolute intensity. → Our D06, and evidence that a standard detector (YOLO) is a reasonable choice.
- Near-shore F1 (~0.4–0.5) is far below overall F1 (~0.75) **even for winners**. Coastal clutter is the hard part, and our Gulf AOIs (a port, a strait) are near-shore heavy. Set expectations accordingly.

**Sources**
- Paper: https://arxiv.org/abs/2206.00897 · NeurIPS: https://proceedings.neurips.cc/paper_files/paper/2022/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html
- Whitepaper (columns, regions): https://iuu.xview.us/xview3_dataset_whitepaper.pdf
- Official reference code + metric: https://github.com/DIUx-xView/xview3-reference
- 1st place: https://github.com/DIUx-xView/xView3_first_place · https://github.com/BloodAxe/xView3-The-First-Place-Solution
- 4th place write-up: https://github.com/DIUx-xView/xView3_fourth_place (whitepaper.pdf)

**Pitfalls:** label noise (missing labels → model punished for correct detections), leaky geographic splits, enormous scenes (I/O dominates), LOW-confidence labels.

---

## 6. AIS and why vessels go "dark"

**What it is.** The *Automatic Identification System* is a VHF transponder that broadcasts identity (MMSI, name), position, speed and course every few seconds to minutes. Received by coastal stations and satellites. **SOLAS** requires it on ships ≥ 300 GT on international voyages, cargo ≥ 500 GT, and all passenger ships. Many fishing boats and small craft are **not required** to carry it.

**Why vessels are "dark" (not visible in AIS):**
1. Not required (small fishing, dhows, leisure craft), the most common reason;
2. **Deliberately switched off**: IUU fishing, sanctions evasion, smuggling, ship-to-ship transfers;
3. **Spoofing / manipulation**: broadcasting false positions or identities (well documented in the Gulf / Hormuz region);
4. Reception gaps: congested areas saturate receivers, satellite AIS coverage gaps;
5. Matching failures: AIS present but the AIS-to-SAR correlation (time interpolation, position error) misses it.

So "dark" ≠ "illegal": it is an *investigation lead*. Fusing SAR detections with AIS gives exactly that lead list, the same fusion Stellaria's Stella platform and Orbion do.

**Global context.** Paolo et al. (2024, *Nature*) mapped 2017–2021 Sentinel-1 detections globally and found **72–76 % of industrial fishing vessels are not publicly tracked**, i.e. dark is the norm, not the exception.

**Gulf AIS access (D13):** Global Fishing Watch API, free for non-commercial use with a token. `public-global-sar-presence` (S1 detections, `matched` filter, 2017 → 5 days ago) and `public-global-presence` (AIS presence, hourly). **Gridded only via API**; per-detection fields (length, matching score) only in GFW's Data Download Portal.

**Sources**
- Paolo, F. et al. (2024). *Satellite mapping reveals extensive industrial activity at sea.* Nature 625, 85–91. doi:10.1038/s41586-023-06825-8. https://pmc.ncbi.nlm.nih.gov/articles/PMC10764273/
- GFW API docs: https://globalfishingwatch.org/our-apis/documentation · 4Wings: https://globalfishingwatch.org/our-apis/documentation/docs/v3/4wings
- IMO, AIS transponders (SOLAS V/19): https://www.imo.org/en/OurWork/Safety/Pages/AIS.aspx

---

## 7. Detection metrics

**Point-based matching (D12).** Predictions and ground truth are points. Build the pairwise distance matrix (metres), solve the **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) for one-to-one matching, accept pairs ≤ 200 m.
- TP = accepted pairs, FP = unmatched predictions, FN = unmatched ground truth.
- **Precision** = TP/(TP+FP), "how many alarms are real"; **Recall** = TP/(TP+FN), "how many ships we found"; **F1** = harmonic mean.
- **Near-shore F1**: same, restricted to objects ≤ 2 km from shore (xView3 definition).
- Breakdown **by vessel length** (where known) and **near-shore vs open sea** (asked by the plan).
- **Dark-flag quality:** among matched detections, precision/recall of predicting "dark" against `source == manual`.

**Why not mAP?** COCO mAP averages over IoU thresholds 0.5–0.95. With synthetic 10 × 10 px boxes, IoU measures how well we guess an arbitrary box, not whether we found the ship. Point distance is what operations care about, and it matches the xView3 leaderboard so numbers are comparable.

**Why Hungarian, not greedy?** In dense anchorages, greedy nearest-neighbour can steal a match from a closer pair, inflating FPs and FNs. Hungarian gives the optimal one-to-one assignment.

**Pitfalls:** results depend on the tolerance (200 m); LOW-confidence labels must be excluded consistently; with label noise, some "FPs" are real unlabelled ships, so failure images must be inspected by eye.

**Sources**
- Official xView3 scorer `metric.py`: https://github.com/DIUx-xView/xview3-reference/blob/master/reference/metric.py
- Kuhn, H. W. (1955). *The Hungarian method for the assignment problem.* Naval Research Logistics Quarterly 2.

---

## 8. Related work: how others tackle this, and how this project differs

| Work | Approach | Difference to this project |
|---|---|---|
| GFW / Paolo et al. 2024 (Nature) | CFAR + CNN on all S1 coastal imagery, AIS matching, global | operational, global; we replicate the *pipeline idea* at small scale and specifically test **transfer to the Gulf** |
| xView3 winners (2022) | point detectors / ensembles trained on xView3 | evaluated in-distribution (same regions); we measure **out-of-distribution** behaviour |
| SAR-LtYOLOv8, AC-YOLO (2024–25) | modified YOLO on SSDD / HRSID chips | chip datasets, no AIS, no scene-level inference |
| SARFish (2023) | adds SLC (complex) data to xView3 labels | uses phase information; we stay with GRD (what's freely available everywhere) |
| ESA SNAP Ship Detection | two-parameter CFAR + discrimination | that's our **baseline** |

**This project's angle:** a compact, explainable SAR+AIS pipeline (CFAR baseline → fine-tuned YOLO → AIS-gated dark flag), measured on a held-out Sentinel-1 test set *and* honestly assessed out-of-distribution in the Gulf, the operating area of Stellaria / Orbion.

---

## 9. Training data actually used: Open SAR Ship Detection Dataset (OSSDD)

**What it is.** Sentinel-1 GRD chips (VV and VH, float32 GeoTIFF) with axis-aligned and rotated ship boxes, published on Hugging Face (`sylviaHoch/OpenSARShip-Ship-Detection-Dataset`, Hammer et al. 2026, arXiv:2608.01963, CC BY-NC-SA 4.0). Train chips 700 px, val/test 512 px. Full dataset 14.4 GB zip; also WebDataset shards (~0.5 GB each).

**Why it fits.** Same sensor, mode and polarisations as xView3 and the Gulf scenes; open and ungated, so training could start without registrations. Non-commercial licence fits a portfolio project.

**What I measured on it.** 5 train shards + 1 val + 1 test → 595 / 200 / 200 chips with 1,178 / 327 / 1,171 ships; median box 13 × 12 px; 45 of the 200 validation chips contain land; up to 46 ships per chip.

**Alternatives.** xView3-SAR (registration, 123 GB, AIS-only train labels), SSDD and HRSID (small chips, mixed sensors, single-pol), OpenSARShip (chips centred on single AIS-matched ships, which suits classification more than detection), SARFish (adds SLC, still needs xView3 labels).

**Pitfalls.** Pixels are **raw GRD amplitude DN**, not calibrated σ⁰ → converted with σ⁰ = DN² / A² using a typical constant A ≈ 550 (Sentinel-1 Product Specification gives the formula; the per-product LUT isn't included), so expect 1–2 dB error. No georeferencing, no land mask (only `land_fraction` per chip), single class (no fixed structures), no AIS link (no dark-flag scoring). Some ships are unlabelled (label noise seen in the failure gallery).

**Sources.** https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset · ESA *Sentinel-1 Product Specification*, doc. S1-RS-MDA-52-7441, issue 3/3 (2016), §6: L1 calibration LUTs and σ⁰ = DN²/A²

---

## 10. Pre-training options

**What it is.** Starting from weights learned on a large dataset instead of random initialisation, then fine-tuning on the target data.

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **COCO detector weights** (Ultralytics YOLOv8s, ~118k photos, 80 classes) | Full detector incl. head and neck; plug-and-play; standard in SAR ship-detection papers | RGB photo statistics differ from SAR | **Used**: fine-tuned in 1.8 h to F1 0.90 |
| ImageNet classification backbone | Generic features | No detection head → head trained from scratch | Rejected |
| SAR self-supervised backbones: **SSL4EO-S12** (Wang et al., Sentinel-1/2 SSL), **CROMA** (Fuller et al., NeurIPS 2023, radar-optical) | Learned from millions of radar patches: closest domain | Backbones only; need a custom detection head and integration | **Next step** |
| Training from scratch | No domain mismatch | ~600 chips is far too little | Rejected |

**Why COCO transfers to SAR.** Early convolution layers learn edges, blobs and local contrast; a ship in SAR is a bright, compact blob on a dark background. Fine-tuning re-learns the higher layers.

**Where the weights come from.** Ultralytics GitHub release assets, downloaded automatically by `YOLO("yolov8s.pt")` (AGPL-3.0). Stored in `models/`.

---

## 11. AIS access for the Gulf: Global Fishing Watch API

**What it is.** GFW 4Wings report API (`POST https://gateway.api.globalfishingwatch.org/v3/4wings/report`) with a bearer token; free for non-commercial use. Datasets: `public-global-presence` (AIS, one position per vessel per hour, ~4-day latency) and `public-global-sar-presence` (GFW's own Sentinel-1 detections with `matched` = AIS, ~5-day latency). Spatial resolution HIGH = 0.01° (~1.1 km).

**Why it fits.** The only free AIS covering the Gulf, and it adds an **independent radar reference** (GFW SAR detections) to check our detector against.

**What I measured.** Khalifa Port, 15 Oct 2026: 1,267 hourly AIS rows, 29 AIS cells near the 02:14 UTC pass; GFW SAR 6 matched / 3 unmatched. Hormuz Larak anchorage, same pass: 288 AIS rows, GFW SAR 13 matched / 17 unmatched. Wider Hormuz AIS density (55.8–57.0 E, 25.9–27.0 N) showed traffic near Khasab / Musandam and the Larak anchorage, and **no continuous lane band** through the strait.

**Pitfalls.** Gridded, not per ship → dark counts are a lower bound in busy cells. Latency means very recent scenes can't be checked. Group-by is required; the response nests rows per dataset.

---

## 12. Land masks from OpenStreetMap

**What it is.** OSM tags shorelines as `natural=coastline` ways drawn with **land on the left**. Fetched per AOI via the Overpass API, split the AOI box along them (shapely `polygonize`), and classified each face by probing points left and right of each segment (STRtree spatial index).

**Why.** Metre-level and current: it captures Khalifa Port's reclaimed island and dredged basins, which Natural Earth (~1 km) misses.

**Pitfalls.** Public Overpass servers fail often (502) → retries + mirrors. Coastline detail varies; CFAR still fires where the mask edge meets the shore (failure case 2), so a 100 m buffer is applied.

---

## 13. Domain shift between training data and Gulf scenes (measured)

| Quantity | OSSDD (training) | Khalifa Port | Hormuz anchorage |
|---|---|---|---|
| Sea VV median | ≈ −18.7 dB | −22.7 dB | −24.4 dB |
| Sea VH median | ≈ −25 dB | −36 dB | below noise |
| VH pixels exactly 0 (below noise) | none | 37.5 % | 57.6 % |

**Cause.** OSSDD is raw DN, with the thermal noise floor still in it; Sentinel Hub subtracts thermal noise and clips negatives to zero. The Gulf sea was also calm (dark).

**Effect.** First Gulf run: CFAR 0 detections (zeros inflate the background std), YOLO missed an obvious ship.

**Fix (physics, no Gulf labels used).** (1) Clip Gulf VV/VH at the OSSDD 1st percentile (VV −29.2 dB, VH −31.9 dB). (2) CFAR background σ ≥ speckle std of multi-look GRD, 4.34·√ψ₁(4.4) = 2.19 dB. Test-set CFAR F1 unchanged (0.745 → 0.746).

**Alternatives.** Tune thresholds on the Gulf (no labels → overfitting to my eye); retrain with Gulf data (needs labels); histogram matching (would also shift ship brightness).

---

## 14. Compute constraints

GTX 1650 (4 GB). Ultralytics `check_amp` disables mixed precision on GTX 16-series (FP16 NaN risk), so training runs in FP32: YOLOv8s batch 8 reported 7.9 GB and spilled into shared system memory (~4.4 s/it). Batch 4 still spilled and was slower per epoch. Final: batch 8, 2 workers, 40 epochs, early stop at 26. YOLO26s OOM (8.9 GB single allocation). Next: Kaggle T4 (16 GB, FP16).

---

## 15. Where each piece of research shows up in results

| Research finding | Result it explains |
|---|---|
| CA/two-parameter CFAR weak with multiple targets and clutter edges (§3) | CFAR F1 on land chips 0.55; fragmentation and coastline false alarms (failure cases 1–2) |
| xView3 winners: near-shore much harder (§5) | Our biggest gain is on land chips (0.55 → 0.79) |
| Small objects need native resolution (§4) | Tiles never resized; median ship 13 px |
| Label noise in SAR datasets (§5) | Ship-shaped "false alarms" in failure case 4 → precision is a lower bound |
| Thermal noise removal in Sentinel Hub (§2) | Gulf domain shift and the noise-floor fix (§13) |
| Gridded AIS (§6, §11) | Dark counts are a lower bound; Khalifa 1/13 vs Hormuz 16/40 |
