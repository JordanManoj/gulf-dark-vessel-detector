"""Sanity tests on synthetic data: these check the logic, they are not results."""
import numpy as np
import pandas as pd

from src.cfar import CfarParams, cfar_detect
from src.dark import flag_dark
from src.evaluate import detection_metrics, match_points, match_table
from src.sar import encode_3ch, valid_mask
from src.tiling import iter_windows, pad_to


def synthetic_scene(seed=0, h=600, w=600, ships=((100, 100), (300, 450), (500, 220))):
    rng = np.random.default_rng(seed)
    # Speckled sea around -25 dB. GRDH is multi-looked (ENL ~4.4), so intensity ~ Gamma(4.4)
    # rather than the single-look exponential; single-look speckle is ~5.6 dB wide in dB and
    # would hide a -5 dB target below a 1e-6 Gaussian threshold.
    sea = rng.gamma(4.4, 1 / 4.4, (h, w)) * 10 ** (-25 / 10)
    vh = 10 * np.log10(sea)
    for r, c in ships:
        vh[r - 1 : r + 2, c - 1 : c + 2] = -5.0  # 3x3 px bright target
    return vh, list(ships)


def test_cfar_finds_ships_and_few_false_alarms():
    vh, ships = synthetic_scene()
    det = cfar_detect(vh, np.ones_like(vh, bool), CfarParams(pfa=1e-6))
    pairs = match_points(det[["row", "col"]].to_numpy(float), np.array(ships, float), tol_m=50)
    assert len(pairs) == len(ships)
    assert len(det) - len(pairs) <= 2


def test_cfar_ignores_invalid_region():
    vh, _ = synthetic_scene(ships=())
    valid = np.ones_like(vh, bool)
    vh[:, :300] = 10.0  # very bright "land"
    valid[:, :300] = False
    det = cfar_detect(vh, valid)
    assert (det.col < 300).sum() == 0


def test_windows_cores_partition_scene():
    h, w = 1300, 1000
    cover = np.zeros((h, w), int)
    for win in iter_windows(h, w, core=512, halo=64):
        cover[win.core_row0 : win.core_row1, win.core_col0 : win.core_col1] += 1
        assert win.row1 - win.row0 <= 640 and win.col1 - win.col0 <= 640
    assert (cover == 1).all()


def test_pad_to():
    assert pad_to(np.ones((100, 50, 3)), 640).shape == (640, 640, 3)


def test_encode_handles_nodata():
    vv = np.full((4, 4), -15.0)
    vh = np.full((4, 4), -22.0)
    vv[0, 0] = -32768  # fill value
    img = encode_3ch(vv, vh)
    assert img.dtype == np.uint8 and img.shape == (4, 4, 3)
    assert (img[0, 0] == 0).all() and img[1, 1].sum() > 0
    assert not valid_mask(vv, vh)[0, 0]


def test_hungarian_beats_greedy():
    # Greedy nearest-neighbour would match pred0->gt1 and leave gt0 unmatched.
    pred = np.array([[0, 10], [0, 25]], float)
    gt = np.array([[0, 0], [0, 12]], float)
    assert len(match_points(pred, gt, tol_m=150)) == 2


def test_metrics_and_low_confidence():
    gt = pd.DataFrame(
        {
            "scene_id": ["s"] * 3,
            "row": [0, 100, 200],
            "col": [0, 100, 200],
            "confidence": ["HIGH", "MEDIUM", "LOW"],
            "is_vessel": [True, True, True],
            "source": ["ais", "manual", "manual"],
            "vessel_length_m": [30.0, np.nan, np.nan],
            "distance_from_shore_km": [10.0, 1.0, 5.0],
        }
    )
    pred = pd.DataFrame({"scene_id": ["s"] * 3, "row": [1, 200, 400], "col": [1, 200, 400]})
    pm, gm = match_table(pred, gt)
    m = detection_metrics(pm, gm)
    # pred@LOW is neither TP nor FP; gt MEDIUM missed -> FN; pred@400 -> FP
    assert m["overall"] == {**m["overall"], "tp": 1, "fp": 1, "fn": 1}


def test_dark_flag():
    det = pd.DataFrame(
        {"scene_id": ["s"] * 3, "row": [0, 100, 300], "col": [0, 100, 300], "cls": ["vessel", "vessel", "fixed_structure"]}
    )
    ais = pd.DataFrame({"scene_id": ["s"], "row": [2], "col": [2]})
    out = flag_dark(det, ais)
    assert out.is_dark.tolist() == [False, True, False]
