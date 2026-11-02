"""Fine-tune a pre-trained Ultralytics YOLO model on xView3 chips (DECISIONS.md D10, D11).

Usage (Kaggle / Colab GPU):
  python -m src.train --data data/processed/chips --model yolov8s.pt --name yolov8s
  python -m src.train --data data/processed/chips --model yolo26s.pt --name yolo26s
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_data_yaml(chips_dir: Path) -> Path:
    """Use the dataset's own data.yaml if the converter wrote one (e.g. 1-class OSSDD), else xView3's 2 classes."""
    if (chips_dir / "data.yaml").exists() and "names" in (chips_dir / "data.yaml").read_text():
        return chips_dir / "data.yaml"
    cfg = {"path": str(chips_dir.resolve()), "train": "images/train", "val": "images/val", "names": {0: "vessel", 1: "fixed_structure"}}
    p = chips_dir / "data.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="chips folder produced by src.make_chips")
    ap.add_argument("--model", required=True, help="pre-trained weights, e.g. yolov8s.pt or yolo26s.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on a subset first to check everything runs")
    ap.add_argument("--batch", type=int, help="override batch size (GTX 1650 4 GB: 8)")
    ap.add_argument("--workers", type=int, help="override data-loader processes (low RAM: 2)")
    a = ap.parse_args()

    from ultralytics import YOLO

    hp = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    for k in ("epochs", "batch", "workers"):
        if getattr(a, k) is not None:
            hp[k] = getattr(a, k)
    model = YOLO(a.model)  # COCO-pretrained weights (D11)
    model.train(data=str(write_data_yaml(Path(a.data))), project=a.project, name=a.name, fraction=a.fraction, **hp)


if __name__ == "__main__":
    main()
