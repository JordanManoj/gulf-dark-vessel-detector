"""Land polygons for a Gulf AOI from OpenStreetMap coastlines (DECISIONS.md D17).

OSM tags the shoreline as `natural=coastline` ways, drawn so that **land is on the left** of the
way direction. We fetch the ways inside the AOI from the Overpass API, split the AOI box into
faces along them, and keep the faces that lie on the left of an adjacent coastline segment.

Usage:
  python -m src.land_mask --aoi aoi/khalifa_port.geojson --out data/interim/khalifa_port_land.geojson
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
from shapely import STRtree, points
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import polygonize, unary_union

# Public Overpass instances are often overloaded (502/504); try mirrors in turn.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def _overpass(query: str) -> dict:
    last = None
    for attempt in range(2):
        for url in OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, timeout=180, headers={"User-Agent": "dark-vessel-detector/0.1"})
                if r.status_code == 200:
                    return r.json()
                last = f"{url}: HTTP {r.status_code}"
            except requests.RequestException as e:
                last = f"{url}: {e}"
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"All Overpass mirrors failed, last error: {last}")


def fetch_coastlines(bounds, margin=0.05) -> list[LineString]:
    w, s, e, n = bounds
    q = f"""[out:json][timeout:120];
way["natural"="coastline"]({s - margin},{w - margin},{n + margin},{e + margin});
out geom;"""
    lines = []
    for el in _overpass(q)["elements"]:
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) >= 2:
            lines.append(LineString(coords))
    return lines


def land_faces(aoi_box, coast: list[LineString], eps: float = 1e-5):
    """Split the AOI by the coastlines and classify each face as land/sea with the left-hand rule."""
    clipped = [c.intersection(aoi_box) for c in coast]
    segs = []
    for c in clipped:
        for part in getattr(c, "geoms", [c]):
            if isinstance(part, LineString) and part.length > 0:
                segs.append(part)
    if not segs:
        return []
    faces = list(polygonize(unary_union(segs + [aoi_box.exterior])))

    # Probe points just left (land side) and right (sea side) of every coastline segment, then
    # let each face count land-vs-sea votes. A spatial index keeps this fast for detailed coasts.
    left_pts, right_pts = [], []
    for line in segs:
        xy = np.asarray(line.coords)
        d = np.diff(xy, axis=0)
        n = np.hypot(d[:, 0], d[:, 1])
        ok = n > 0
        mid = (xy[:-1] + xy[1:])[ok] / 2
        normal = np.stack([-d[ok, 1], d[ok, 0]], axis=1) / n[ok, None] * eps
        left_pts += list(points(mid + normal))
        right_pts += list(points(mid - normal))

    tree = STRtree(faces)
    votes = np.zeros(len(faces))
    for pts, sign in [(left_pts, 1), (right_pts, -1)]:
        pt_idx, face_idx = tree.query(pts, predicate="within")
        np.add.at(votes, face_idx, sign)
    return [f for f, v in zip(faces, votes) if v > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    aoi = shape(json.loads(Path(a.aoi).read_text())["features"][0]["geometry"])
    aoi_box = box(*aoi.bounds)
    coast = fetch_coastlines(aoi.bounds)
    land = land_faces(aoi_box, coast)
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"source": "OpenStreetMap natural=coastline (ODbL)"}, "geometry": mapping(p)} for p in land],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc))
    area = sum(p.area for p in land) / aoi_box.area
    print(f"{len(coast)} coastline ways -> {len(land)} land polygons, {area:.1%} of AOI is land -> {out}")


if __name__ == "__main__":
    main()
