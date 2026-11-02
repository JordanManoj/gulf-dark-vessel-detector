"""Gulf Sentinel-1 access via the Copernicus Data Space Ecosystem (DECISIONS.md D04).

  search : anonymous OData catalogue query (which IW GRD scenes cover an AOI?)
  fetch  : Sentinel Hub Process API -> σ⁰ (thermal-noise removed, calibrated, orthorectified)
           VV/VH in dB at 10 m in UTM, cropped to the AOI. Same physical quantity and pixel
           size as xView3, without downloading 1.7 GB scenes or installing SNAP.

Usage:
  python -m src.gulf_fetch search --aoi aoi/khalifa_port.geojson --start 2026-10-01 --end 2026-10-22
  python -m src.gulf_fetch fetch  --aoi aoi/khalifa_port.geojson --date 2026-10-21 --out data/raw/gulf
"""
import argparse
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from rasterio.warp import transform_bounds
from shapely.geometry import shape

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
# CDSE docs show both forms in different pages; try the first, fall back to the second on 404.
PROCESS_URLS = ["https://sh.dataspace.copernicus.eu/api/v1/process", "https://sh.dataspace.copernicus.eu/process/v1"]
UTM_40N = "EPSG:32640"  # both AOIs (54-60°E) fall in UTM zone 40N
MAX_PX = 2500
NODATA_DB = -9999.0

# dB conversion happens server-side; pixels outside the swath get NODATA_DB.
EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: ["VV", "VH", "dataMask"],
          output: {id: "default", bands: 2, sampleType: SampleType.FLOAT32}};
}
function db(x) { return 10 * Math.log(Math.max(x, 1e-6)) / Math.LN10; }
function evaluatePixel(s) {
  if (s.dataMask == 0) { return [%NODATA%, %NODATA%]; }
  return [db(s.VV), db(s.VH)];
}
""".replace("%NODATA%", str(NODATA_DB))


def load_aoi(path: str):
    gj = json.loads(Path(path).read_text())
    return shape(gj["features"][0]["geometry"])


def search(aoi_path: str, start: str, end: str, top: int = 50) -> list[dict]:
    geom = load_aoi(aoi_path)
    flt = (
        "Collection/Name eq 'SENTINEL-1' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
        "att/OData.CSC.StringAttribute/Value eq 'IW_GRDH_1S') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{geom.wkt}') and "
        f"ContentDate/Start gt {start}T00:00:00.000Z and ContentDate/Start lt {end}T23:59:59.000Z"
    )
    r = requests.get(ODATA_URL, params={"$filter": flt, "$top": top, "$orderby": "ContentDate/Start desc"}, timeout=60)
    r.raise_for_status()
    return [
        {"name": p["Name"], "start": p["ContentDate"]["Start"], "online": p.get("Online"), "gb": round(p["ContentLength"] / 1e9, 2)}
        for p in r.json()["value"]
    ]


def _token() -> str:
    load_dotenv()
    cid, secret = os.getenv("CDSE_CLIENT_ID"), os.getenv("CDSE_CLIENT_SECRET")
    if not cid or not secret:
        raise SystemExit("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in .env (see data/README.md).")
    r = requests.post(
        TOKEN_URL, data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret}, timeout=60
    )
    r.raise_for_status()
    return r.json()["access_token"]


def fetch(aoi_path: str, date: str, out_dir: str) -> Path:
    geom = load_aoi(aoi_path)
    bbox_utm = list(transform_bounds("EPSG:4326", UTM_40N, *geom.bounds))
    # Process API max output is 2500 x 2500 px; the UTM-projected box can come out slightly
    # larger than the lon/lat box, so clamp it around its centre to 25 km at 10 m.
    for lo, hi in [(0, 2), (1, 3)]:
        if bbox_utm[hi] - bbox_utm[lo] > MAX_PX * 10:
            mid = (bbox_utm[lo] + bbox_utm[hi]) / 2
            bbox_utm[lo], bbox_utm[hi] = mid - MAX_PX * 5, mid + MAX_PX * 5
    body = {
        "input": {
            "bounds": {"bbox": bbox_utm, "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/32640"}},
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {"from": f"{date}T00:00:00Z", "to": f"{date}T23:59:59Z"},
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "resolution": "HIGH",
                    },
                    "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True, "demInstance": "COPERNICUS_30"},
                }
            ],
        },
        "output": {"resx": 10, "resy": 10, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    headers = {"Authorization": f"Bearer {_token()}", "Accept": "image/tiff"}
    for url in PROCESS_URLS:
        r = requests.post(url, json=body, headers=headers, timeout=300)
        if r.status_code != 404:
            break
    if r.status_code != 200:
        raise SystemExit(f"Process API error {r.status_code}: {r.text[:500]}")

    out = Path(out_dir) / f"{Path(aoi_path).stem}_{date}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--aoi", required=True)
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--aoi", required=True)
    f.add_argument("--date", required=True)
    f.add_argument("--out", default="data/raw/gulf")
    a = ap.parse_args()
    if a.cmd == "search":
        for p in search(a.aoi, a.start, a.end):
            print(f"{p['start'][:19]}  {p['gb']:>5} GB  online={p['online']}  {p['name']}")
    else:
        print("saved", fetch(a.aoi, a.date, a.out))


if __name__ == "__main__":
    main()
