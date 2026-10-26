"""Build data/catalog.csv from the UrbanTALES website's data feed.

Stage: S01 Data ingestion

The website lists its cases through two JSON endpoints, one per subset. Each
entry holds the case name, a few descriptors and the direct download link and
size of its six files. This script flattens both into one row per case and adds
split_group, the unit kept together when splitting train and test.

Usage: python pipeline/S01_data_ingestion/build_catalog.py [output_csv]
"""
import csv
import json
import os
import re
import socket
import sys
import urllib.request

socket.setdefaulttimeout(90)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = "https://urbantales.vercel.app/api/metadata_{}"
KINDS = {"ped": "{c}_ped.nc", "topo": "{c}_topo", "profile": "profile-{c}.csv",
         "ts": "{c}_ts.nc", "p3d": "{c}_p3d", "png": "{c}.png"}
PATTERNS = {"SM": "continuous", "CL": "clustered", "HR": "high-rise"}
COLUMNS = ["subset", "case", "split_group", "country", "city", "lon", "lat", "height_mode",
           "layout", "lambda_p", "height_pattern", "height_std_m", "wind_dir_deg",
           "density_class"] + [f"{k}_{s}" for k in KINDS for s in ("url", "mb")]


def split_group(subset, case):
    base = re.sub(r"_d\d+$", "", case)
    if subset == "idealized":
        return "ideal-" + base
    parts = base.split("-")
    return "city-" + "-".join(parts[:3] if parts[0] == "Val" else parts[:2]).upper()


def row(subset, entry):
    case = entry["Folder Name"]
    files = {f["File Name"]: f for f in entry["Files"]}
    wind = re.search(r"_d(\d+)$", case)
    out = {"subset": subset, "case": case, "split_group": split_group(subset, case),
           "wind_dir_deg": str(int(wind.group(1))) if wind else "",
           "density_class": entry["Plan Area Density"]}
    if subset == "idealized":
        code = re.match(r"([UV])([AS])\d+(?:-(\w\w)-(\d\d))?_d", case)
        out.update(height_mode="uniform" if code[1] == "U" else "variable",
                   layout=entry["Horizontal Configuration"].lower(),
                   lambda_p=str(entry["areaDensity"]),
                   height_pattern=PATTERNS.get(code[3], "uniform"),
                   height_std_m=str(int(code[4]) / 10 if code[4] else 0.0))
    else:
        out.update(country=entry["Country"], city=entry["City"],
                   lon=str(entry["coordinates"][0]), lat=str(entry["coordinates"][1]),
                   height_mode=entry["Config"].split("-")[1], layout="realistic")
    for kind, pattern in KINDS.items():
        f = files.get(pattern.format(c=case))  # the 10 cases without _dNN have no preview png
        if f:
            out[f"{kind}_url"] = f["Direct Download Link"]
            out[f"{kind}_mb"] = str(float(f["Size (MB)"]))
    return out


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data", "catalog.csv")
    rows = []
    for subset, endpoint in (("idealized", "ideal"), ("realistic", "rea")):
        with urllib.request.urlopen(API.format(endpoint)) as response:
            rows += [row(subset, entry) for entry in json.load(response)]
    with open(target, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} cases written to {target}")


if __name__ == "__main__":
    main()
