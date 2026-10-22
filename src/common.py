"""Shared paths, ArcPy helpers and plotting helpers for the Esri MOOC rebuild.

Every section script imports this module. Nothing here spends ArcGIS Online
credits: all inputs are the local exercise geodatabases copied into ``data/``.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
RES = ROOT / "results"
LOGS = OUT / "logs"

for _d in (OUT, FIG, RES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# Course data (copies of C:\EsriTraining\...; never committed)
DE_GDB = DATA / "DataEngineering_and_Visualization" / "DataEngineering_and_Visualization.gdb"
DE_APRX = DATA / "DataEngineering_and_Visualization" / "DataEngineering_and_Visualization.aprx"
PRED_GDB = DATA / "Prediction" / "Prediction_Data.gdb"
SUIT_GDB = DATA / "Suitability" / "Data.gdb"
SUIT_COURSE_GDB = DATA / "Suitability" / "Headquarters_Siting.gdb"
SUIT_SAM = DATA / "Suitability" / "Suitability.sam"
PAT_GDB = DATA / "PatternDetection_SpaceTime" / "Data.gdb"

# Esri hot spot / cluster colours (Gi_Bin -3..3, LISA categories)
GI_COLORS = {
    -3: "#2c7bb6", -2: "#6fa8d2", -1: "#b9d3e8", 0: "#f2f2f2",
    1: "#f4c0a7", 2: "#e8775d", 3: "#d7191c",
}
GI_LABELS = {
    -3: "Cold spot, 99%", -2: "Cold spot, 95%", -1: "Cold spot, 90%", 0: "Not significant",
    1: "Hot spot, 90%", 2: "Hot spot, 95%", 3: "Hot spot, 99%",
}
LISA_COLORS = {"HH": "#d7191c", "LL": "#2c7bb6", "HL": "#f4a582", "LH": "#92c5de", "": "#f2f2f2"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def import_arcpy():
    """Import arcpy once (takes about a minute) and set sane environments."""
    t0 = time.time()
    import arcpy  # noqa: WPS433

    arcpy.env.overwriteOutput = True
    arcpy.env.addOutputsToMap = False
    # Fixed seed for tools that draw random numbers (Forest validation splits, permutations)
    arcpy.env.randomGenerator = "20261022 ACM599"
    log(f"arcpy {arcpy.GetInstallInfo()['Version']} imported in {time.time() - t0:.0f} s")
    return arcpy


def ensure_gdb(name: str) -> str:
    """Return outputs/<name>.gdb, creating it if needed."""
    import arcpy

    path = OUT / f"{name}.gdb"
    if not arcpy.Exists(str(path)):
        arcpy.management.CreateFileGDB(str(OUT), f"{name}.gdb")
    # Give each section its own scratch workspace so sections can run side by side
    # (several tools write temporary tables to arcpy.env.scratchGDB).
    scratch = OUT / f"scratch_{name}"
    scratch.mkdir(exist_ok=True)
    arcpy.env.scratchWorkspace = str(scratch)  # arcpy.env.scratchGDB becomes <scratch>/scratch.gdb
    return str(path)


def save_messages(name: str) -> str:
    """Write arcpy.GetMessages() for the last tool to outputs/logs/<name>.txt and return it."""
    import arcpy

    text = arcpy.GetMessages()
    (LOGS / f"{name}.txt").write_text(text, encoding="utf-8")
    return text


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    log(f"wrote {path.relative_to(ROOT)}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        fieldnames += [k for k in r if k not in fieldnames]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")


def table_to_df(table: str, fields: list[str] | None = None, where: str | None = None):
    """Read a table or feature class attribute table into a pandas DataFrame."""
    import arcpy
    import pandas as pd

    if fields is None:
        fields = [f.name for f in arcpy.ListFields(table) if f.type not in ("Geometry", "Blob", "Raster")]
    rows = list(arcpy.da.SearchCursor(table, fields, where))
    return pd.DataFrame.from_records(rows, columns=fields)


def count(table: str, where: str | None = None) -> int:
    import arcpy

    if where:
        lyr = arcpy.management.MakeTableView(table, "cnt_view", where)
        n = int(arcpy.management.GetCount(lyr)[0])
        arcpy.management.Delete(lyr)
        return n
    return int(arcpy.management.GetCount(table)[0])


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    })
    return plt


def read_polygons(fc: str, fields: list[str], where: str | None = None, spatial_reference=None):
    """Return (matplotlib Paths, DataFrame of fields) for a polygon feature class.

    Multipart polygons and holes are kept as compound paths.
    """
    import arcpy
    import pandas as pd
    from matplotlib.path import Path as MplPath

    paths, recs = [], []
    with arcpy.da.SearchCursor(fc, ["SHAPE@JSON"] + fields, where, spatial_reference=spatial_reference) as cur:
        for row in cur:
            if row[0] is None:
                continue
            rings = json.loads(row[0]).get("rings", [])
            verts, codes = [], []
            for ring in rings:
                ring = np.asarray(ring, dtype=float)[:, :2]
                if len(ring) < 3:
                    continue
                verts.append(ring)
                c = np.full(len(ring), MplPath.LINETO)
                c[0] = MplPath.MOVETO
                c[-1] = MplPath.CLOSEPOLY
                codes.append(c)
            if not verts:
                continue
            paths.append(MplPath(np.vstack(verts), np.concatenate(codes)))
            recs.append(row[1:])
    return paths, pd.DataFrame.from_records(recs, columns=fields)


def draw_paths(ax, paths, facecolors, edgecolor="#555555", linewidth=0.15):
    from matplotlib.collections import PathCollection

    pc = PathCollection(paths, facecolors=facecolors, edgecolors=edgecolor, linewidths=linewidth)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()
    return pc


def std_dev_breaks(values: np.ndarray, interval: float = 1.0) -> list[float]:
    """Inner class breaks like ArcGIS Pro's standard-deviation classification.

    With a 1 SD interval the classes are < -2.5 SD, -2.5..-1.5, ..., 1.5..2.5, > 2.5 SD
    (Pro caps the tails at +/-2.5 SD rather than extending to the data range).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    mean, sd = v.mean(), v.std(ddof=1)
    ks = np.arange(interval / 2, 2.5 + 1e-9, interval)
    ks = np.concatenate([-ks[::-1], ks])
    return [mean + k * sd for k in ks]


def parse_number(pattern: str, text: str, flags=re.I) -> float | None:
    m = re.search(pattern, text, flags)
    return float(m.group(1).replace(",", "")) if m else None
