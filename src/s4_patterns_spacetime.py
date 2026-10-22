"""Section 4: pattern detection and space-time pattern mining on SNAP participation.

Course steps:
  1. Hot Spot Analysis (Getis-Ord Gi*) on county SNAP participation rate, first with the
     tool's default fixed distance band, then with polygon contiguity (edges and corners).
  2. Cluster and Outlier Analysis (Anselin Local Moran's I), contiguity, 999 permutations.
  3. Create Space Time Cube From Defined Locations: 3,105 counties x 10 yearly steps
     (2010-2019) from the related SNAP_County_Rates2010_2019 table.
  4. Emerging Hot Spot Analysis and Local Outlier Analysis on the cube.

FDR: applied in steps 1 and 2 (the tools offer it). The space-time tools (EHSA, Local
Outlier Analysis) have no FDR option in ArcGIS Pro 3.7, so none is applied there.

Outputs: outputs/s4_patterns.gdb, outputs/s4_snap_cube.nc, results/s4_*.csv|json, figures/s4_*.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

COUNTIES = str(C.PAT_GDB / "US_Counties")
STATES = str(C.PAT_GDB / "US_States")
SNAP_TABLE = str(C.PAT_GDB / "SNAP_County_Rates2010_2019")
RATE_FIELD = "SNAPRate_1"  # 2019 rate on the county polygons (POPESTIMATE2019 denominator)

EHSA_COLORS = {
    "New Hot Spot": "#ff9e9e", "Consecutive Hot Spot": "#ff5c5c", "Intensifying Hot Spot": "#d7191c", "Persistent Hot Spot": "#a50026",
    "Diminishing Hot Spot": "#f4a582", "Sporadic Hot Spot": "#fddbc7", "Oscillating Hot Spot": "#e6a5a5", "Historical Hot Spot": "#c9a9a9",
    "New Cold Spot": "#9ecae1", "Consecutive Cold Spot": "#6baed6", "Intensifying Cold Spot": "#2171b5", "Persistent Cold Spot": "#08306b",
    "Diminishing Cold Spot": "#9fc5e8", "Sporadic Cold Spot": "#d1e5f0", "Oscillating Cold Spot": "#a6bddb", "Historical Cold Spot": "#9fb3c8",
    "No Pattern Detected": "#f2f2f2",
}
LOA_COLORS = {
    "Only High-High Cluster": "#d7191c", "Only High-Low Outlier": "#f4a582", "Only Low-High Outlier": "#92c5de", "Only Low-Low Cluster": "#2c7bb6",
    "Multiple Types": "#9e9e9e", "Never Significant": "#f2f2f2",
}


def neighbor_stats(arcpy, counties, text, band=None):
    """Neighbour counts for the hot spot run.

    ArcGIS Pro 3.7 prints only the default distance band (WARNING 000853), not the
    neighbour summary, so the counts are computed here from the same inputs the tool
    uses: polygon centroids within the band (fixed distance) or Polygon Neighbors
    (contiguity, edges and corners).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    out = {"warnings": [ln.strip() for ln in text.splitlines() if "WARNING" in ln]}
    if band is None:
        band = C.parse_number(r"threshold was ([\d.]+) meters", text)
    if band:
        out["distance_band_m"] = band
        xy = np.array([r[0] for r in arcpy.da.SearchCursor(counties, ["SHAPE@XY"])])
        tree = cKDTree(xy)
        counts = np.array([len(tree.query_ball_point(p, band)) - 1 for p in xy])
        out["conceptualization"] = "FIXED_DISTANCE_BAND"
    else:
        tbl = str(Path(arcpy.env.scratchGDB) / "poly_neighbors")
        arcpy.analysis.PolygonNeighbors(counties, tbl, "OBJECTID_1", "NO_AREA_OVERLAP", "BOTH_SIDES")
        df = C.table_to_df(tbl, ["src_OBJECTID_1"])
        n = C.count(counties)
        counts = df["src_OBJECTID_1"].value_counts().reindex(range(1, n + 1), fill_value=0).values
        out["conceptualization"] = "CONTIGUITY_EDGES_CORNERS"
    out.update({"min_neighbors": int(counts.min()), "mean_neighbors": round(float(counts.mean()), 2),
                "max_neighbors": int(counts.max()), "features_without_neighbors": int((counts == 0).sum())})
    return out


def main():
    arcpy = C.import_arcpy()
    plt = C.setup_matplotlib()
    gdb = C.ensure_gdb("s4_patterns")
    arcpy.env.workspace = gdb
    summary = {}

    counties = arcpy.management.Copy(COUNTIES, f"{gdb}/Counties")[0]
    n = C.count(counties)
    C.log(f"{n} counties; rate field {RATE_FIELD}")

    # ---- 1. Hot Spot Analysis, default fixed distance band ----------------------
    hs_fixed = f"{gdb}/HotSpots_fixed"
    arcpy.stats.HotSpots(counties, RATE_FIELD, hs_fixed, "FIXED_DISTANCE_BAND", "EUCLIDEAN_DISTANCE", "NONE",
                         None, None, None, "APPLY_FDR")
    txt = C.save_messages("s4_hotspots_fixed")
    summary["hotspot_fixed"] = neighbor_stats(arcpy, counties, txt)
    C.log(f"Fixed distance band: {summary['hotspot_fixed']}")

    # ---- 1b. Hot Spot Analysis, contiguity ------------------------------------------
    hs_cont = f"{gdb}/HotSpots_contiguity"
    arcpy.stats.HotSpots(counties, RATE_FIELD, hs_cont, "CONTIGUITY_EDGES_CORNERS", "EUCLIDEAN_DISTANCE", "NONE",
                         None, None, None, "APPLY_FDR")
    txt = C.save_messages("s4_hotspots_contiguity")
    summary["hotspot_contiguity"] = neighbor_stats(arcpy, counties, txt, band=0)

    # ---- 2. Cluster and Outlier Analysis, contiguity -----------------------------
    lisa = f"{gdb}/ClustersOutliers_contiguity"
    arcpy.stats.ClustersOutliers(counties, RATE_FIELD, lisa, "CONTIGUITY_EDGES_CORNERS", "EUCLIDEAN_DISTANCE", "NONE",
                                 None, None, "APPLY_FDR", 999)
    txt = C.save_messages("s4_clusters_outliers")
    summary["clusters_outliers"] = {"conceptualization": "CONTIGUITY_EDGES_CORNERS", "permutations": 999, "fdr": True,
                                    "warnings": [ln.strip() for ln in txt.splitlines() if "WARNING" in ln]}

    # ---- 3. Space-time cube from defined locations -------------------------------
    cube = str(C.OUT / "s4_snap_cube.nc")
    arcpy.stpm.CreateSpaceTimeCubeDefinedLocations(counties, cube, "CountyID", "NO_TEMPORAL_AGGREGATION", "TimeofYear",
                                                   "1 Years", "END_TIME", None, [["SNAPRate", "ZEROS"]], None,
                                                   SNAP_TABLE, "CountyID")
    txt = C.save_messages("s4_create_cube")
    import netCDF4

    with netCDF4.Dataset(cube) as nc:
        cube_vars = [v for v in nc.variables if v.upper().startswith("SNAPRATE")]
        summary["cube"] = {"variables": list(nc.variables), "dimensions": {k: len(v) for k, v in nc.dimensions.items()}}
    analysis_var = cube_vars[0]
    C.log(f"Cube variable: {analysis_var}; dims {summary['cube']['dimensions']}")
    summary["cube"]["analysis_variable"] = analysis_var
    summary["cube"]["messages_excerpt"] = [ln for ln in txt.splitlines() if any(k in ln for k in ("Locations", "Time", "Trend", "Bins", "%"))][:30]

    # ---- 4. Emerging Hot Spot Analysis and Local Outlier Analysis ---------------
    ehsa = f"{gdb}/EmergingHotSpots"
    arcpy.stpm.EmergingHotSpotAnalysis(cube, analysis_var, ehsa, None, 1, None, "CONTIGUITY_EDGES_CORNERS")
    txt = C.save_messages("s4_ehsa")
    summary["ehsa"] = {"neighborhood_time_step": 1, "conceptualization": "CONTIGUITY_EDGES_CORNERS", "fdr": "not offered by the tool",
                       "messages_excerpt": [ln for ln in txt.splitlines() if "%" in ln or "Trend" in ln or "trend" in ln][:20]}
    loa = f"{gdb}/LocalOutliers"
    arcpy.stpm.LocalOutlierAnalysis(cube, analysis_var, loa, None, 1, 499, None, "CONTIGUITY_EDGES_CORNERS")
    txt = C.save_messages("s4_loa")
    summary["loa"] = {"permutations": 499, "neighborhood_time_step": 1, "conceptualization": "CONTIGUITY_EDGES_CORNERS", "fdr": "not offered by the tool"}

    # ---- Tabulate --------------------------------------------------------------------
    df_fixed = C.table_to_df(hs_fixed, ["SOURCE_ID", "GiZScore", "GiPValue", "Gi_Bin"]).rename(columns={"Gi_Bin": "bin_fixed", "GiZScore": "z_fixed"})
    df_cont = C.table_to_df(hs_cont, ["SOURCE_ID", "GiZScore", "GiPValue", "Gi_Bin"]).rename(columns={"Gi_Bin": "bin_cont", "GiZScore": "z_cont"})
    both = df_fixed.merge(df_cont[["SOURCE_ID", "bin_cont", "z_cont"]], on="SOURCE_ID")
    both["class_fixed"] = both["bin_fixed"].map(C.GI_LABELS)
    both["class_cont"] = both["bin_cont"].map(C.GI_LABELS)
    gi_rows = []
    for b in range(-3, 4):
        gi_rows.append({"Gi_Bin": b, "class": C.GI_LABELS[b], "fixed_distance": int((both["bin_fixed"] == b).sum()),
                        "contiguity": int((both["bin_cont"] == b).sum())})
    C.write_csv(gi_rows, C.RES / "s4_hotspot_counts.csv")
    xtab = pd.crosstab(both["bin_fixed"].map(lambda b: "hot" if b > 0 else "cold" if b < 0 else "ns"),
                       both["bin_cont"].map(lambda b: "hot" if b > 0 else "cold" if b < 0 else "ns"))
    xtab.to_csv(C.RES / "s4_hotspot_fixed_vs_contiguity.csv")
    agree = float((both["bin_fixed"].clip(-1, 1) == both["bin_cont"].clip(-1, 1)).mean())
    summary["hotspot_comparison"] = {"share_same_class_hot_cold_ns": round(agree, 4), "z_correlation": round(float(both["z_fixed"].corr(both["z_cont"])), 4),
                                     "hot_fixed": int((both["bin_fixed"] > 0).sum()), "hot_contiguity": int((both["bin_cont"] > 0).sum()),
                                     "cold_fixed": int((both["bin_fixed"] < 0).sum()), "cold_contiguity": int((both["bin_cont"] < 0).sum())}

    df_lisa = C.table_to_df(lisa, ["SOURCE_ID", "LMiIndex", "LMiZScore", "LMiPValue", "COType"])
    lisa_counts = df_lisa["COType"].fillna("").replace("", "Not significant").value_counts()
    C.write_csv([{"COType": k, "counties": int(v)} for k, v in lisa_counts.items()], C.RES / "s4_lisa_counts.csv")

    df_ehsa = C.table_to_df(ehsa)
    pat_col = next(c for c in df_ehsa.columns if c.upper() == "PATTERN")
    ehsa_counts = df_ehsa[pat_col].value_counts()
    C.write_csv([{"pattern": k, "counties": int(v)} for k, v in ehsa_counts.items()], C.RES / "s4_ehsa_counts.csv")

    df_loa = C.table_to_df(loa)
    cat_col = next(c for c in df_loa.columns if c.upper() == "CATEGORY")
    loa_counts = df_loa[cat_col].value_counts()
    C.write_csv([{"category": k, "counties": int(v)} for k, v in loa_counts.items()], C.RES / "s4_loa_counts.csv")
    summary["counts"] = {"gi": gi_rows, "lisa": {k: int(v) for k, v in lisa_counts.items()},
                         "ehsa": {k: int(v) for k, v in ehsa_counts.items()}, "loa": {k: int(v) for k, v in loa_counts.items()}}
    summary["ehsa_fields"] = list(df_ehsa.columns)
    summary["loa_fields"] = list(df_loa.columns)
    C.write_json(summary, C.RES / "s4_patterns.json")

    # ---- Maps --------------------------------------------------------------------------
    from matplotlib.patches import Patch

    sr = arcpy.Describe(counties).spatialReference
    state_paths, _ = C.read_polygons(STATES, ["STATE_ABBR"], spatial_reference=sr)
    paths_f, dff = C.read_polygons(hs_fixed, ["Gi_Bin"])
    paths_c, dfc = C.read_polygons(hs_cont, ["Gi_Bin"])
    paths_l, dfl = C.read_polygons(lisa, ["COType"])
    paths_e, dfe = C.read_polygons(ehsa, [pat_col])
    paths_o, dfo = C.read_polygons(loa, [cat_col])
    paths_r, dfr = C.read_polygons(counties, [RATE_FIELD])

    def legend(ax, items, title):
        ax.legend(handles=[Patch(facecolor=c, edgecolor="#999999", label=l) for l, c in items], loc="lower left", fontsize=6, frameon=False, title=title, title_fontsize=7)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    import numpy as np

    v = dfr[RATE_FIELD].astype(float).values
    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(0, np.nanpercentile(v, 99))
    C.draw_paths(ax, paths_r, [cmap(norm(x)) for x in v], edgecolor="white", linewidth=0.1)
    C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="SNAP participation rate, 2019")
    ax.set_title("SNAP participation rate (share of population), 2019")
    for ax, paths, df, ttl, d in [(axes[0, 1], paths_f, dff, f"Gi* hot spots, fixed distance band ({summary['hotspot_fixed'].get('distance_band_m', 0) / 1000:.0f} km), FDR", summary["hotspot_fixed"]),
                                  (axes[1, 0], paths_c, dfc, "Gi* hot spots, polygon contiguity (edges and corners), FDR", summary["hotspot_contiguity"])]:
        C.draw_paths(ax, paths, [C.GI_COLORS[int(b)] for b in df["Gi_Bin"]], edgecolor="white", linewidth=0.1)
        C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
        ax.set_title(ttl)
        legend(ax, [(C.GI_LABELS[b], C.GI_COLORS[b]) for b in (3, 2, 1, 0, -1, -2, -3)], f"mean neighbours {d.get('mean_neighbors', '?')}")
    ax = axes[1, 1]
    C.draw_paths(ax, paths_l, [C.LISA_COLORS.get(t or "", "#f2f2f2") for t in dfl["COType"]], edgecolor="white", linewidth=0.1)
    C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
    ax.set_title("Local Moran's I clusters and outliers, contiguity, 999 permutations, FDR")
    legend(ax, [("High-High cluster", C.LISA_COLORS["HH"]), ("High-Low outlier", C.LISA_COLORS["HL"]), ("Low-High outlier", C.LISA_COLORS["LH"]),
                ("Low-Low cluster", C.LISA_COLORS["LL"]), ("Not significant", C.LISA_COLORS[""])], "Cluster/outlier type")
    fig.tight_layout()
    fig.savefig(C.FIG / "s4_hotspots_lisa.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    C.draw_paths(ax, paths_e, [EHSA_COLORS.get(p, "#cccccc") for p in dfe[pat_col]], edgecolor="white", linewidth=0.1)
    C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
    ax.set_title("Emerging Hot Spot Analysis, SNAP rate 2010-2019 (contiguity, 1 time step)")
    present = [p for p in EHSA_COLORS if p in set(dfe[pat_col])]
    legend(ax, [(f"{p} ({int(ehsa_counts.get(p, 0))})", EHSA_COLORS[p]) for p in present], "Pattern (counties)")
    ax = axes[1]
    C.draw_paths(ax, paths_o, [LOA_COLORS.get(p, "#cccccc") for p in dfo[cat_col]], edgecolor="white", linewidth=0.1)
    C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
    ax.set_title("Local Outlier Analysis, SNAP rate 2010-2019 (contiguity, 499 permutations)")
    present = [p for p in LOA_COLORS if p in set(dfo[cat_col])]
    legend(ax, [(f"{p} ({int(loa_counts.get(p, 0))})", LOA_COLORS[p]) for p in present], "Category (counties)")
    fig.tight_layout()
    fig.savefig(C.FIG / "s4_spacetime.png")
    plt.close(fig)
    C.log("Section 4 done")


if __name__ == "__main__":
    main()
