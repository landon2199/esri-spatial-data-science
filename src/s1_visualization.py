"""Section 1, Exercise 3: exploring voter turnout 2020 visually.

Uses the course's enriched layer ``CountyElections2020`` (3,112 counties, Esri
demographic enrichment already attached by the course; no Enrich run here).

Figures (figures/):
  s1_turnout_hist.png        histogram of 2020 turnout with mean, median and a normal curve
  s1_turnout_choropleth.png  standard-deviation classification, diverging ramp (matplotlib)
  s1_turnout_layout.png      ArcGIS Pro layout exported with arcpy.mp (same classification)
  s1_state_boxplot.png       turnout by state
  s1_scatter_matrix.png      turnout vs median age, per-capita income, bachelor's degree share
Tables (results/): s1_skewness.csv, s1_scatter_r2.csv, s1_state_turnout.csv
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

FC = str(C.DE_GDB / "CountyElections2020")
STATES = str(C.DE_GDB / "US_States")
VARS = {
    "voter_turnout_2020": "Voter turnout 2020",
    "gender_MEDAGE_CY": "Median age",
    "householdincome_PCI_CY": "Per-capita income ($)",
    "educationalattainment_BACHDEG_CY_P": "Bachelor's degree (% of 25+)",
    "educationalattainment_NOHS_CY_P": "No high school diploma (%)",
    "households_ACSHHBPOV_P": "Households below poverty (%)",
    "pop_sqmi": "Population per sq mi",
    "pctdiff_dem_vs_gop_2020": "Dem minus GOP share 2020",
}


def skewness_table(df):
    from scipy import stats

    rows = []
    for col, label in VARS.items():
        v = df[col].dropna().astype(float)
        rows.append({"variable": col, "label": label, "n": len(v), "mean": round(v.mean(), 4), "median": round(v.median(), 4),
                     "std": round(v.std(ddof=1), 4), "skewness": round(float(stats.skew(v)), 3),
                     "kurtosis_excess": round(float(stats.kurtosis(v)), 3), "min": round(v.min(), 4), "max": round(v.max(), 4)})
    return rows


def histogram(plt, df):
    from scipy import stats

    v = df["voter_turnout_2020"].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(v, bins=40, color="#7a9cc6", edgecolor="white", density=True)
    xs = np.linspace(v.min(), v.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, v.mean(), v.std(ddof=1)), color="#333333", lw=1.2, label="Normal curve")
    ax.axvline(v.mean(), color="#d7191c", lw=1.2, label=f"Mean {v.mean():.3f}")
    ax.axvline(v.median(), color="#2c7bb6", lw=1.2, ls="--", label=f"Median {v.median():.3f}")
    ax.set_xlabel("Voter turnout 2020 (votes / citizen voting-age population)")
    ax.set_ylabel("Density")
    ax.set_title(f"County voter turnout, 2020 (n = {len(v):,}, skewness {stats.skew(v):+.2f})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG / "s1_turnout_hist.png")
    plt.close(fig)


def choropleth(plt, paths, df, state_paths):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    v = df["voter_turnout_2020"].astype(float).values
    inner = C.std_dev_breaks(v, 1.0)
    cmap = plt.get_cmap("RdBu", len(inner) + 1)
    colors = [cmap(i) for i in range(len(inner) + 1)]
    cls = np.digitize(v, inner)
    face = [colors[i] for i in cls]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    C.draw_paths(ax, paths, face, edgecolor="#ffffff", linewidth=0.1)
    C.draw_paths(ax, state_paths, "none", edgecolor="#444444", linewidth=0.4)
    mean, sd = v.mean(), v.std(ddof=1)
    labels = []
    edges = [-np.inf] + inner + [np.inf]
    for i in range(len(edges) - 1):
        lo = (edges[i] - mean) / sd
        hi = (edges[i + 1] - mean) / sd
        if np.isinf(lo):
            labels.append(f"< {hi:+.1f} SD")
        elif np.isinf(hi):
            labels.append(f"> {lo:+.1f} SD")
        else:
            labels.append(f"{lo:+.1f} to {hi:+.1f} SD")
    handles = [Patch(facecolor=colors[i], edgecolor="#999999", label=labels[i]) for i in range(len(labels))]
    # Clip to the contiguous states (the county layer also holds Hawaii)
    xs = np.concatenate([p.vertices[:, 0] for p in state_paths])
    ys = np.concatenate([p.vertices[:, 1] for p in state_paths])
    ax.set_xlim(xs.min() - 5e4, xs.max() + 5e4)
    ax.set_ylim(ys.min() - 5e4, ys.max() + 5e4)
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=False, title=f"Turnout 2020\nmean {mean:.3f}, SD {sd:.3f}", title_fontsize=7)
    ax.set_title("Voter turnout 2020 by county, standard-deviation classes (1 SD, tails capped at ±2.5 SD)")
    fig.tight_layout()
    fig.savefig(C.FIG / "s1_turnout_choropleth.png")
    plt.close(fig)
    return inner


def state_boxplot(plt, df):
    g = df.groupby("state_abbr")["voter_turnout_2020"]
    order = g.median().sort_values().index.tolist()
    data = [df.loc[df["state_abbr"] == s, "voter_turnout_2020"].dropna().values for s in order]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.boxplot(data, tick_labels=order, showfliers=True, flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
               medianprops={"color": "#d7191c"}, boxprops={"color": "#444444"}, whiskerprops={"color": "#444444"})
    ax.set_ylabel("Voter turnout 2020")
    ax.set_title("County voter turnout 2020 by state (states ordered by median)")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "s1_state_boxplot.png")
    plt.close(fig)
    rows = [{"state": s, "counties": int(g.count()[s]), "median": round(float(g.median()[s]), 4),
             "mean": round(float(g.mean()[s]), 4), "min": round(float(g.min()[s]), 4), "max": round(float(g.max()[s]), 4)} for s in order]
    return rows


def scatter_matrix(plt, df):
    cols = ["voter_turnout_2020", "gender_MEDAGE_CY", "householdincome_PCI_CY", "educationalattainment_BACHDEG_CY_P"]
    labels = ["Turnout 2020", "Median age", "Per-capita income", "Bachelor's %"]
    d = df[cols].dropna().astype(float)
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(9, 9))
    r2 = []
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(d[cols[i]], bins=30, color="#7a9cc6", edgecolor="white")
            elif i > j:
                x, y = d[cols[j]].values, d[cols[i]].values
                ax.scatter(x, y, s=3, alpha=0.3, color="#2c5d8f", rasterized=True)
                b1, b0 = np.polyfit(x, y, 1)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, b0 + b1 * xs, color="#d7191c", lw=1)
                rr = float(np.corrcoef(x, y)[0, 1] ** 2)
                ax.text(0.03, 0.92, f"R² = {rr:.3f}", transform=ax.transAxes, fontsize=8)
                if i == 0 or j == 0 or True:
                    r2.append({"x": cols[j], "y": cols[i], "r2": round(rr, 4), "slope_y_on_x": round(float(b1), 6), "n": len(d)})
            else:
                rr = float(np.corrcoef(d[cols[j]], d[cols[i]])[0, 1] ** 2)
                ax.text(0.5, 0.5, f"R² = {rr:.3f}", ha="center", va="center", fontsize=11, transform=ax.transAxes)
                ax.set_axis_off()
            if i == n - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])
            ax.tick_params(labelsize=7)
    fig.suptitle("Scatterplot matrix: voter turnout 2020 vs demographic variables (3,112 counties)")
    fig.tight_layout()
    fig.savefig(C.FIG / "s1_scatter_matrix.png")
    plt.close(fig)
    return [r for r in r2 if r["y"] == "voter_turnout_2020" or r["x"] == "voter_turnout_2020"]


def pro_layout(arcpy, breaks):
    """Build a real ArcGIS Pro map + layout headlessly with arcpy.mp and export it."""
    aprx_path = C.OUT / "s1_visualization.aprx"
    shutil.copy(C.DE_APRX, aprx_path)
    aprx = arcpy.mp.ArcGISProject(str(aprx_path))
    m = aprx.createMap("Turnout 2020", "MAP")
    for bl in m.listLayers():  # drop the default online basemap so the export needs no network
        if bl.isBasemapLayer:
            m.removeLayer(bl)
    m.spatialReference = arcpy.Describe(FC).spatialReference
    lyr = m.addDataFromPath(FC)
    lyr.name = "Voter turnout 2020 (votes / CVAP)"
    states_lyr = m.addDataFromPath(STATES)
    states_lyr.name = "States"
    sym = lyr.symbology
    sym.updateRenderer("GraduatedColorsRenderer")
    sym.renderer.classificationField = "voter_turnout_2020"
    sym.renderer.classificationMethod = "StandardDeviation"
    sym.renderer.standardDeviationInterval = 1
    ramps = aprx.listColorRamps("Red-Blue (Continuous)") or aprx.listColorRamps("Blue-Red*") or aprx.listColorRamps()
    sym.renderer.colorRamp = ramps[0]
    lyr.symbology = sym
    # Thin county outlines
    sym = lyr.symbology
    for brk in sym.renderer.classBreaks:
        brk.symbol.outlineColor = {"RGB": [255, 255, 255, 60]}
        brk.symbol.outlineWidth = 0.1
    lyr.symbology = sym
    # States as hollow outlines underneath the counties
    m.moveLayer(states_lyr, lyr, "BEFORE")
    ssym = states_lyr.symbology
    ssym.renderer.symbol.color = {"RGB": [0, 0, 0, 0]}
    ssym.renderer.symbol.outlineColor = {"RGB": [60, 60, 60, 100]}
    ssym.renderer.symbol.outlineWidth = 0.6
    states_lyr.symbology = ssym
    lyt = aprx.createLayout(11, 8.5, "INCH", "Turnout 2020 layout")
    mf = lyt.createMapFrame(arcpy.Polygon(arcpy.Array([arcpy.Point(0.3, 0.9), arcpy.Point(0.3, 7.9),
                                                       arcpy.Point(10.7, 7.9), arcpy.Point(10.7, 0.9)])), m, "Map Frame")
    # Zoom to the contiguous states (the county layer also holds Hawaii, which would widen the view)
    mf.camera.setExtent(arcpy.Describe(STATES).extent.projectAs(m.spatialReference))
    mf.camera.scale *= 1.04
    aprx.createTextElement(lyt, arcpy.Point(0.3, 8.15), "POINT", "County voter turnout, 2020 presidential election", 16, "Arial", "Bold", None, "Title")
    aprx.createTextElement(lyt, arcpy.Point(0.3, 0.45), "POINT",
                           "Standard-deviation classification (1 SD). Data: MIT Election Data and Science Lab, U.S. Census Bureau CVAP; Esri MOOC exercise data.",
                           8, "Arial", "Regular", None, "Credit")
    leg_box = arcpy.Polygon(arcpy.Array([arcpy.Point(0.6, 1.1), arcpy.Point(0.6, 3.4), arcpy.Point(3.0, 3.4), arcpy.Point(3.0, 1.1)]))
    leg = lyt.createMapSurroundElement(leg_box, "LEGEND", mf, None, "Legend")
    leg.showTitle = False
    for item in leg.items:
        if item.name == states_lyr.name:
            leg.removeItem(item)
    lyt.exportToPNG(str(C.FIG / "s1_turnout_layout.png"), resolution=150)
    aprx.save()
    C.log("arcpy.mp layout exported to figures/s1_turnout_layout.png")


def main():
    arcpy = C.import_arcpy()
    plt = C.setup_matplotlib()
    fields = list(VARS.keys()) + ["fips", "name", "state_abbr", "state_name"]
    paths, df = C.read_polygons(FC, fields)
    state_paths, _ = C.read_polygons(STATES, ["STATE_ABBR"], spatial_reference=arcpy.Describe(FC).spatialReference)
    C.log(f"Read {len(df)} county polygons")

    C.write_csv(skewness_table(df), C.RES / "s1_skewness.csv")
    histogram(plt, df)
    breaks = choropleth(plt, paths, df, state_paths)
    C.write_csv(state_boxplot(plt, df), C.RES / "s1_state_turnout.csv")
    C.write_csv(scatter_matrix(plt, df), C.RES / "s1_scatter_r2.csv")
    try:
        pro_layout(arcpy, breaks)
    except Exception as exc:  # the matplotlib figures are the primary output; the layout is a bonus
        C.log(f"arcpy.mp layout failed: {exc!r}")
        (C.LOGS / "s1_layout_error.txt").write_text(repr(exc), encoding="utf-8")
    C.log("Section 1 visualization done")


if __name__ == "__main__":
    main()
