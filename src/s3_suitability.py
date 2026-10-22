"""Section 3: suitability model for a carbon-neutral headquarters in Vermont.

Rebuilds the course's Suitability Modeler workflow with Spatial Analyst:
  1. Derive criteria rasters the course derives in the exercise: slope (Surface
     Parameters) and distance to major roads (Distance Accumulation with the Green
     Mountain ridge as a barrier). The other distance criteria (Dist_*), Aspect, Solar
     and Views_Zero are shipped pre-computed in Data.gdb and are used as-is.
  2. Transform every criterion to a 1-10 suitability scale with Rescale by Function
     (continuous criteria) or a lookup table (land use, Reclassify).
  3. Build four submodels (building site, energy independence, access to amenities,
     environmental concerns) as weighted sums, combine them, and run Locate Regions.

Which numbers are the course's and which are mine:
  * Suitability.sam (the course's saved Suitability Modeler file) stores the exact
    transformation functions, parameters and weights for three submodels:
    HQ_Environmental_Concerns, HQ_Access_to_Amenities and HQ_Energy_Independence.
    Those are read from the file at run time and used verbatim.
  * The Building Site submodel and the final combination are not saved in the .sam, so
    the land-use lookup values, the building-site weights (all 1.0), the equal weights
    of the four submodels and the Locate Regions area (2 km2) are my assumptions.
    They are flagged "assumption" in results/s3_criteria.csv.

Outputs: outputs/s3_suitability.gdb, results/s3_*.csv|json, figures/s3_*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ELEV = str(C.SUIT_GDB / "Elevation_30")
LANDUSE_LOOKUP = {  # assumption: NLCD class -> suitability 1-10 for a low-impact building site
    11: 1,   # Open Water
    21: 8,   # Developed, Open Space
    22: 9,   # Developed, Low Intensity
    23: 10,  # Developed, Medium Intensity
    24: 7,   # Developed, High Intensity (already built out)
    31: 9,   # Barren Land
    41: 3, 42: 3, 43: 3,  # Forest (clearing forest is not carbon-neutral)
    52: 6,   # Shrub/Scrub
    71: 8,   # Herbaceous
    81: 7,   # Hay/Pasture
    82: 5,   # Cultivated Crops
    90: 1, 95: 1,  # Wetlands
}
SUBMODEL_WEIGHTS = {"HQ_Building_Site": 1.0, "HQ_Energy_Independence": 1.0, "HQ_Access_to_Amenities": 1.0, "HQ_Environmental_Concerns": 1.0}
LOCATE_AREA_KM2 = 2.0


def read_sam():
    """Pull the transformation function, its parameters and the weight of every criterion from Suitability.sam."""
    d = json.loads(C.SUIT_SAM.read_text(encoding="utf-8"))["SAModels"]
    models = {}
    for name in d["Info"]["ModelNames"]:
        s = d[name]["Suitability"]
        crits = []
        for i, crit in enumerate(s["CriteriaNames"]):
            fn = s["ContinuousTransformName"][i]
            params = next(p[fn] for p in s["TransformParameters"][i]["ContinuousTransformParams"] if fn in p)
            params = {k: v for k, v in params.items() if k != "EnableItems"}
            crits.append({"criterion": crit, "function": fn, "params": params, "weight": s["WeightValues"][i]})
        models[name] = {"criteria": crits, "from_scale": s["FromScale"], "to_scale": s["ToScale"]}
    return models


def tf_from_sam(sa, fn, p):
    """Translate a Suitability Modeler function record into an arcpy.sa transformation object."""
    lo, blo, hi, ahi = p["LowerThreshold"], p["BelowThreshold"], p["UpperThreshold"], p["AboveThreshold"]
    if fn == "Linear":
        a, b = p["MinimumValue"], p["MaximumValue"]
        if p.get("InvertFunction"):
            a, b = b, a
        return sa.TfLinear(a, b, lo, blo, hi, ahi)
    if fn == "LogisticGrowth":
        return sa.TfLogisticGrowth(p["MinimumValue"], p["MaximumValue"], p["YInterceptPct"], lo, blo, hi, ahi)
    if fn == "LogisticDecay":
        return sa.TfLogisticDecay(p["MinimumValue"], p["MaximumValue"], p["YInterceptPct"], lo, blo, hi, ahi)
    if fn == "MSLarge":
        return sa.TfMSLarge(p["MeanMultiplier"], p["StdvMultiplier"], lo, blo, hi, ahi)
    if fn == "MSSmall":
        return sa.TfMSSmall(p["MeanMultiplier"], p["StdvMultiplier"], lo, blo, hi, ahi)
    if fn == "Large":
        return sa.TfLarge(p["MidPoint"], p["PointSpread"], lo, blo, hi, ahi)
    if fn == "Small":
        return sa.TfSmall(p["MidPoint"], p["PointSpread"], lo, blo, hi, ahi)
    if fn == "Gaussian":
        return sa.TfGaussian(p["MidPoint"], p["PointSpread"], lo, blo, hi, ahi)
    raise ValueError(fn)


def raster_minmax(arcpy, r):
    return (float(arcpy.management.GetRasterProperties(r, "MINIMUM")[0]),
            float(arcpy.management.GetRasterProperties(r, "MAXIMUM")[0]))


def raster_array(arcpy, path):
    r = arcpy.Raster(path)
    arr = arcpy.RasterToNumPyArray(r, nodata_to_value=-99999).astype(float)
    ext = r.extent
    return np.ma.masked_where(arr <= -99998, arr), (ext.XMin, ext.XMax, ext.YMin, ext.YMax)


def read_lines(fc, spatial_reference=None):
    import arcpy

    segs = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@JSON"], spatial_reference=spatial_reference) as cur:
        for (js,) in cur:
            for path in json.loads(js).get("paths", []):
                segs.append(np.asarray(path, float)[:, :2])
    return segs


def main():
    arcpy = C.import_arcpy()
    plt = C.setup_matplotlib()
    sa = arcpy.sa
    arcpy.CheckOutExtension("Spatial")
    gdb = C.ensure_gdb("s3_suitability")
    arcpy.env.workspace = gdb
    arcpy.env.snapRaster = ELEV
    arcpy.env.extent = ELEV
    arcpy.env.cellSize = ELEV
    arcpy.env.outputCoordinateSystem = arcpy.Describe(ELEV).spatialReference
    arcpy.env.mask = str(C.SUIT_GDB / "Study_Mask")
    sam = read_sam()
    criteria_rows, transformed = [], {}

    # ---- 1. Derived criteria -------------------------------------------------
    C.log("Surface Parameters: slope (degrees, quadratic surface, 30 m)")
    slope = sa.SurfaceParameters(ELEV, "SLOPE", "QUADRATIC", None, "FIXED_NEIGHBORHOOD", "METER", "DEGREE")
    slope.save(f"{gdb}/Slope")
    C.log("Distance Accumulation: distance to major roads with the Green Mountain ridge as a barrier")
    dist_mroads = sa.DistanceAccumulation(str(C.SUIT_GDB / "MajorRoads"), str(C.SUIT_GDB / "GreenMTRidge"))
    dist_mroads.save(f"{gdb}/Dist_MRoads")
    # same distance without the barrier, to show what the barrier changes
    dist_nobar = sa.DistanceAccumulation(str(C.SUIT_GDB / "MajorRoads"))
    dist_nobar.save(f"{gdb}/Dist_MRoads_no_barrier")

    # ---- 2. Transform: building-site submodel (course functions, my parameters) ----
    lu = sa.Reclassify(str(C.SUIT_GDB / "LandUse"), "Value", sa.RemapValue([[k, v] for k, v in LANDUSE_LOOKUP.items()]), "NODATA")
    lu.save(f"{gdb}/T_LandUse")
    transformed["LandUse"] = lu
    criteria_rows.append({"submodel": "HQ_Building_Site", "criterion": "LandUse", "function": "Lookup table (Reclassify)",
                          "parameters": json.dumps(LANDUSE_LOOKUP), "weight": 1.0, "source": "assumption (course uses a lookup table; values not saved in .sam)"})

    s_min, s_max = raster_minmax(arcpy, slope)
    t = sa.RescaleByFunction(slope, sa.TfMSSmall(1, 1, s_min, 10, s_max, 1), 1, 10)
    t.save(f"{gdb}/T_Slope")
    transformed["Slope"] = t
    criteria_rows.append({"submodel": "HQ_Building_Site", "criterion": "Slope", "function": "MSSmall", "parameters": "mean x1, sd x1",
                          "weight": 1.0, "source": "course function (MSSmall); weight 1.0 is an assumption"})

    d_min, d_max = raster_minmax(arcpy, dist_mroads)
    t = sa.RescaleByFunction(dist_mroads, sa.TfSmall(d_max / 2, 5, d_min, 10, d_max, 1), 1, 10)
    t.save(f"{gdb}/T_Dist_MRoads")
    transformed["Dist_MRoads"] = t
    criteria_rows.append({"submodel": "HQ_Building_Site", "criterion": "Dist_MRoads", "function": "Small", "parameters": f"midpoint {d_max / 2:.1f} m (half of max), spread 5",
                          "weight": 1.0, "source": "course function (Small); midpoint = default half-range as in the .sam criteria; weight assumption"})

    v_min, v_max = raster_minmax(arcpy, str(C.SUIT_GDB / "Views_Zero"))
    t = sa.RescaleByFunction(str(C.SUIT_GDB / "Views_Zero"), sa.TfLogisticDecay(v_min, v_max, 99, v_min, 10, v_max, 1), 1, 10)
    t.save(f"{gdb}/T_Views_Zero")
    transformed["Views_Zero"] = t
    criteria_rows.append({"submodel": "HQ_Building_Site", "criterion": "Views_Zero", "function": "LogisticDecay", "parameters": f"min {v_min:.0f}, max {v_max:.0f}, y-intercept 99 % (tool default)",
                          "weight": 1.0, "source": "course function (logistic decay, per the exercise); parameters default; weight assumption"})

    # ---- 2b. Transform: the three submodels saved in Suitability.sam (verbatim) ----
    for model, spec in sam.items():
        for crit in spec["criteria"]:
            src = str(C.SUIT_GDB / crit["criterion"])
            t = sa.RescaleByFunction(src, tf_from_sam(sa, crit["function"], crit["params"]), spec["from_scale"], spec["to_scale"])
            t.save(f"{gdb}/T_{crit['criterion']}")
            transformed[crit["criterion"]] = t
            criteria_rows.append({"submodel": model, "criterion": crit["criterion"], "function": crit["function"],
                                  "parameters": json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in crit["params"].items()}),
                                  "weight": crit["weight"], "source": "course (Suitability.sam)"})
    C.write_csv(criteria_rows, C.RES / "s3_criteria.csv")

    # ---- 3. Submodels: weighted sums on the 1-10 scale -------------------------
    submodels = {"HQ_Building_Site": [("LandUse", 1.0), ("Slope", 1.0), ("Dist_MRoads", 1.0), ("Views_Zero", 1.0)]}
    for model, spec in sam.items():
        submodels[model] = [(c["criterion"], c["weight"]) for c in spec["criteria"]]
    sub_rasters, ranges = {}, []
    for model, items in submodels.items():
        total = None
        for crit, w in items:
            term = sa.Times(transformed[crit], w)
            total = term if total is None else sa.Plus(total, term)
        total.save(f"{gdb}/{model}")
        sub_rasters[model] = total
        mn, mx = raster_minmax(arcpy, total)
        row = {"submodel": model, "weight_sum": sum(w for _, w in items), "my_min": round(mn, 3), "my_max": round(mx, 3)}
        course = str(C.SUIT_COURSE_GDB / model)
        if arcpy.Exists(course):
            cmn, cmx = raster_minmax(arcpy, course)
            row.update({"course_min": round(cmn, 3), "course_max": round(cmx, 3)})
        ranges.append(row)
        C.log(f"{model}: range {mn:.2f}-{mx:.2f} (weights sum {row['weight_sum']})")
    C.write_csv(ranges, C.RES / "s3_submodel_ranges.csv")

    # ---- 4. Final model: rescale each submodel to 1-10, equal weights ----------
    final = None
    for model, w in SUBMODEL_WEIGHTS.items():
        mn, mx = raster_minmax(arcpy, sub_rasters[model])
        s10 = sa.RescaleByFunction(sub_rasters[model], sa.TfLinear(mn, mx), 1, 10)
        s10.save(f"{gdb}/{model}_1to10")
        term = sa.Times(s10, w)
        final = term if final is None else sa.Plus(final, term)
    final = sa.Divide(final, sum(SUBMODEL_WEIGHTS.values()))
    final.save(f"{gdb}/HQ_Suitability")
    f_min, f_max = raster_minmax(arcpy, final)
    f_mean = float(arcpy.management.GetRasterProperties(final, "MEAN")[0])
    C.log(f"Final suitability: {f_min:.2f}-{f_max:.2f}, mean {f_mean:.2f}")

    # ---- 5. Locate the best region ---------------------------------------------
    region = sa.LocateRegions(final, LOCATE_AREA_KM2, "SQUARE_KILOMETERS", 1, "CIRCLE", 0, 50, "HIGHEST_AVERAGE_VALUE")
    region.save(f"{gdb}/Best_Region")
    C.save_messages("s3_locate_regions")
    poly = arcpy.conversion.RasterToPolygon(region, f"{gdb}/Best_Region_poly", "NO_SIMPLIFY", "Value")[0]
    arcpy.sa.ZonalStatisticsAsTable(region, "Value", final, f"{gdb}/Best_Region_stats", "DATA", "ALL")
    zs = C.table_to_df(f"{gdb}/Best_Region_stats")
    cen = next(arcpy.da.SearchCursor(poly, ["SHAPE@TRUECENTROID", "SHAPE@AREA"]))
    sr = arcpy.Describe(ELEV).spatialReference
    pt = arcpy.PointGeometry(arcpy.Point(*cen[0]), sr).projectAs(arcpy.SpatialReference(4326)).firstPoint
    # Which submodels are strong at the site?
    sub_at_site = {}
    for model in SUBMODEL_WEIGHTS:
        arcpy.sa.ZonalStatisticsAsTable(region, "Value", f"{gdb}/{model}_1to10", f"{gdb}/zs_{model}", "DATA", "MEAN")
        sub_at_site[model] = round(float(C.table_to_df(f"{gdb}/zs_{model}")["MEAN"].iloc[0]), 2)
    site = {
        "locate_regions": {"total_area_km2": LOCATE_AREA_KM2, "shape": "CIRCLE", "regions": 1, "evaluation": "HIGHEST_AVERAGE_VALUE", "shape_tradeoff": 50,
                           "note": "area is my choice; the .sam Locate block only holds the dialog default (338.56 km2 = 10 % of the extent)"},
        "best_region": {"area_km2": round(float(cen[1]) / 1e6, 3), "centroid_x_vt_stateplane_m": round(cen[0][0], 1), "centroid_y_vt_stateplane_m": round(cen[0][1], 1),
                        "centroid_lon": round(pt.X, 5), "centroid_lat": round(pt.Y, 5),
                        "mean_suitability": round(float(zs["MEAN"].iloc[0]), 3), "min_suitability": round(float(zs["MIN"].iloc[0]), 3),
                        "max_suitability": round(float(zs["MAX"].iloc[0]), 3), "submodel_means_1to10": sub_at_site},
        "final_surface": {"min": round(f_min, 3), "max": round(f_max, 3), "mean": round(f_mean, 3), "submodel_weights": SUBMODEL_WEIGHTS},
        "submodel_ranges": ranges, "criteria": criteria_rows,
    }
    C.write_json(site, C.RES / "s3_suitability.json")

    # ---- 6. Figures ---------------------------------------------------------------
    figures(arcpy, plt, gdb, criteria_rows)
    C.log("Section 3 done")


SUIT_CMAP = "RdYlBu"  # red = least suitable, blue = most; readable with red-green colour blindness
SUBMODEL_TITLES = {"HQ_Building_Site": "Building site", "HQ_Energy_Independence": "Energy independence",
                   "HQ_Access_to_Amenities": "Access to amenities", "HQ_Environmental_Concerns": "Environmental concerns"}
CRITERION_TITLES = {
    "LandUse": "Land use (lookup table)", "Slope": "Slope (MSSmall)", "Dist_MRoads": "Distance to major roads (Small)",
    "Views_Zero": "Visibility from roads (LogisticDecay)", "Dist_Protect": "Distance to protected areas (LogisticGrowth)",
    "Dist_Biol": "Distance to biological areas (MSLarge)", "Dist_Wetlands": "Distance to wetlands (Linear)",
    "Dist_Streams": "Distance to streams (Large)", "Dist_Airport": "Distance to airport (Small)",
    "Dist_Rec": "Distance to recreation sites (MSSmall)", "Dist_Resident": "Distance to residential areas (Linear, inverted)",
    "Aspect": "Aspect, south-facing best (Gaussian)", "Dist_Elect": "Distance to electric corridors (Small)",
    "Solar": "Solar radiation (Large)",
}


def figures(arcpy, plt, gdb, criteria_rows):
    """Draw the Section 3 figures from the rasters in outputs/s3_suitability.gdb."""
    from matplotlib.collections import LineCollection

    roads = read_lines(str(C.SUIT_GDB / "MajorRoads"))
    ridge = read_lines(str(C.SUIT_GDB / "GreenMTRidge"))
    poly = f"{gdb}/Best_Region_poly"
    region_poly, _ = C.read_polygons(poly, ["gridcode"])
    cen = next(arcpy.da.SearchCursor(poly, ["SHAPE@TRUECENTROID"]))[0]
    hs, ext = raster_array(arcpy, str(C.SUIT_GDB / "Hillshade"))
    fin, fext = raster_array(arcpy, f"{gdb}/HQ_Suitability")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(hs, cmap="gray", extent=ext, vmin=0, vmax=255)
    im = ax.imshow(fin, cmap=SUIT_CMAP, extent=fext, alpha=0.75, vmin=1, vmax=10)
    ax.add_collection(LineCollection(roads, colors="#333333", linewidths=0.5))
    ax.add_collection(LineCollection(ridge, colors="#000000", linewidths=1.4, linestyles="--"))
    C.draw_paths(ax, region_poly, "none", edgecolor="#000000", linewidth=2)
    ax.annotate(f"Best {LOCATE_AREA_KM2:g} km² region", xy=(cen[0], cen[1]), xytext=(20, 30), textcoords="offset points",
                fontsize=9, fontweight="bold", arrowprops={"arrowstyle": "->"})
    ax.set_xlim(fext[0], fext[1])
    ax.set_ylim(fext[2], fext[3])
    ax.set_axis_off()
    ax.set_title("Headquarters suitability, Vermont study area (1 = least, 10 = most suitable)\n"
                 "solid black = major roads, dashed black = Green Mountain ridge barrier", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Suitability (1-10)")
    fig.tight_layout()
    fig.savefig(C.FIG / "s3_suitability_map.png")
    plt.close(fig)

    names = list(SUBMODEL_WEIGHTS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout="constrained")
    for ax, model in zip(axes.ravel(), names):
        arr, e = raster_array(arcpy, f"{gdb}/{model}_1to10")
        ax.imshow(hs, cmap="gray", extent=ext, vmin=0, vmax=255)
        im = ax.imshow(arr, cmap=SUIT_CMAP, extent=e, alpha=0.75, vmin=1, vmax=10)
        ax.set_title(SUBMODEL_TITLES.get(model, model))
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.5, label="Suitability (1-10)")
    fig.suptitle("Submodels rescaled to 1-10 (equal weights in the final model)")
    fig.savefig(C.FIG / "s3_submodels.png", bbox_inches="tight")
    plt.close(fig)

    crit_names = [r["criterion"] for r in criteria_rows]
    n = len(crit_names)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 2.9 * rows_n), layout="constrained")
    for ax, crit in zip(axes.ravel(), crit_names):
        arr, e = raster_array(arcpy, f"{gdb}/T_{crit}")
        im = ax.imshow(arr, cmap=SUIT_CMAP, extent=e, vmin=1, vmax=10)
        ax.set_title(CRITERION_TITLES.get(crit, crit), fontsize=8)
        ax.set_axis_off()
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.colorbar(im, ax=axes.ravel()[n:].tolist() or axes.ravel().tolist(), shrink=0.6, label="Suitability (1-10)")
    fig.suptitle("Transformed criteria (1 = least suitable, 10 = most); top row = building-site submodel (assumed weights)")
    fig.savefig(C.FIG / "s3_criteria_grid.png", bbox_inches="tight")
    plt.close(fig)

    # Barrier effect figure
    a, e = raster_array(arcpy, f"{gdb}/Dist_MRoads")
    b, _ = raster_array(arcpy, f"{gdb}/Dist_MRoads_no_barrier")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, arr, ttl in zip(axes, [b, a], ["Straight-line distance to major roads", "With the Green Mountain ridge as a barrier"]):
        im = ax.imshow(arr / 1000, cmap="magma_r", extent=e)
        ax.add_collection(LineCollection(roads, colors="#ffffff", linewidths=0.4))
        ax.add_collection(LineCollection(ridge, colors="#00e5ff", linewidths=1.0, linestyles="--"))
        ax.set_title(ttl, fontsize=9)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.7, label="Distance to major roads (km)")
    fig.savefig(C.FIG / "s3_distance_barrier.png", bbox_inches="tight")
    plt.close(fig)


def figures_only():
    """Redraw the figures from results/s3_criteria.csv and the existing geodatabase."""
    import csv

    arcpy = C.import_arcpy()
    plt = C.setup_matplotlib()
    with open(C.RES / "s3_criteria.csv", encoding="utf-8", newline="") as f:
        criteria_rows = list(csv.DictReader(f))
    figures(arcpy, plt, str(C.OUT / "s3_suitability.gdb"), criteria_rows)
    C.log("Section 3 figures redrawn")


if __name__ == "__main__":
    figures_only() if "--figures-only" in sys.argv else main()
