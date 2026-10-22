"""Section 2: predicting voter turnout with Forest-based and Boosted Classification and Regression.

Course logic (Prediction exercises):
  * Train on ``CountyElections_IA_and_Neighbors`` (739 counties in Iowa and its
    neighbours, 2016 turnout and Esri demographic enrichment), response ``voter_turnout``.
  * Model progression: baseline variables -> many variables -> + distance to cities
    (Cities5..Cities10 as distance features) -> refined model keeping only variables
    above an importance threshold.
  * Downscale: predict the refined model onto ``Tracts_Iowa`` with explanatory-variable
    matching, and read the tool's explanatory variable range diagnostics.
  * Validation: 10 repeated random 10 % hold-outs (the tool's Number of Runs for Validation).

Honest extra: leave-one-state-out spatial cross-validation (train on 7 states, predict the 8th).

Also one national run (``CountyElections2020``, 3,112 counties, response ``voter_turnout_2020``)
so the county-level result is reported for the whole country too.

Outputs: outputs/s2_prediction.gdb, results/s2_*.csv|json, figures/s2_*.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TRAIN = str(C.PRED_GDB / "CountyElections_IA_and_Neighbors")
TRACTS = str(C.PRED_GDB / "Tracts_Iowa")
NATIONAL = str(C.DE_GDB / "CountyElections2020")
CITIES = [str(C.PRED_GDB / f"Cities{k}") for k in range(5, 11)]
N_RUNS = 10
N_TREES = 100
IMPORTANCE_THRESHOLD = 0.03  # keep variables with mean importance >= 3 % (my choice; the course picks by eye)

BASELINE = ["gender_medage_cy", "householdincome_pci_cy", "educationalattainment_bachdeg_cy_p"]
# county field -> tract field (identical unless noted); voter_laws is categorical
MANY = {
    "gender_medage_cy": "F5yearincrements_medage_cy",
    "householdincome_pci_cy": "householdincome_pci_cy",
    "educationalattainment_nohs_cy_p": "educationalattainment_nohs_cy_p",
    "educationalattainment_somehs_cy_p_1": "educationalattainment_somehs_cy_p",
    "educationalattainment_hsgrad_cy_p": "educationalattainment_hsgrad_cy_p",
    "educationalattainment_ged_cy_p": "educationalattainment_ged_cy_p",
    "educationalattainment_smcoll_cy_p": "educationalattainment_smcoll_cy_p",
    "educationalattainment_asscdeg_cy_p": "educationalattainment_asscdeg_cy_p",
    "educationalattainment_bachdeg_cy_p": "educationalattainment_bachdeg_cy_p",
    "educationalattainment_graddeg_cy_p": "educationalattainment_graddeg_cy_p",
    "maritalstatustotals_nevmarr_cy_p": "maritalstatustotals_nevmarr_cy_p",
    "F5yearincrements_pop18up_cy": "gender_pop18up_cy",
    "F5yearincrements_pop65_cy_p": "F5yearincrements_pop65_cy_p",
    "F5yearincrements_pop70_cy_p": "F5yearincrements_pop70_cy_p",
    "F5yearincrements_pop75_cy_p": "F5yearincrements_pop75_cy_p",
    "raceandhispanicorigin_divindx_cy": "raceandhispanicorigin_divindx_cy",
    "households_acshhbpov_p": "atrisk_acshhbpov_p",
    "psychographicsadvertising_mp28001a_b_p": "psychographicsadvertising_mp28001a_b_p",
    "financial_x14060_x_a": "financial_x14060_x_a",
    "financial_x14058_x_a": "financial_x14058_x_a",
    "state_votes_pcnt_diff": "state_votes_pcnt_diff",
    "voter_laws": "voter_laws",
}
CATEGORICAL = {"voter_laws"}
NATIONAL_BASE = ["gender_MEDAGE_CY", "householdincome_PCI_CY", "educationalattainment_BACHDEG_CY_P"]
NATIONAL_MANY = NATIONAL_BASE + [
    "educationalattainment_SOMEHS_CY_P", "households_ACSHHBPOV_P", "disposableincome_AVGDI_CY",
    "maritalstatustotals_NEVMARR_CY_P", "educationalattainment_GED_CY_P", "educationalattainment_HSGRAD_CY_P",
    "educationalattainment_GRADDEG_CY_P", "educationalattainment_NOHS_CY_P", "householdincome_AVGHINC_CY",
    "financial_X14060_X_A", "financial_X14068_X_A", "miscellaneous_X12024_X_A", "pop_sqmi",
]


def expl(vars_):
    return [[v, "true" if v in CATEGORICAL else "false"] for v in vars_]


def parse_oob(text):
    """'% of variation explained' is printed for 50 and 100 trees; keep the full-forest value."""
    m = re.search(r"% of variation explained\s+([\d.\s]+)", text)
    return float(m.group(1).split()[-1]) if m else None


def parse_range_diagnostics(text):
    """Parse the 'Explanatory Variable Range Diagnostics' table printed by the tool.

    Columns: variable, training min/max, validation-or-prediction min/max, share (a) and,
    for TRAIN runs, share (b). '*' marks ranges that do not coincide; '+' marks extrapolation
    into the prediction data (the share is < 1 and the prediction range is wider).
    """
    rows = []
    m = re.search(r"Explanatory Variable Range Diagnostics -+\n(.*?)\(a\) %", text, re.S)
    if not m:
        return rows
    lines = m.group(1).splitlines()[2:]
    for line in lines:
        parts = re.split(r"\s{2,}", line.strip())
        nums = parts[1:]
        try:
            vals = [float(x.rstrip("*+")) for x in nums]
        except ValueError:
            continue
        name = parts[0]
        if len(vals) == 9:  # PREDICT run: training, validation, prediction min/max + shares (a), (b), (c)
            rows.append({"variable": name, "train_min": vals[0], "train_max": vals[1], "validation_min": vals[2], "validation_max": vals[3],
                         "prediction_min": vals[4], "prediction_max": vals[5], "share_training_a": vals[6], "share_validation_b": vals[7],
                         "share_prediction_c": vals[8], "extrapolation_flag": nums[8].endswith("+"),
                         "note": "share (c) > 1 or '+' means the prediction data reach outside the training range (extrapolation)"})
        elif len(vals) == 6:  # TRAIN run: training, validation min/max + shares (a), (b)
            rows.append({"variable": name, "train_min": vals[0], "train_max": vals[1], "validation_min": vals[2], "validation_max": vals[3],
                         "share_training_a": vals[4], "share_validation_b": vals[5], "extrapolation_flag": False})
    return rows


def run_forest(arcpy, name, in_features, response, variables, distance=None, predict_to=None, matching=None,
               n_runs=N_RUNS, gdb=None):
    """Run the Forest tool once (TRAIN or PREDICT_FEATURES), save messages and tables; return a summary dict."""
    out_feat = f"{gdb}/{name}_pred" if predict_to else None
    kwargs = dict(
        prediction_type="PREDICT_FEATURES" if predict_to else "TRAIN",
        in_features=in_features, variable_predict=response, treat_variable_as_categorical="NUMERIC",
        explanatory_variables=expl(variables), distance_features=distance or None,
        features_to_predict=predict_to, output_features=out_feat,
        explanatory_variable_matching=matching, explanatory_distance_matching=[[d, d] for d in distance] if (distance and predict_to) else None,
        output_importance_table=f"{gdb}/{name}_importance", number_of_trees=N_TREES,
        percentage_for_training=10, output_validation_table=f"{gdb}/{name}_validation",
        number_validation_runs=n_runs, model_type="FOREST-BASED",
    )
    arcpy.stats.Forest(**{k: v for k, v in kwargs.items() if v is not None})
    text = C.save_messages(f"s2_forest_{name}")
    val = C.table_to_df(f"{gdb}/{name}_validation")
    imp = C.table_to_df(f"{gdb}/{name}_importance")
    r2col = next((c for c in val.columns if c.upper().startswith("R2")), None)
    r2 = val[r2col].astype(float) if r2col else None
    # importance table: one column per variable (upper-cased), one row per validation run
    varnames = list(variables) + [Path(d).name for d in (distance or [])]
    imp_mean = {}
    import pandas as pd

    for col in imp.columns:
        if col.upper() == "OBJECTID":
            continue
        vals = pd.to_numeric(imp[col], errors="coerce")
        if vals.notna().sum() == 0:
            continue
        orig = next((v for v in varnames if v.upper() == col.upper()), col)
        imp_mean[orig] = float(vals.mean())
    imp_mean = dict(sorted(imp_mean.items(), key=lambda kv: -kv[1]))
    summary = {
        "model": name, "n_features": C.count(in_features), "variables": variables, "distance_features": [Path(d).name for d in (distance or [])],
        "validation_runs": n_runs, "validation_r2_mean": float(r2.mean()) if r2 is not None else None,
        "validation_r2_min": float(r2.min()) if r2 is not None else None, "validation_r2_max": float(r2.max()) if r2 is not None else None,
        "validation_r2_values": [round(float(x), 4) for x in r2] if r2 is not None else None,
        "oob_pct_variation_explained": parse_oob(text),
        "importance_mean": {k: round(v, 4) for k, v in imp_mean.items()},
    }
    if predict_to:
        summary["range_diagnostics"] = parse_range_diagnostics(text)
        summary["output_features"] = out_feat
    C.log(f"{name}: validation R2 mean {summary['validation_r2_mean']:.3f} "
          f"[{summary['validation_r2_min']:.3f}, {summary['validation_r2_max']:.3f}], OOB {summary['oob_pct_variation_explained']}% ")
    return summary


def r2_score(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def leave_one_state_out(arcpy, gdb, variables, distance):
    """Spatially blocked CV: hold out one whole state at a time."""
    states = sorted(set(C.table_to_df(TRAIN, ["state"])["state"]))
    rows, pooled_y, pooled_hat = [], [], []
    for st in states:
        q = f"state = '{st}'"
        train_lyr = arcpy.management.MakeFeatureLayer(TRAIN, "loso_train", f"NOT ({q})")
        test_lyr = arcpy.management.MakeFeatureLayer(TRAIN, "loso_test", q)
        out = f"{gdb}/loso_{re.sub('[^A-Za-z]', '', st)}"
        arcpy.stats.Forest(
            prediction_type="PREDICT_FEATURES", in_features=train_lyr, variable_predict="voter_turnout",
            treat_variable_as_categorical="NUMERIC", explanatory_variables=expl(variables), distance_features=distance,
            features_to_predict=test_lyr, output_features=out, explanatory_variable_matching=[[v, v] for v in variables],
            explanatory_distance_matching=[[d, d] for d in distance], number_of_trees=N_TREES, percentage_for_training=10,
            number_validation_runs=1, model_type="FOREST-BASED",
        )
        C.save_messages(f"s2_loso_{st}")
        # the output keeps SOURCE_ID (= OBJECTID of the held-out features); join the observed turnout back on it
        pred = C.table_to_df(out, ["SOURCE_ID", "PREDICTED"])
        obs = C.table_to_df(TRAIN, ["OBJECTID", "voter_turnout"], q).rename(columns={"OBJECTID": "SOURCE_ID"})
        df = pred.merge(obs, on="SOURCE_ID")
        y, yhat = df["voter_turnout"].values.astype(float), df["PREDICTED"].values.astype(float)
        pooled_y += list(y)
        pooled_hat += list(yhat)
        rows.append({"held_out_state": st, "n_test": len(df), "n_train": C.count(TRAIN) - len(df), "r2_held_out": round(r2_score(y, yhat), 4),
                     "rmse": round(float(np.sqrt(np.mean((y - yhat) ** 2))), 4), "mean_actual": round(float(y.mean()), 4), "mean_predicted": round(float(yhat.mean()), 4)})
        C.log(f"LOSO {st}: R2 {rows[-1]['r2_held_out']:.3f}, RMSE {rows[-1]['rmse']:.3f}")
        for lyr in (train_lyr, test_lyr):
            arcpy.management.Delete(lyr)
    pooled = r2_score(pooled_y, pooled_hat)
    return rows, pooled


def figures(arcpy, plt, models, loso_rows, loso_pooled, gdb, refined_vars):
    # 1. validation R2 per model
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [m["model"] for m in models]
    for i, m in enumerate(models):
        vals = m["validation_r2_values"]
        ax.scatter([i] * len(vals), vals, s=14, color="#2c5d8f", alpha=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [m["validation_r2_mean"]] * 2, color="#d7191c", lw=2, zorder=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Validation R² (10 random 10 % hold-outs)")
    ax.set_title("Forest-based regression: validation R² by model (red = mean)")
    fig.tight_layout()
    fig.savefig(C.FIG / "s2_validation_r2.png")
    plt.close(fig)

    # 2. importance of the refined model
    ref = next(m for m in models if m["model"] == "M4_refined")
    imp = ref["importance_mean"]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(imp) + 1.2))
    keys = list(imp.keys())[::-1]
    ax.barh(keys, [imp[k] for k in keys], color="#7a9cc6")
    ax.set_xlabel("Mean variable importance (10 runs)")
    ax.set_title("Refined model: variable importance")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "s2_importance.png")
    plt.close(fig)

    # 3. LOSO vs random hold-out
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sts = [r["held_out_state"] for r in loso_rows]
    ax.bar(sts, [r["r2_held_out"] for r in loso_rows], color="#7a9cc6", label="Leave-one-state-out R²")
    ax.axhline(ref["validation_r2_mean"], color="#d7191c", lw=1.5, label=f"Random 10 % hold-out, mean R² = {ref['validation_r2_mean']:.2f}")
    ax.axhline(loso_pooled, color="#2c7bb6", lw=1.5, ls="--", label=f"LOSO pooled R² = {loso_pooled:.2f}")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_ylabel("R² on held-out counties")
    ax.set_title("Spatially blocked validation (refined model)")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", labelsize=8, rotation=20)
    fig.tight_layout()
    fig.savefig(C.FIG / "s2_spatial_cv.png")
    plt.close(fig)

    # 4. tract map
    paths, df = C.read_polygons(f"{gdb}/M4_refined_tracts_pred", ["PREDICTED"])
    cpaths, cdf = C.read_polygons(TRAIN, ["voter_turnout", "state"], "state = 'Iowa'", spatial_reference=arcpy.Describe(TRACTS).spatialReference)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    vmin, vmax = 0.45, 0.85
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin, vmax)
    C.draw_paths(axes[0], cpaths, [cmap(norm(v)) for v in cdf["voter_turnout"]], edgecolor="white", linewidth=0.3)
    axes[0].set_title("Observed county turnout (2016), Iowa")
    C.draw_paths(axes[1], paths, [cmap(norm(v)) for v in df["PREDICTED"]], edgecolor="white", linewidth=0.1)
    axes[1].set_title("Predicted tract turnout (refined forest model)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, shrink=0.7, label="Voter turnout")
    fig.savefig(C.FIG / "s2_tract_prediction.png", bbox_inches="tight")
    plt.close(fig)


def main():
    arcpy = C.import_arcpy()
    plt = C.setup_matplotlib()
    gdb = C.ensure_gdb("s2_prediction")
    arcpy.env.workspace = gdb
    models = []

    many_vars = list(MANY.keys())
    models.append(run_forest(arcpy, "M1_baseline", TRAIN, "voter_turnout", BASELINE, gdb=gdb))
    models.append(run_forest(arcpy, "M2_many", TRAIN, "voter_turnout", many_vars, gdb=gdb))
    models.append(run_forest(arcpy, "M3_distance", TRAIN, "voter_turnout", many_vars, distance=CITIES, gdb=gdb))

    # Refine: keep variables (and distance features) above the importance threshold
    imp = models[-1]["importance_mean"]
    total = sum(imp.values())
    keep = [k for k, v in imp.items() if v / total >= IMPORTANCE_THRESHOLD]
    refined_vars = [v for v in many_vars if v in keep]
    refined_dist = [d for d in CITIES if Path(d).name in keep]
    C.log(f"Refined variables ({len(refined_vars)}): {refined_vars}; distance features kept: {[Path(d).name for d in refined_dist]}")
    models.append(run_forest(arcpy, "M4_refined", TRAIN, "voter_turnout", refined_vars, distance=refined_dist or None, gdb=gdb))

    # Downscale to Iowa tracts
    matching = [[MANY[v], v] for v in refined_vars]
    down = run_forest(arcpy, "M4_refined_tracts", TRAIN, "voter_turnout", refined_vars, distance=refined_dist or None,
                      predict_to=TRACTS, matching=matching, gdb=gdb)
    tract_pred = C.table_to_df(down["output_features"], ["PREDICTED"])["PREDICTED"].astype(float)
    down["tract_prediction_summary"] = {"n_tracts": len(tract_pred), "mean": round(float(tract_pred.mean()), 4),
                                        "min": round(float(tract_pred.min()), 4), "max": round(float(tract_pred.max()), 4)}
    extrap = [r for r in down.get("range_diagnostics", []) if r["extrapolation_flag"]]
    down["n_variables_extrapolating"] = len(extrap)
    down["variables_extrapolating"] = [r["variable"] for r in extrap]
    C.log(f"Range diagnostics: {len(down.get('range_diagnostics', []))} variables parsed, {len(extrap)} flagged '+' "
          f"(prediction range wider than training): {[r['variable'] for r in extrap]}")
    C.write_csv(down.get("range_diagnostics", []), C.RES / "s2_range_diagnostics.csv")

    # Spatially blocked validation
    loso_rows, loso_pooled = leave_one_state_out(arcpy, gdb, refined_vars, refined_dist)
    C.write_csv(loso_rows, C.RES / "s2_spatial_cv_by_state.csv")

    # National county-level run (whole country, 2020)
    nat = [run_forest(arcpy, "N1_baseline_2020", NATIONAL, "voter_turnout_2020", NATIONAL_BASE, gdb=gdb),
           run_forest(arcpy, "N2_many_2020", NATIONAL, "voter_turnout_2020", NATIONAL_MANY, gdb=gdb)]

    rows = []
    for m in models + [down] + nat:
        rows.append({"model": m["model"], "features": m["n_features"], "n_variables": len(m["variables"]),
                     "n_distance_features": len(m["distance_features"]), "validation_runs": m["validation_runs"],
                     "val_r2_mean": round(m["validation_r2_mean"], 4), "val_r2_min": round(m["validation_r2_min"], 4),
                     "val_r2_max": round(m["validation_r2_max"], 4), "oob_pct_var_explained": m["oob_pct_variation_explained"]})
    C.write_csv(rows, C.RES / "s2_model_comparison.csv")
    imp_rows = [{"model": m["model"], "variable": k, "mean_importance": v} for m in models + nat for k, v in (m["importance_mean"] or {}).items()]
    C.write_csv(imp_rows, C.RES / "s2_variable_importance.csv")
    C.write_json({"models": models, "downscale": down, "national": nat, "importance_threshold": IMPORTANCE_THRESHOLD,
                  "refined_variables": refined_vars, "refined_distance_features": [Path(d).name for d in refined_dist],
                  "spatial_cv": {"by_state": loso_rows, "pooled_r2": loso_pooled,
                                 "mean_state_r2": float(np.mean([r["r2_held_out"] for r in loso_rows])),
                                 "random_holdout_r2_mean": models[-1]["validation_r2_mean"]}},
                 C.RES / "s2_prediction.json")
    figures(arcpy, plt, models, loso_rows, loso_pooled, gdb, refined_vars)
    C.log("Section 2 done")


if __name__ == "__main__":
    main()
