"""Section 1, Exercises 1-2: data engineering for county voter turnout.

Course steps reproduced (see COURSE_NOTES.md):
  1. Join Field: CountyPresElect.FIPS_txt <- CountyCVAP.FIPS (geoname, cvap_est_2008..2020)
     and Validate Join to count matches.
  2. Find where the null CVAP values cluster, remove Alaska, document the rest.
  3. Multi-year field calculations (votes_major, rawdiff, votes_other, pctdiff,
     voter_turnout, voter_turnout_dem/gop, winning_party), reimplemented from the
     course's "Multi-Year Calculate Field" script tool.
  4. Validate turnout: cap at 1 and record how many counties were capped per year.
  5. Geoenable: Join Field onto US_Counties on fips.
  6. Write a data-quality table.

Run:  python src/s1_data_engineering.py
Outputs: outputs/s1_data_engineering.gdb, results/s1_*.csv|json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

YEARS = ["2008", "2012", "2016", "2020"]
EXPECTED_JOIN_MATCHES = 3115  # course text: "3,115 of 3,153 records matched"
EXPECTED_COUNTY_MATCHES = 3112  # course text for the US_Counties join


def validate_join(arcpy, table, field, join_table, join_field, name):
    """Run Validate Join, keep its messages, and return the number of matching records."""
    out_tbl = str(Path(arcpy.env.scratchGDB) / f"vj_{name}")
    arcpy.management.ValidateJoin(table, field, join_table, join_field, out_tbl)
    text = C.save_messages(f"s1_validate_join_{name}")
    matched = C.parse_number(r"join has matched (\d[\d,]*) records", text)
    return int(matched) if matched is not None else None, text


def multi_year_calculate(arcpy, table, out_prefix, dtype, var1, var2, operator):
    """Clean reimplementation of the course's Multi-Year Calculate Field tool.

    For every 4-digit year suffix found in the table, computes
    ``<out_prefix>_<year> = <var1>_<year> <operator> <var2>_<year>``.
    """
    fields = [f.name for f in arcpy.ListFields(table)]
    years = sorted({f[-4:] for f in fields if re.fullmatch(r"\d{4}", f[-4:])})
    done = []
    for y in years:
        a, b = f"{var1}_{y}", f"{var2}_{y}"
        if a not in fields or b not in fields:
            continue
        out = f"{out_prefix}_{y}"
        if out not in fields:
            arcpy.management.AddField(table, out, dtype)
        # Null-safe: a null on either side gives a null result rather than an error.
        expr = f"calc(!{a}!, !{b}!)"
        code = (
            "def calc(a, b):\n"
            "    if a is None or b is None:\n"
            "        return None\n"
            f"    return a {operator} b\n"
        )
        if operator == "/":
            code = code.replace("    return a / b\n", "    return None if b == 0 else a / b\n")
        arcpy.management.CalculateField(table, out, expr, "PYTHON3", code)
        done.append(out)
    return done


def main():
    arcpy = C.import_arcpy()
    gdb = C.ensure_gdb("s1_data_engineering")
    arcpy.env.workspace = gdb
    quality = {}

    # --- Copy the course tables so the originals stay untouched -----------------
    elect = arcpy.management.Copy(str(C.DE_GDB / "CountyPresElect"), f"{gdb}/CountyPresElect")[0]
    cvap = arcpy.management.Copy(str(C.DE_GDB / "CountyCVAP"), f"{gdb}/CountyCVAP")[0]
    n_elect, n_cvap = C.count(elect), C.count(cvap)
    C.log(f"CountyPresElect: {n_elect} rows; CountyCVAP: {n_cvap} rows")

    # --- Step 1: Validate Join, then Join Field ---------------------------------
    matched, vj_text = validate_join(arcpy, elect, "FIPS_txt", cvap, "FIPS", "cvap")
    C.log(f"Validate Join (FIPS_txt <- FIPS): {matched} matching records; course expected {EXPECTED_JOIN_MATCHES}")
    arcpy.management.JoinField(elect, "FIPS_txt", cvap, "FIPS", ["geoname"] + [f"cvap_est_{y}" for y in YEARS])
    joined = C.count(elect, "geoname IS NOT NULL")
    quality["join_cvap"] = {
        "input_rows": n_elect, "join_table_rows": n_cvap, "validate_join_matches": matched,
        "rows_with_cvap_after_join": joined, "course_expected_matches": EXPECTED_JOIN_MATCHES,
    }

    # --- Step 2: where are the nulls? ---------------------------------------------
    df = C.table_to_df(elect, ["FIPS_txt", "county_name", "state", "geoname"] + [f"cvap_est_{y}" for y in YEARS]
                       + [f"totalvotes_{y}" for y in YEARS])
    null_mask = df["geoname"].isna()
    by_state = df.loc[null_mask, "state"].value_counts()
    C.log(f"Rows with no CVAP match, by state:\n{by_state.to_string()}")
    nulls_before = {y: int(df[f"cvap_est_{y}"].isna().sum()) for y in YEARS}
    non_alaska_nulls = df.loc[null_mask & (df["state"] != "ALASKA"), ["FIPS_txt", "county_name", "state"]]
    quality["nulls_after_join"] = {
        "rows_without_cvap_match": int(null_mask.sum()),
        "by_state": {k: int(v) for k, v in by_state.items()},
        "non_alaska_unmatched": non_alaska_nulls.to_dict("records"),
        "cvap_null_by_year_before_cleaning": nulls_before,
    }

    # Alaska reports election returns by legislative district (FIPS 02001, 02002, ...),
    # not by borough/census area, so its rows cannot match county CVAP. Remove them.
    n_alaska = C.count(elect, "state = 'ALASKA'")
    lyr = arcpy.management.MakeTableView(elect, "ak_view", "state = 'ALASKA'")
    arcpy.management.DeleteRows(lyr)
    arcpy.management.Delete(lyr)
    remaining_null = C.count(elect, "geoname IS NULL")
    C.log(f"Removed {n_alaska} Alaska district rows; {remaining_null} unmatched row(s) remain")
    quality["alaska"] = {"rows_removed": n_alaska, "reason": "Alaska reports votes by legislative district, not by borough; no CVAP match possible"}

    # --- Step 3: multi-year field calculations ------------------------------------
    multi_year_calculate(arcpy, elect, "votes_major", "DOUBLE", "votes_dem", "votes_gop", "+")
    multi_year_calculate(arcpy, elect, "rawdiff_dem_vs_gop", "DOUBLE", "votes_dem", "votes_gop", "-")
    multi_year_calculate(arcpy, elect, "votes_other", "DOUBLE", "totalvotes", "votes_major", "-")
    multi_year_calculate(arcpy, elect, "pctdiff_dem_vs_gop", "DOUBLE", "rawdiff_dem_vs_gop", "totalvotes", "/")
    multi_year_calculate(arcpy, elect, "voter_turnout", "DOUBLE", "totalvotes", "cvap_est", "/")
    multi_year_calculate(arcpy, elect, "voter_turnout_dem", "DOUBLE", "votes_dem", "cvap_est", "/")
    multi_year_calculate(arcpy, elect, "voter_turnout_gop", "DOUBLE", "votes_gop", "cvap_est", "/")
    for y in YEARS:
        fld = f"winning_party_{y}"
        arcpy.management.AddField(elect, fld, "TEXT", field_length=20)
        arcpy.management.CalculateField(
            elect, fld, f"win(!votes_dem_{y}!, !votes_gop_{y}!, !votes_other_{y}!)", "PYTHON3",
            "def win(d, g, o):\n"
            "    if d is None or g is None or o is None:\n        return None\n"
            "    if d > g and d > o:\n        return 'Democratic Party'\n"
            "    if g > d and g > o:\n        return 'Republican Party'\n"
            "    return 'Other'\n",
        )
    C.log("Multi-year fields calculated")

    # --- Step 4: validate turnout, cap at 1 ---------------------------------------
    capped_rows = []
    caps = {}
    for y in YEARS:
        where = f"voter_turnout_{y} > 1"
        over = C.table_to_df(elect, ["FIPS_txt", "county_name", "state", f"voter_turnout_{y}", f"totalvotes_{y}", f"cvap_est_{y}"], where)
        for _, r in over.iterrows():
            capped_rows.append({"year": y, "FIPS": r["FIPS_txt"], "county": r["county_name"], "state": r["state"],
                                "voter_turnout": round(float(r[f"voter_turnout_{y}"]), 4),
                                "totalvotes": int(r[f"totalvotes_{y}"]), "cvap_est": int(r[f"cvap_est_{y}"])})
        caps[y] = len(over)
        if len(over):
            lyr = arcpy.management.MakeTableView(elect, "cap_view", where)
            arcpy.management.CalculateField(lyr, f"voter_turnout_{y}", "1.0", "PYTHON3")
            arcpy.management.Delete(lyr)
    C.log(f"Turnout > 1 capped per year: {caps}")
    C.write_csv(capped_rows, C.RES / "s1_turnout_capped_counties.csv")
    quality["turnout_cap"] = {
        "counties_capped_by_year": caps, "total_capped_rows": len(capped_rows),
        "reason": ("CVAP is a 5-year ACS average while votes are counted on one day; in small counties "
                   "sampling error and population change can push votes above the estimated citizen voting-age population"),
    }

    # Null counts after cleaning
    df2 = C.table_to_df(elect, ["FIPS_txt"] + [f"cvap_est_{y}" for y in YEARS] + [f"voter_turnout_{y}" for y in YEARS])
    still_null = C.table_to_df(elect, ["FIPS_txt", "county_name", "state", "geoname"] + [f"cvap_est_{y}" for y in YEARS],
                               " OR ".join(f"cvap_est_{y} IS NULL" for y in YEARS))
    C.log("Rows with a null CVAP value in some year after cleaning:\n" + still_null.to_string())
    quality["nulls_after_cleaning"] = {
        "rows": len(df2),
        "rows_with_any_null_cvap": still_null.to_dict("records"),
        "cvap_null_by_year": {y: int(df2[f"cvap_est_{y}"].isna().sum()) for y in YEARS},
        "turnout_null_by_year": {y: int(df2[f"voter_turnout_{y}"].isna().sum()) for y in YEARS},
    }

    # --- Step 5: geoenable --------------------------------------------------------
    counties = arcpy.management.Copy(str(C.DE_GDB / "US_Counties"), f"{gdb}/CountyElections")[0]
    matched_c, _ = validate_join(arcpy, counties, "fips", elect, "FIPS_txt", "counties")
    keep = ["county_name", "state", "geoname"]
    for pref in ["totalvotes", "cvap_est", "votes_dem", "votes_gop", "votes_major", "votes_other", "rawdiff_dem_vs_gop",
                 "pctdiff_dem_vs_gop", "voter_turnout", "voter_turnout_dem", "voter_turnout_gop", "winning_party"]:
        keep += [f"{pref}_{y}" for y in YEARS]
    arcpy.management.JoinField(counties, "fips", elect, "FIPS_txt", keep)
    n_counties = C.count(counties)
    n_geo = C.count(counties, "voter_turnout_2020 IS NOT NULL")
    unmatched = C.table_to_df(counties, ["fips", "name", "state_name"], "county_name IS NULL")
    C.log(f"US_Counties: {n_counties} polygons; {n_geo} have 2020 turnout; Validate Join matches = {matched_c} (course expected {EXPECTED_COUNTY_MATCHES})")
    quality["geoenable"] = {
        "county_polygons": n_counties, "validate_join_matches": matched_c,
        "polygons_with_turnout_2020": n_geo, "course_expected_matches": EXPECTED_COUNTY_MATCHES,
        "unmatched_polygons": unmatched.to_dict("records"),
    }
    # Election rows that found no polygon (e.g. independent cities / FIPS changes)
    polys = set(C.table_to_df(counties, ["fips"])["fips"])
    orphan = df2.loc[~df2["FIPS_txt"].isin(polys), "FIPS_txt"].tolist()
    quality["geoenable"]["election_rows_without_polygon"] = len(orphan)
    quality["geoenable"]["election_rows_without_polygon_fips"] = orphan[:50]

    # --- Step 6: data-quality table -----------------------------------------------
    rows = [
        {"check": "CountyPresElect rows", "value": n_elect},
        {"check": "CountyCVAP rows", "value": n_cvap},
        {"check": "Validate Join matches (FIPS_txt <- FIPS)", "value": matched},
        {"check": "Course expected join matches", "value": EXPECTED_JOIN_MATCHES},
        {"check": "Rows without CVAP match", "value": int(null_mask.sum())},
        {"check": "  of which Alaska", "value": int(by_state.get("ALASKA", 0))},
        {"check": "  of which other", "value": int(null_mask.sum() - by_state.get("ALASKA", 0))},
        {"check": "Alaska rows removed", "value": n_alaska},
        {"check": "Rows after cleaning", "value": len(df2)},
    ]
    for y in YEARS:
        rows.append({"check": f"Turnout > 1 capped, {y}", "value": caps[y]})
    rows += [
        {"check": "US_Counties polygons", "value": n_counties},
        {"check": "Validate Join matches (fips <- FIPS_txt)", "value": matched_c},
        {"check": "Course expected county matches", "value": EXPECTED_COUNTY_MATCHES},
        {"check": "Polygons with 2020 turnout", "value": n_geo},
        {"check": "Election rows without a polygon", "value": len(orphan)},
    ]
    C.write_csv(rows, C.RES / "s1_data_quality.csv")
    C.write_json(quality, C.RES / "s1_data_quality.json")

    # Small summary table of turnout per year for the README
    summ = []
    for y in YEARS:
        s = df2[f"voter_turnout_{y}"].describe()
        summ.append({"year": y, "n": int(s["count"]), "mean": round(float(s["mean"]), 4), "median": round(float(df2[f"voter_turnout_{y}"].median()), 4),
                     "min": round(float(s["min"]), 4), "max": round(float(s["max"]), 4), "capped": caps[y]})
    C.write_csv(summ, C.RES / "s1_turnout_summary_by_year.csv")
    C.log("Section 1 data engineering done")


if __name__ == "__main__":
    main()
