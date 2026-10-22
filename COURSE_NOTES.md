# Course notes: script ↔ exercise map

Reference for working through the Esri MOOC *Spatial Data Science: The New Frontier in Analytics* (Fall 2026) with ArcGIS Pro 3.7. Each row says which part of a script reproduces which exercise step, the tool the course uses, and what to look at in the output. The original exercise folders in `C:\EsriTraining\` are untouched; the scripts work on the copies in `data/`.

Run everything from the ArcGIS Pro Python environment:

```
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" run_all.py          # all sections
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe" run_all.py s2       # one section
```

Each script writes its ArcGIS outputs to `outputs/<section>.gdb`, every tool's messages to `outputs/logs/`, metric tables to `results/`, and figures to `figures/`.

## Section 1 — Data engineering (`src/s1_data_engineering.py`)

| Exercise step | Course tool / notebook cell | Script function | Output to check |
|---|---|---|---|
| Ex 1: join CVAP to election table on FIPS | Join Field (`CountyPresElect.FIPS_txt` ← `CountyCVAP.FIPS`), Validate Join | `validate_join()`, `JoinField` in `main()` | `outputs/logs/s1_validate_join_cvap.txt` ("A one-to-one join has matched 3115 records"), `results/s1_data_quality.json → join_cvap` |
| Ex 1: find and explain null CVAP values | Select by attribute `cvap_est IS NULL`, sort by state | null analysis in `main()` | `s1_data_quality.json → nulls_after_join` (38 unmatched rows, all Alaska legislative districts) |
| Ex 1: remove Alaska | Select + Delete Rows | `DeleteRows` on `state = 'ALASKA'` | 41 rows removed (38 unmatched + 3 districts whose FIPS happen to match a borough code) |
| Ex 1: one Louisiana record | — | `still_null` query | LaSalle Parish (22059) has no 2008 CVAP; kept, 2008 turnout is null there |
| Ex 2: Multi-Year Calculate Field | Script tool in `Multi-Year Calculate Field.atbx` | `multi_year_calculate()` — clean reimplementation (null-safe, division-by-zero safe) | fields `votes_major_*`, `rawdiff_dem_vs_gop_*`, `votes_other_*`, `pctdiff_dem_vs_gop_*`, `voter_turnout_*`, `voter_turnout_dem_*`, `voter_turnout_gop_*`, `winning_party_*` |
| Ex 2: validate turnout ≤ 1 | Select `voter_turnout > 1`, Calculate Field = 1 | cap loop in `main()` | `results/s1_turnout_capped_counties.csv` (25 county-years), `s1_turnout_summary_by_year.csv` |
| Ex 2: geoenable | Join Field `US_Counties.fips` ← `FIPS_txt`, Validate Join | second `validate_join()` + `JoinField` | `outputs/s1_data_engineering.gdb/CountyElections` (3,112 of 3,112 matched) |
| Data-quality report | — | end of `main()` | `results/s1_data_quality.csv` |

The course notebook (`Data Engineering Notebook.ipynb`) does the same work in pandas and also merges Kansas City into Jackson County, MO and Bedford city into Bedford County, VA, then geoenables through the Living Atlas counties layer. The tables shipped in the course geodatabase already have those fixes applied, which is why the Pro-tool route above only meets the Alaska and LaSalle Parish issues.

## Section 1 — Visualization (`src/s1_visualization.py`)

| Exercise step | Course tool | Script function | Output |
|---|---|---|---|
| Ex 3: histogram of `voter_turnout_2020` with mean/median/normal | Chart → Histogram (Show mean, median, normal distribution) | `histogram()` | `figures/s1_turnout_hist.png` |
| Ex 3: skewness of the key variables | Histogram statistics panel | `skewness_table()` | `results/s1_skewness.csv` |
| Ex 3: standard-deviation choropleth, diverging ramp | Symbology → Graduated Colors → Standard Deviation, 1 SD, Red-Blue | `choropleth()` (matplotlib) and `pro_layout()` (arcpy.mp, `GraduatedColorsRenderer`, `classificationMethod = "StandardDeviation"`) | `figures/s1_turnout_choropleth.png`, `figures/s1_turnout_layout.png`, `outputs/s1_visualization.aprx` |
| Ex 3: box plot by state | Chart → Box Plot, Category = `state_abbr` | `state_boxplot()` | `figures/s1_state_boxplot.png`, `results/s1_state_turnout.csv` |
| Ex 3: scatterplot matrix with R² | Chart → Scatter Plot Matrix (Show R²) | `scatter_matrix()` | `figures/s1_scatter_matrix.png`, `results/s1_scatter_r2.csv` |

`pro_layout()` is the one place that builds a real Pro map and layout headlessly: `aprx.createMap`, `addDataFromPath`, symbology edits, `createLayout`, `createMapFrame`, `createMapSurroundElement("LEGEND")`, `exportToPNG`. Open `outputs/s1_visualization.aprx` in Pro to see it.

## Section 2 — Prediction (`src/s2_prediction.py`)

| Exercise step | Course tool | Script | Output |
|---|---|---|---|
| Train baseline model | Forest-based and Boosted Classification and Regression, Train only, `voter_turnout` ~ median age, per-capita income, bachelor's % | `run_forest("M1_baseline")` | `outputs/logs/s2_forest_M1_baseline.txt` (Model Out of Bag Errors, Validation diagnostics, Range diagnostics) |
| Add many variables | same tool, 22 explanatory variables incl. `voter_laws` (categorical) | `run_forest("M2_many")` | `results/s2_model_comparison.csv` |
| Add distance features | Explanatory Training Distance Features = `Cities5` … `Cities10` | `run_forest("M3_distance")` | importance of `Cities*` rows in `results/s2_variable_importance.csv` |
| Refine by importance | drop variables below a chosen importance | threshold `IMPORTANCE_THRESHOLD` (3 % of total mean importance) | `results/s2_prediction.json → refined_variables` |
| Number of Runs for Validation | Validation Options → 10 runs | `number_validation_runs=10`, `output_validation_table` | `*_validation` tables in `outputs/s2_prediction.gdb`; R² range in `s2_model_comparison.csv` |
| Predict to tracts | Predict to features = `Tracts_Iowa`, Match Explanatory Variables | `run_forest("M4_refined_tracts", predict_to=TRACTS, matching=...)` | `outputs/s2_prediction.gdb/M4_refined_tracts_pred`, `figures/s2_tract_prediction.png` |
| Explanatory Variable Range Diagnostics | tool messages | `parse_range_diagnostics()` | `results/s2_range_diagnostics.csv` (share of overlap, `+` = extrapolation) |
| Extra (not in course): spatial cross-validation | — | `leave_one_state_out()` | `results/s2_spatial_cv_by_state.csv`, `figures/s2_spatial_cv.png` |
| National county model, 2020 | — | `N1_baseline_2020`, `N2_many_2020` on `CountyElections2020` | `s2_model_comparison.csv` |

The course's `ElectionPartyVotes.ipynb` computes `state_votes_pcnt_diff` (statewide margin) for the county and tract layers; the shipped layers already contain it, so the notebook is not rerun.

## Section 3 — Suitability (`src/s3_suitability.py`)

| Exercise step | Course tool | Script | Output |
|---|---|---|---|
| Slope | Surface Parameters (slope, degrees) | `sa.SurfaceParameters(ELEV, "SLOPE", "QUADRATIC", ...)` | `outputs/s3_suitability.gdb/Slope` |
| Distance to major roads around the ridge | Distance Accumulation, barrier = `GreenMTRidge` | `sa.DistanceAccumulation(MajorRoads, GreenMTRidge)` | `Dist_MRoads`, `figures/s3_distance_barrier.png` |
| Transform criteria to 1–10 | Suitability Modeler → Transform (Rescale by Function / Unique categories) | `sa.RescaleByFunction` with `TfSmall`, `TfMSSmall`, `TfLogisticDecay`, …; `sa.Reclassify` for land use | `T_*` rasters, `figures/s3_criteria_grid.png`, `results/s3_criteria.csv` (which values are the course's, which are mine) |
| Submodels and weights | Suitability Modeler → Weight, Calculate | weighted `sa.Times`/`sa.Plus` | `HQ_Building_Site`, `HQ_Energy_Independence`, `HQ_Access_to_Amenities`, `HQ_Environmental_Concerns`; `results/s3_submodel_ranges.csv` compares the range with the course's rasters in `Headquarters_Siting.gdb` |
| Final model | Suitability Modeler with submodels as criteria | linear rescale of each submodel to 1–10, equal weights | `HQ_Suitability`, `figures/s3_submodels.png` |
| Locate | Suitability Modeler → Locate (Locate Regions) | `sa.LocateRegions(final, 2, "SQUARE_KILOMETERS", 1, "CIRCLE", 0, 50, "HIGHEST_AVERAGE_VALUE")` | `Best_Region`, `results/s3_suitability.json`, `figures/s3_suitability_map.png` |

`Suitability.sam` is the course's saved Suitability Modeler state. `read_sam()` pulls the transformation function, its parameters and the weight for every criterion of the three submodels it stores; those are used verbatim. The Building Site submodel and the final combination are not in the file, so their values are flagged `assumption` in `results/s3_criteria.csv`.

## Section 4 — Pattern detection and space-time (`src/s4_patterns_spacetime.py`)

| Exercise step | Course tool | Script | Output |
|---|---|---|---|
| Hot spots, default distance | Hot Spot Analysis (Getis-Ord Gi*), Fixed distance band (blank = tool default), FDR | first `arcpy.stats.HotSpots` | `outputs/logs/s4_hotspots_fixed.txt` (distance band, neighbour counts), `HotSpots_fixed` |
| Hot spots, contiguity | same, Contiguity edges corners, FDR | second `HotSpots` | `HotSpots_contiguity`, `results/s4_hotspot_counts.csv`, `s4_hotspot_fixed_vs_contiguity.csv` |
| Cluster and outlier analysis | Cluster and Outlier Analysis (Anselin Local Moran's I), contiguity, 999 permutations, FDR | `arcpy.stats.ClustersOutliers` | `ClustersOutliers_contiguity`, `results/s4_lisa_counts.csv` |
| Space-time cube | Create Space Time Cube From Defined Locations, related table `SNAP_County_Rates2010_2019`, 1 year | `arcpy.stpm.CreateSpaceTimeCubeDefinedLocations` | `outputs/s4_snap_cube.nc`, `outputs/logs/s4_create_cube.txt` |
| Emerging hot spots | Emerging Hot Spot Analysis, contiguity, 1 time step | `arcpy.stpm.EmergingHotSpotAnalysis` | `EmergingHotSpots`, `results/s4_ehsa_counts.csv`, `figures/s4_spacetime.png` |
| Local outliers | Local Outlier Analysis, contiguity, 499 permutations | `arcpy.stpm.LocalOutlierAnalysis` | `LocalOutliers`, `results/s4_loa_counts.csv` |

Visualize the cube in Pro with *Visualize Space Time Cube in 3D* (not scripted here; the `.nc` file is in `outputs/`).

## Section 5 — Deep learning

Not reproduced. The course's Section 5 (object detection / pixel classification with the Image Analyst deep learning tools) needs its own data download and GPU time, so it is out of scope for this repo.
