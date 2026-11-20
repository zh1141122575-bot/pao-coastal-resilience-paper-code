# Public Attention to the Ocean (PAO) and coastal urban resilience

This repository contains the Python code for the article "A big data-driven Public Attention to the Ocean (PAO) index could be a complementary indicator for diagnosing the resilience of China's coastal cities" (Humanities and Social Sciences Communications).

## Data

The code uses the two data files provided with the article as Supplementary Data. Supplementary Data 1 (`Baidu index.xlsx`) contains the annual Baidu Index values of the 44 ocean-related keywords, and Supplementary Data 2 (`Coastal urban resilience.xlsx`) contains the values of the 14 coastal urban resilience indicators and the resilience scores, both for the 53 coastal cities from 2014 to 2024. Before running the code, place both files in the root folder of this repository, next to the `code` folder.

## Running the code

Install the dependencies with `pip install -r requirements.txt` and run the scripts in the `code` folder in numerical order. `common.py` and `resilience_tools.py` contain the functions shared by the scripts. Results are written to `code/outputs`, or to the folder given by the environment variable `PAO_OUTPUT_DIR`.

The reported results were obtained on Linux (aarch64) with Python 3.10.12 and the package versions listed in `requirements.txt`. Random seeds are fixed in all scripts. The machine-learning results (Fig. 3 and Tables S17–S25) may differ slightly under other versions of scikit-learn or XGBoost or on other operating systems, so the same versions should be used to reproduce them exactly.

## Scripts and results

| Script | Results in the article |
|---|---|
| `01_PAO_index.py` | Construction of the PAO index (Eqs. 1–2), Tables S7 and S16 |
| `02_resilience_index.py` | Coastal urban resilience index (AHP-EWM weighting and BPNN-PSO), Tables S2 and S3 |
| `03_resilience_MC_alpha.py` | Fig. S3, Tables S4 and S5 |
| `04_resilience_indicator_subsets.py` | Fig. S4 |
| `05_PDL_lag_selection.py` | Fig. 1b, Fig. S5, Tables S8 and S10 |
| `06_PDL_alternative_PAO.py` | Table S9 |
| `07_panel_unit_root_tests.py` | Table S11 |
| `08_PDL_transformed_variables.py` | Table S12 |
| `09_reverse_lag_placebo.py` | Tables S13 and S14 |
| `10_spatial_autocorrelation.py` | Table S15 and the region-year fixed-effects PDL |
| `11_XGBoost_SHAP.py` | Fig. 3, Tables S17–S23 |
| `12_XGBoost_alternative_resilience.py` | Tables S24 and S25 |
| `13_mismatch.py` | Fig. 2 (lag-adjusted PAO–resilience matching types) |
| `plot_figures.py` | Sub-figures of Fig. 1–3 and Fig. S3–S5 |

The sub-figures were combined into the final figures in PowerPoint. `plot_figures.py` uses Arial fonts, which can be supplied through the environment variable `PAO_FONT_DIR`. The maps in Fig. 2 are based on the standard map of China (No. GS(2024)0650) from the National Geographic Information Public Service Platform, which is not included in this repository, and its path can be supplied through the environment variable `PAO_BASEMAP`.
