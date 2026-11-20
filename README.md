# Public Attention to the Ocean (PAO) and coastal urban resilience

Code for the article "A big data-driven Public Attention to the Ocean (PAO) index could be a complementary indicator for diagnosing the resilience of China's coastal cities" (Humanities and Social Sciences Communications).

## Codes

The Python code used for all analyses is openly available at https://github.com/zh1141122575-bot/pao-coastal-resilience-paper-code. The scripts read the two data files provided as Supplementary Data 1 and Supplementary Data 2. The two files, `common.py` and `resilience_tools.py`, contain functions shared by the scripts.

The reported results were obtained on Linux (aarch64) with Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, statsmodels 0.15.0, scikit-learn 1.4.2, XGBoost 2.1.4, SHAP 0.49.1, openpyxl 3.1.5, matplotlib 3.10.9 and geopandas 1.1.4, as listed in `requirements.txt`. Each script produces the following results.

- Construction of the PAO index (Eqs. 1–2), Table S7 and Table S16 using `01_PAO_index.py`
- Coastal urban resilience index (AHP-EWM weighting and BPNN-PSO), Table S2 and Table S3 using `02_resilience_index.py`
- Fig. S3, Table S4 and Table S5 using `03_resilience_MC_alpha.py`
- Fig. S4 using `04_resilience_indicator_subsets.py`
- Fig. 1b, Fig. S5, Table S8 and Table S10 using `05_PDL_lag_selection.py`
- Table S9 using `06_PDL_alternative_PAO.py`
- Table S11 using `07_panel_unit_root_tests.py`
- Table S12 using `08_PDL_transformed_variables.py`
- Table S13 and Table S14 using `09_reverse_lag_placebo.py`
- Table S15 and the region-year fixed-effects PDL using `10_spatial_autocorrelation.py`
- Fig. 3 and Tables S17–S23 using `11_XGBoost_SHAP.py`
- Table S24 and Table S25 using `12_XGBoost_alternative_resilience.py`
- Fig. 2 (lag-adjusted PAO–resilience matching types) using `13_mismatch.py`
- All sub-figures of Fig. 1–3 and Fig. S3–S5 using `plot_figures.py`, and then combined in PowerPoint. The maps in Fig. 2 are based on the standard map of China (No. GS(2024)0650) from the National Geographic Information Public Service Platform, which is not included in the code files.

## Data availability

The cleaned datasets used in all analyses are provided as Supplementary Data. Supplementary Data 1 contains the annual Baidu Index values of the 44 ocean-related keywords (Table S6), and Supplementary Data 2 contains the values of the 14 coastal urban resilience indicators (Table S2) together with the resulting resilience scores, both for the 53 coastal cities from 2014 to 2024. The definitions, units, directions and weights of the resilience indicators are given in Table S2.

The original data underlying these datasets are not redistributed here. The annual Baidu Index values were obtained by summing the daily Baidu Index values of each keyword in each city, and the daily series are not provided because the platform restricts automated collection and redistribution of Baidu Index data. They can be queried at Baidu Index (https://index.baidu.com). The resilience indicators were compiled from statistical data published by government agencies, and their sources are listed below for reference. The share of tertiary industry, total retail sales of consumer goods, educational expenditure, number of hospitals and health institutions, number of persons insured for unemployment insurance, mobile phone users at year-end and centralized sewage treatment rate were obtained from the China City Statistical Yearbook of the National Bureau of Statistics of China (https://www.stats.gov.cn). Investment in rail and public transport, municipal public facilities investment, total urban water supply and green area were obtained from the China Urban Construction Statistical Yearbook of the Ministry of Housing and Urban-Rural Development (https://www.mohurd.gov.cn). Marine-related economic activity intensity was calculated from city GDP and provincial gross ocean product reported in the provincial statistical bulletins and the China Marine Economic Statistical Bulletin of the Ministry of Natural Resources (https://www.mnr.gov.cn). Green patents granted were obtained from the China National Intellectual Property Administration (https://www.cnipa.gov.cn), and local government governance attention was derived from the annual government work reports published on the official websites of the municipal governments.

The maps in Fig. 2 are based on the standard map of China (No. GS(2024)0650) from the National Geographic Information Public Service Platform.
