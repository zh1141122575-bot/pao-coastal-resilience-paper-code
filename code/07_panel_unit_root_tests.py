#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller


from common import load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("07_panel_unit_root_tests")

COL_YEAR = "年份"
COL_CITY = "城市"
COL_Y = "Resilience_Score"
PAO_COL = "PAO_baseline"
PAO_DIMENSION_COLS = [
    "PAO_baseline_Economic",
    "PAO_baseline_Ecological",
    "PAO_baseline_Social",
]

ADF_MAXLAG = 1
N_SIM = 1000
RANDOM_SEED = 20260522
RUN_FIRST_DIFFERENCE_WHEN_MIXED = False


def normalize_city(name: object) -> str:
    text = str(name).strip()
    return text if text.endswith("市") else f"{text}市"


def load_panel() -> pd.DataFrame:
    y = load_model_base()

    pao = read_pao("baseline_main_balanced", [COL_CITY, COL_YEAR, PAO_COL] + PAO_DIMENSION_COLS)

    df = y.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != y.shape[0]:
        raise ValueError(f"Merged rows {df.shape[0]} do not match resilience rows {y.shape[0]}.")

    return df.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def city_adf_stats(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    rows = []
    for city, group in df.groupby(COL_CITY):
        series = group.sort_values(COL_YEAR)[variable].astype(float).to_numpy()
        try:
            stat, p_value, used_lag, nobs, *_ = adfuller(
                series,
                maxlag=ADF_MAXLAG,
                regression="c",
                autolag=None,
            )
        except Exception as exc:
            stat, p_value, used_lag, nobs = np.nan, np.nan, np.nan, np.nan
            rows.append({"city": city, "error": str(exc)})
            continue
        rows.append(
            {
                "city": city,
                "variable": variable,
                "adf_stat": stat,
                "adf_p_value": p_value,
                "used_lag": used_lag,
                "nobs": nobs,
                "error": "",
            }
        )
    return pd.DataFrame(rows)


def fisher_adf_p(adf_p_values: pd.Series) -> tuple[float, float, int]:
    pvals = adf_p_values.dropna().clip(lower=1e-12, upper=1.0)
    statistic = float(-2.0 * np.log(pvals).sum())
    df_chi = int(2 * len(pvals))
    p_value = float(stats.chi2.sf(statistic, df_chi))
    return statistic, p_value, len(pvals)


def city_pp_stats(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    rows = []
    for city, group in df.groupby(COL_CITY):
        series = group.sort_values(COL_YEAR)[variable].astype(float).to_numpy()
        if len(series) < 5 or np.nanstd(series) == 0:
            rows.append(
                {
                    "city": city,
                    "variable": variable,
                    "pp_stat": np.nan,
                    "pp_p_value": np.nan,
                    "hac_lags": np.nan,
                    "nobs": len(series),
                    "error": "too few observations or constant series",
                }
            )
            continue

        y = series[1:]
        y_lag = series[:-1]
        dy = y - y_lag
        x = sm.add_constant(y_lag)
        try:
            result = sm.OLS(dy, x).fit()
            hac_lags = max(1, int(np.floor(4 * (len(series) / 100) ** (2 / 9))))
            hac_cov = cov_hac(result, nlags=hac_lags)
            se = float(np.sqrt(hac_cov[1, 1]))
            pp_stat = float(result.params[1] / se)
            pp_p = float(mackinnonp(pp_stat, regression="c", N=1))
            error = ""
        except Exception as exc:
            pp_stat, pp_p, hac_lags, error = np.nan, np.nan, np.nan, str(exc)

        rows.append(
            {
                "city": city,
                "variable": variable,
                "pp_stat": pp_stat,
                "pp_p_value": pp_p,
                "hac_lags": hac_lags,
                "nobs": len(series) - 1,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def fisher_pp_p(pp_p_values: pd.Series) -> tuple[float, float, int]:
    pvals = pp_p_values.dropna().clip(lower=1e-12, upper=1.0)
    statistic = float(-2.0 * np.log(pvals).sum())
    df_chi = int(2 * len(pvals))
    p_value = float(stats.chi2.sf(statistic, df_chi))
    return statistic, p_value, len(pvals)


def build_simulated_unit_root_panel(n_cities: int, years: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    t = len(years)
    for city_idx in range(n_cities):
        innovations = rng.normal(loc=0.0, scale=1.0, size=t)
        values = np.cumsum(innovations)
        for year, value in zip(years, values):
            rows.append({COL_CITY: f"city_{city_idx:03d}", COL_YEAR: int(year), "sim_y": value})
    return pd.DataFrame(rows)


def ips_statistic(df: pd.DataFrame, variable: str) -> float:
    stats_df = city_adf_stats(df, variable)
    return float(stats_df["adf_stat"].dropna().mean())


def ips_p_value(observed_stat: float, n_cities: int, years: np.ndarray, rng: np.random.Generator) -> float:
    simulated = []
    for _ in range(N_SIM):
        sim_df = build_simulated_unit_root_panel(n_cities, years, rng)
        simulated.append(ips_statistic(sim_df.rename(columns={"sim_y": "value"}), "value"))
    simulated = np.array(simulated, dtype=float)
    return float(np.mean(simulated <= observed_stat))


def pooled_adf_t_stat(df: pd.DataFrame, variable: str) -> float:
    d = df.sort_values([COL_CITY, COL_YEAR]).copy()
    d["y"] = d[variable].astype(float)
    d["dy"] = d.groupby(COL_CITY)["y"].diff()
    d["y_lag1"] = d.groupby(COL_CITY)["y"].shift(1)
    d["dy_lag1"] = d.groupby(COL_CITY)["dy"].shift(1)
    d = d.dropna(subset=["dy", "y_lag1", "dy_lag1"]).copy()
    result = smf.ols(f"dy ~ y_lag1 + dy_lag1 + C({COL_CITY})", data=d).fit()
    return float(result.tvalues["y_lag1"])


def llc_p_value(observed_t: float, n_cities: int, years: np.ndarray, rng: np.random.Generator) -> float:
    simulated = []
    for _ in range(N_SIM):
        sim_df = build_simulated_unit_root_panel(n_cities, years, rng)
        simulated.append(pooled_adf_t_stat(sim_df, "sim_y"))
    simulated = np.array(simulated, dtype=float)
    return float(np.mean(simulated <= observed_t))


def conclusion_from_tests(llc_p: float, ips_p: float, fisher_adf_p_value: float, fisher_pp_p_value: float) -> str:
    p_values = [llc_p, ips_p, fisher_adf_p_value, fisher_pp_p_value]
    reject_count = sum(p < 0.05 for p in p_values if np.isfinite(p))
    if reject_count >= 3:
        return "Stationary at level"
    if reject_count >= 1:
        return "Mixed evidence; check transformed specification"
    return "Unit-root concern; use transformed robustness"


def run_tests(df: pd.DataFrame, variable: str, label: str) -> tuple[dict, pd.DataFrame]:
    years = np.array(sorted(df[COL_YEAR].unique()))
    n_cities = int(df[COL_CITY].nunique())
    variable_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(variable))
    rng = np.random.default_rng(RANDOM_SEED + variable_seed % 10000)

    adf_by_city = city_adf_stats(df, variable)
    fisher_stat, fisher_p, n_adf = fisher_adf_p(adf_by_city["adf_p_value"])
    pp_by_city = city_pp_stats(df, variable)
    fisher_pp_stat, fisher_pp_p_value, n_pp = fisher_pp_p(pp_by_city["pp_p_value"])
    ips_stat = ips_statistic(df, variable)
    ips_p = ips_p_value(ips_stat, n_cities, years, rng)
    llc_stat = pooled_adf_t_stat(df, variable)
    llc_p = llc_p_value(llc_stat, n_cities, years, rng)

    row = {
        "Variable": label,
        "Transformation": "Level",
        "LLC statistic": llc_stat,
        "LLC p-value": llc_p,
        "IPS statistic": ips_stat,
        "IPS p-value": ips_p,
        "Fisher-ADF statistic": fisher_stat,
        "Fisher-ADF p-value": fisher_p,
        "Fisher-PP statistic": fisher_pp_stat,
        "Fisher-PP p-value": fisher_pp_p_value,
        "City-level ADF tests": n_adf,
        "City-level PP tests": n_pp,
        "Conclusion": conclusion_from_tests(llc_p, ips_p, fisher_p, fisher_pp_p_value),
    }
    city_stats = adf_by_city.merge(
        pp_by_city,
        on=["city", "variable"],
        how="outer",
        suffixes=("_adf", "_pp"),
    )
    return row, city_stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_panel()

    variables = [
        (PAO_COL, "PAO"),
        ("PAO_baseline_Economic", "Economic PAO"),
        ("PAO_baseline_Ecological", "Ecological PAO"),
        ("PAO_baseline_Social", "Social PAO"),
        (COL_Y, "Resilience score"),
    ]

    summary_rows = []
    city_tables = []
    for variable, label in variables:
        row, adf_city = run_tests(df, variable, label)
        summary_rows.append(row)
        city_tables.append(adf_city)

        if RUN_FIRST_DIFFERENCE_WHEN_MIXED and (
            "Unit-root concern" in row["Conclusion"] or "Mixed evidence" in row["Conclusion"]
        ):
            diff_col = f"D_{variable}"
            df[diff_col] = df.groupby(COL_CITY)[variable].diff()
            diff_df = df.dropna(subset=[diff_col]).copy()
            diff_row, diff_adf_city = run_tests(diff_df, diff_col, f"Delta {label}")
            diff_row["Transformation"] = "First difference"
            summary_rows.append(diff_row)
            city_tables.append(diff_adf_city)

    summary = pd.DataFrame(summary_rows)
    city_level = pd.concat(city_tables, ignore_index=True)

    output_xlsx = OUTPUT_DIR / "panel_unit_root_tests.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Table_panel_unit_root", index=False)
        city_level.to_excel(writer, sheet_name="city_level_adf", index=False)


    note = f"""# Panel unit-root diagnostics

Variables tested: baseline PAO, Economic PAO, Ecological PAO, Social PAO, and resilience score.

Panel: {df[COL_CITY].nunique()} cities, {df[COL_YEAR].min()}-{df[COL_YEAR].max()}.

Tests:
- LLC: pooled ADF regression with city fixed effects and one lagged difference. P-values are finite-sample Monte Carlo p-values under a unit-root null.
- IPS: mean of city-level ADF t-statistics. P-values are finite-sample Monte Carlo p-values under a unit-root null.
- Fisher-ADF: Maddala-Wu Fisher combination of city-level ADF p-values from statsmodels.
- Fisher-PP: Fisher combination of city-level Phillips-Perron-style HAC unit-root p-values.

Given the relatively short time dimension of the city-level panel, these panel unit-root tests are used as diagnostic evidence rather than definitive proof of stationarity.

ADF settings: intercept only, maxlag={ADF_MAXLAG}, no autolag. PP settings: intercept only with HAC standard errors. Simulations: {N_SIM}, seed={RANDOM_SEED}.
"""
    (OUTPUT_DIR / "panel_unit_root_tests_note.md").write_text(note, encoding="utf-8")

    print(f"Saved panel unit-root outputs to: {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
