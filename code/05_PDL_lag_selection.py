#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


from common import COL_CITY, COL_YEAR, COL_Y, PAO_SCALE, load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("05_PDL_lag_selection")
PAO_COL = "PAO_baseline"
POLY_DEGREE = 2


ALMON_SPECS = [
    {"specification": "Almon PDL", "lag_window": "L1-L3", "lags": [1, 2, 3], "include_l0": "No", "purpose": "Short lag window"},
    {"specification": "Almon PDL", "lag_window": "L1-L4", "lags": [1, 2, 3, 4], "include_l0": "No", "purpose": "Main final specification candidate"},
    {"specification": "Almon PDL", "lag_window": "L1-L5", "lags": [1, 2, 3, 4, 5], "include_l0": "No", "purpose": "Longer lag window comparable with early unrestricted PDL"},
    {"specification": "Almon PDL", "lag_window": "L0-L3", "lags": [0, 1, 2, 3], "include_l0": "Yes", "purpose": "Contemporaneous diagnostic"},
    {"specification": "Almon PDL", "lag_window": "L0-L4", "lags": [0, 1, 2, 3, 4], "include_l0": "Yes", "purpose": "Robustness including current PAO"},
]


def normalize_city(name: object) -> str:
    text = str(name).strip()
    return text if text.endswith("市") else f"{text}市"


def p_value_from_z(value: float, se: float) -> float:
    if not np.isfinite(value) or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.0 * (1.0 - stats.norm.cdf(abs(value / se)))


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def load_data() -> pd.DataFrame:
    y = load_model_base()
    pao = read_pao("baseline_main_balanced", [COL_CITY, COL_YEAR, PAO_COL])

    df = y.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != y.shape[0]:
        raise ValueError(f"Merged rows {df.shape[0]} do not match resilience rows {y.shape[0]}.")
    df["PAO_scaled"] = df[PAO_COL] / PAO_SCALE
    return df


def add_lags(df: pd.DataFrame, lags: list[int], required_lags: list[int] | None = None) -> tuple[pd.DataFrame, list[str]]:
    d = df.sort_values([COL_CITY, COL_YEAR]).copy()
    lags_to_create = sorted(set(lags + (required_lags or [])))
    created_cols = []
    for lag in lags_to_create:
        col = f"PAO_L{lag}"
        d[col] = d.groupby(COL_CITY)["PAO_scaled"].shift(lag)
        created_cols.append(col)
    lag_cols = [f"PAO_L{lag}" for lag in lags]
    required_cols = [f"PAO_L{lag}" for lag in (required_lags or lags)]
    d = d.dropna(subset=[COL_Y] + required_cols).copy()
    return d, lag_cols


def almon_weights(lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lag_values = np.array(lags, dtype=float)
    centered = lag_values - lag_values.mean()
    return lag_values, centered


def fit_almon_pdl(df: pd.DataFrame, lags: list[int], required_lags: list[int] | None = None) -> dict:
    d, lag_cols = add_lags(df, lags, required_lags=required_lags)
    lag_values, centered = almon_weights(lags)

    z_cols = []
    for degree in range(POLY_DEGREE + 1):
        z_col = f"Z{degree}"
        z_cols.append(z_col)
        d[z_col] = 0.0
        weights = centered**degree
        for pos, col in enumerate(lag_cols):
            d[z_col] += weights[pos] * d[col]

    formula = f"{COL_Y} ~ {' + '.join(z_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})

    coef = result.params.reindex(z_cols).fillna(0.0)
    vcov = result.cov_params().loc[z_cols, z_cols]
    lag_mean = lag_values.mean()

    def lag_effect(lag: float) -> float:
        kh = lag - lag_mean
        return float(coef["Z0"] + coef["Z1"] * kh + coef["Z2"] * kh * kh)

    def lag_se(lag: float) -> float:
        kh = lag - lag_mean
        g = np.array([1.0, kh, kh * kh])
        var = float(g @ vcov.values @ g.T)
        return float(np.sqrt(max(var, 0.0)))

    discrete_effects = pd.DataFrame(
        {
            "lag": lags,
            "marginal_effect": [lag_effect(lag) for lag in lags],
            "std_error": [lag_se(lag) for lag in lags],
        }
    )
    discrete_effects["z"] = discrete_effects["marginal_effect"] / discrete_effects["std_error"]
    discrete_effects["p_value"] = [
        p_value_from_z(v, se)
        for v, se in zip(discrete_effects["marginal_effect"], discrete_effects["std_error"])
    ]
    discrete_effects["stars"] = discrete_effects["p_value"].map(stars)

    grid = np.linspace(float(min(lags)), float(max(lags)), 301)
    grid_effects = np.array([lag_effect(lag) for lag in grid])
    cumulative_effect = float(discrete_effects["marginal_effect"].sum())
    direction = np.sign(cumulative_effect) if cumulative_effect != 0 else np.sign(grid_effects[np.argmax(np.abs(grid_effects))])
    peak_idx = int(np.argmax(grid_effects) if direction >= 0 else np.argmin(grid_effects))
    peak_lag = float(grid[peak_idx])
    peak_effect = float(grid_effects[peak_idx])
    peak_se = lag_se(peak_lag)
    peak_p = p_value_from_z(peak_effect, peak_se)

    cumulative_weights = np.array([float(np.sum(centered**degree)) for degree in range(POLY_DEGREE + 1)])
    cumulative_se = float(np.sqrt(max(float(cumulative_weights @ vcov.values @ cumulative_weights.T), 0.0)))
    cumulative_p = p_value_from_z(cumulative_effect, cumulative_se)
    joint_p = float(result.wald_test("Z0 = 0, Z1 = 0, Z2 = 0", scalar=True).pvalue)

    return {
        "result": result,
        "data": d,
        "lag_effects": discrete_effects,
        "nobs": int(result.nobs),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "adj_r2": float(result.rsquared_adj),
        "joint_p": joint_p,
        "peak_lag": peak_lag,
        "peak_effect": peak_effect,
        "peak_p": peak_p,
        "cumulative_effect": cumulative_effect,
        "cumulative_se": cumulative_se,
        "cumulative_p": cumulative_p,
        "coef": coef,
        "vcov": vcov,
        "lag_mean": lag_mean,
    }


def fit_unrestricted_pdl(df: pd.DataFrame, lags: list[int]) -> dict:
    d, lag_cols = add_lags(df, lags)
    formula = f"{COL_Y} ~ {' + '.join(lag_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})
    lag_effects = pd.DataFrame(
        {
            "lag": lags,
            "term": lag_cols,
            "coefficient": [float(result.params[col]) for col in lag_cols],
            "std_error": [float(result.bse[col]) for col in lag_cols],
            "p_value": [float(result.pvalues[col]) for col in lag_cols],
        }
    )
    lag_effects["stars"] = lag_effects["p_value"].map(stars)
    cumulative_effect = float(lag_effects["coefficient"].sum())
    cov = result.cov_params().loc[lag_cols, lag_cols].values
    ones = np.ones(len(lag_cols))
    cumulative_se = float(np.sqrt(max(float(ones @ cov @ ones.T), 0.0)))
    cumulative_p = p_value_from_z(cumulative_effect, cumulative_se)
    joint_p = float(result.wald_test(", ".join([f"{col} = 0" for col in lag_cols]), scalar=True).pvalue)

    wald_tests = []
    for test_name, constraint in [
        ("Additional fifth-year lag", "PAO_L5 = 0"),
        ("Additional fourth- and fifth-year lags", "PAO_L4 = 0, PAO_L5 = 0"),
    ]:
        if all(term.strip().split(" ")[0] in lag_cols for term in constraint.split(",")):
            test = result.wald_test(constraint, scalar=True)
            wald_tests.append(
                {
                    "Model": "Unrestricted PDL L1-L5",
                    "Test": test_name,
                    "Null hypothesis": constraint,
                    "Statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "stars": stars(float(test.pvalue)),
                    "Interpretation": (
                        "Reject: added lag(s) provide extra explanatory power"
                        if float(test.pvalue) < 0.1
                        else "Do not reject: added lag(s) do not provide clear extra explanatory power"
                    ),
                }
            )

    return {
        "result": result,
        "data": d,
        "lag_effects": lag_effects,
        "wald_tests": pd.DataFrame(wald_tests),
        "nobs": int(result.nobs),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "adj_r2": float(result.rsquared_adj),
        "joint_p": joint_p,
        "cumulative_effect": cumulative_effect,
        "cumulative_se": cumulative_se,
        "cumulative_p": cumulative_p,
    }


def fit_cumulative_window(df: pd.DataFrame, max_window: int) -> dict:
    lags = list(range(1, max_window + 1))
    d, lag_cols = add_lags(df, lags)
    cum_col = f"CumPAO_L1_L{max_window}"
    d[cum_col] = d[lag_cols].sum(axis=1)
    formula = f"{COL_Y} ~ {cum_col} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})
    coef = float(result.params[cum_col])
    se = float(result.bse[cum_col])
    p_value = float(result.pvalues[cum_col])
    return {
        "window": f"L1-L{max_window}",
        "nobs": int(result.nobs),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "adj_r2": float(result.rsquared_adj),
        "coefficient": coef,
        "std_error": se,
        "p_value": p_value,
        "stars": stars(p_value),
    }


def fit_almon_period_stability(df: pd.DataFrame, start_year: int, end_year: int, label: str) -> dict:
    period_df = df[(df[COL_YEAR] >= start_year) & (df[COL_YEAR] <= end_year)].copy()
    fitted = fit_almon_pdl(period_df, [1, 2, 3, 4])
    direction = (
        "Positive"
        if fitted["cumulative_effect"] > 0
        else "Negative"
        if fitted["cumulative_effect"] < 0
        else "Near zero"
    )
    return {
        "Period": label,
        "Start year": start_year,
        "End year": end_year,
        "N obs after lags": fitted["nobs"],
        "AIC": fitted["aic"],
        "BIC": fitted["bic"],
        "Adj. R2": fitted["adj_r2"],
        "Joint p": fitted["joint_p"],
        "Peak lag": fitted["peak_lag"],
        "Peak effect": fitted["peak_effect"],
        "Peak p": fitted["peak_p"],
        "Cumulative effect": fitted["cumulative_effect"],
        "Cumulative p": fitted["cumulative_p"],
        "Cumulative direction": direction,
        "Interpretation": (
            "Direction remains positive and lag terms are jointly significant"
            if fitted["cumulative_effect"] > 0 and fitted["joint_p"] < 0.1
            else "Direction remains positive but statistical strength is weaker"
            if fitted["cumulative_effect"] > 0
            else "Direction differs from the main specification"
        ),
    }


def pdl_curves(fitted: dict, lags: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continuous marginal association w(h) and cumulative association S(h) with 95% CIs (Fig. 1b)."""
    coef = fitted["coef"].values
    vcov = fitted["vcov"].values
    kbar = fitted["lag_mean"]
    lo = float(min(lags))
    grid = np.linspace(lo, float(max(lags)), 301)
    marginal, cumulative = [], []
    for h in grid:
        g = np.array([1.0, h - kbar, (h - kbar) ** 2])
        est, se = float(g @ coef), float(np.sqrt(max(float(g @ vcov @ g), 0.0)))
        marginal.append({"lag": h, "effect": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})
        w = np.array([h - lo, ((h - kbar) ** 2 - (lo - kbar) ** 2) / 2.0, ((h - kbar) ** 3 - (lo - kbar) ** 3) / 3.0])
        est, se = float(w @ coef), float(np.sqrt(max(float(w @ vcov @ w), 0.0)))
        cumulative.append({"lag": h, "effect": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})
    return pd.DataFrame(marginal), pd.DataFrame(cumulative)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    lag_selection_rows = []
    almon_lag_effect_frames = []
    almon_results_by_window = {}

    for spec in ALMON_SPECS:
        fitted = fit_almon_pdl(df, spec["lags"])
        almon_results_by_window[spec["lag_window"]] = fitted
        lag_selection_rows.append(
            {
                "Specification": spec["specification"],
                "Lag window": spec["lag_window"],
                "Include L0?": spec["include_l0"],
                "Purpose": spec["purpose"],
                "N obs": fitted["nobs"],
                "AIC": fitted["aic"],
                "BIC": fitted["bic"],
                "Adj. R2": fitted["adj_r2"],
                "Joint p": fitted["joint_p"],
                "Peak lag": fitted["peak_lag"],
                "Peak effect": fitted["peak_effect"],
                "Peak p": fitted["peak_p"],
                "Cumulative effect": fitted["cumulative_effect"],
                "Cumulative p": fitted["cumulative_p"],
                "Significance": f"Joint p={fitted['joint_p']:.3g}{stars(fitted['joint_p'])}; cumulative p={fitted['cumulative_p']:.3g}{stars(fitted['cumulative_p'])}",
            }
        )
        lag_effects = fitted["lag_effects"].copy()
        lag_effects.insert(0, "Lag window", spec["lag_window"])
        almon_lag_effect_frames.append(lag_effects)

    table_1 = pd.DataFrame(lag_selection_rows)
    almon_lag_effects = pd.concat(almon_lag_effect_frames, ignore_index=True)

    common_sample_rows = []
    common_sample_specs = [
        {
            "Comparison group": "No L0 common sample",
            "Lag window": "L1-L3",
            "lags": [1, 2, 3],
            "required_lags": [1, 2, 3, 4, 5],
        },
        {
            "Comparison group": "No L0 common sample",
            "Lag window": "L1-L4",
            "lags": [1, 2, 3, 4],
            "required_lags": [1, 2, 3, 4, 5],
        },
        {
            "Comparison group": "No L0 common sample",
            "Lag window": "L1-L5",
            "lags": [1, 2, 3, 4, 5],
            "required_lags": [1, 2, 3, 4, 5],
        },
        {
            "Comparison group": "L0-inclusive common sample",
            "Lag window": "L0-L3",
            "lags": [0, 1, 2, 3],
            "required_lags": [0, 1, 2, 3, 4],
        },
        {
            "Comparison group": "L0-inclusive common sample",
            "Lag window": "L0-L4",
            "lags": [0, 1, 2, 3, 4],
            "required_lags": [0, 1, 2, 3, 4],
        },
    ]
    for spec in common_sample_specs:
        fitted = fit_almon_pdl(df, spec["lags"], required_lags=spec["required_lags"])
        common_sample_rows.append(
            {
                "Comparison group": spec["Comparison group"],
                "Specification": "Almon PDL",
                "Lag window": spec["Lag window"],
                "Required complete lags": ",".join([f"L{lag}" for lag in spec["required_lags"]]),
                "N obs": fitted["nobs"],
                "AIC": fitted["aic"],
                "BIC": fitted["bic"],
                "Adj. R2": fitted["adj_r2"],
                "Joint p": fitted["joint_p"],
                "Peak lag": fitted["peak_lag"],
                "Peak effect": fitted["peak_effect"],
                "Peak p": fitted["peak_p"],
                "Cumulative effect": fitted["cumulative_effect"],
                "Cumulative p": fitted["cumulative_p"],
            }
        )
    table_1_common = pd.DataFrame(common_sample_rows)

    unrestricted = fit_unrestricted_pdl(df, [1, 2, 3, 4, 5])
    final_window = "L1-L4"
    final_almon = almon_results_by_window[final_window]
    table_2 = pd.DataFrame(
        [
            {
                "Model type": "Unrestricted PDL",
                "Lag window": "L1-L5",
                "Purpose": "Exploratory lag diagnosis",
                "N obs": unrestricted["nobs"],
                "AIC": unrestricted["aic"],
                "BIC": unrestricted["bic"],
                "Adj. R2": unrestricted["adj_r2"],
                "Joint p": unrestricted["joint_p"],
                "Cumulative effect": unrestricted["cumulative_effect"],
                "Cumulative p": unrestricted["cumulative_p"],
                "Main finding": "Tests lag-specific coefficients without smoothness constraint.",
            },
            {
                "Model type": "Centered Almon PDL",
                "Lag window": final_window,
                "Purpose": "Final model candidate used for main figures/scenario simulation",
                "N obs": final_almon["nobs"],
                "AIC": final_almon["aic"],
                "BIC": final_almon["bic"],
                "Adj. R2": final_almon["adj_r2"],
                "Joint p": final_almon["joint_p"],
                "Cumulative effect": final_almon["cumulative_effect"],
                "Cumulative p": final_almon["cumulative_p"],
                "Main finding": "Smooths lag effects and reduces instability from correlated annual PAO lags.",
            },
        ]
    )

    unrestricted_lag_effects = unrestricted["lag_effects"].copy()
    unrestricted_wald_tests = unrestricted["wald_tests"].copy()

    table_3 = pd.DataFrame([fit_cumulative_window(df, window) for window in range(1, 6)])
    table_3["Window interpretation"] = table_3.apply(
        lambda row: (
            "Significant cumulative association"
            if row["p_value"] < 0.1
            else "Cumulative association not statistically significant"
        ),
        axis=1,
    )

    rolling_periods = [(year, year + 7, f"{year}-{year + 7}") for year in range(2014, 2018)]
    split_periods = [
        (2014, 2020, "Split A: 2014-2020"),
        (2018, 2024, "Split B: 2018-2024"),
    ]
    table_4 = pd.DataFrame(
        [fit_almon_period_stability(df, start, end, label) for start, end, label in rolling_periods]
    )
    table_5 = pd.DataFrame(
        [fit_almon_period_stability(df, start, end, label) for start, end, label in split_periods]
    )

    fig1b_marginal, fig1b_cumulative = pdl_curves(final_almon, [1, 2, 3, 4])

    output_xlsx = OUTPUT_DIR / "PDL_lag_selection_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        table_1.to_excel(writer, sheet_name="Table1_lag_window", index=False)
        table_1_common.to_excel(writer, sheet_name="Table1b_common_sample", index=False)
        almon_lag_effects.to_excel(writer, sheet_name="Almon_lag_effects", index=False)
        table_2.to_excel(writer, sheet_name="Table2_unrestricted_vs_almon", index=False)
        unrestricted_lag_effects.to_excel(writer, sheet_name="Unrestricted_L1_L5_effects", index=False)
        unrestricted_wald_tests.to_excel(writer, sheet_name="Wald_tests_extra_lags", index=False)
        table_3.to_excel(writer, sheet_name="Table3_cumulative_window", index=False)
        table_4.to_excel(writer, sheet_name="Table4_rolling_8yr", index=False)
        table_5.to_excel(writer, sheet_name="Table5_split_period", index=False)
        fig1b_marginal.to_excel(writer, sheet_name="Fig1b_marginal_curve", index=False)
        fig1b_cumulative.to_excel(writer, sheet_name="Fig1b_cumulative_curve", index=False)


    note = f"""# PDL lag-window selection diagnostics

Data source: `Coastal urban resilience.xlsx` (Resilience_Score) and `Baidu index.xlsx` (baseline PAO).

PAO (Eq. 2) is divided by {PAO_SCALE:,.0f}, so coefficients are per 100,000 PAO units. All models include city fixed effects and year fixed effects with city-clustered standard errors.

The Almon PDL specifications use a centered quadratic lag polynomial. The unrestricted PDL L1-L5 model is treated as an exploratory diagnostic, while L0-inclusive specifications are contemporaneous diagnostics.

Additional diagnostics include cluster-robust Wald tests for whether L4-L5 terms add explanatory power in the unrestricted L1-L5 PDL, cumulative PAO window tests for L1 through L5, and 8-year rolling/split-period stability checks using the L1-L4 centered Almon PDL.
"""
    (OUTPUT_DIR / "PDL_lag_selection_summary.md").write_text(note, encoding="utf-8")

    print(f"Saved lag-selection outputs to: {OUTPUT_DIR}")
    print("\nTable 1: Lag-window selection")
    print(table_1.to_string(index=False))
    print("\nTable 1b: Common-sample lag-window selection")
    print(table_1_common.to_string(index=False))
    print("\nTable 2: Unrestricted vs Almon")
    print(table_2.to_string(index=False))
    print("\nTable 2b: Wald tests for additional lags")
    print(unrestricted_wald_tests.to_string(index=False))
    print("\nTable 3: Cumulative-window comparison")
    print(table_3.to_string(index=False))
    print("\nTable 4: 8-year rolling stability")
    print(table_4.to_string(index=False))
    print("\nTable 5: Split-period stability")
    print(table_5.to_string(index=False))


if __name__ == "__main__":
    main()
