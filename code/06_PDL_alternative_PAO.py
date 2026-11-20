#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


from common import (
    COL_CITY, COL_YEAR, COL_Y, LAG_MAX, PAO_SCALE, POLY_DEGREE,
    load_model_base, output_dir, read_pao,
)

OUTPUT_DIR = output_dir("06_PDL_alternative_PAO")


PAO_VERSIONS = [
    {
        "construction": "Baseline PAO",
        "sheet": "baseline_main_balanced",
        "pao_col": "PAO_baseline",
        "dimension_cols": [
            "PAO_baseline_Economic",
            "PAO_baseline_Ecological",
            "PAO_baseline_Social",
        ],
    },
    {
        "construction": "Raw-sum PAO",
        "sheet": "R1_raw_sum",
        "pao_col": "PAO_raw",
        "dimension_cols": [],
    },
    {
        "construction": "Excluding national-political terms",
        "sheet": "R2_no_political",
        "pao_col": "PAO_R2_no_political",
        "dimension_cols": [
            "PAO_R2_no_political_Economic",
            "PAO_R2_no_political_Ecological",
            "PAO_R2_no_political_Social",
        ],
    },
    {
        "construction": "Adding market-sensitive terms",
        "sheet": "R3_market_sensitive",
        "pao_col": "PAO_R3_market_sensitive",
        "dimension_cols": [
            "PAO_R3_market_sensitive_Economic",
            "PAO_R3_market_sensitive_Ecological",
            "PAO_R3_market_sensitive_Social",
        ],
    },
    {
        "construction": "Adding tourism/event-sensitive terms",
        "sheet": "R4_tourism_event",
        "pao_col": "PAO_R4_tourism_event",
        "dimension_cols": [
            "PAO_R4_tourism_event_Economic",
            "PAO_R4_tourism_event_Ecological",
            "PAO_R4_tourism_event_Social",
        ],
    },
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


def fit_centered_almon_pdl(df_in: pd.DataFrame) -> dict:
    d = df_in.sort_values([COL_CITY, COL_YEAR]).copy()

    lag_cols = []
    for lag in range(1, LAG_MAX + 1):
        col = f"PAO_L{lag}"
        d[col] = d.groupby(COL_CITY)["PAO_scaled"].shift(lag)
        lag_cols.append(col)

    d = d.dropna(subset=lag_cols + [COL_Y]).copy()

    idx = np.arange(1, LAG_MAX + 1, dtype=float)
    kbar = idx.mean()
    centered_lags = idx - kbar

    for degree in range(POLY_DEGREE + 1):
        z_col = f"Z{degree}"
        d[z_col] = 0.0
        weights = centered_lags**degree
        for pos, lag in enumerate(range(1, LAG_MAX + 1)):
            d[z_col] += weights[pos] * d[f"PAO_L{lag}"]

    z_cols = [f"Z{degree}" for degree in range(POLY_DEGREE + 1)]
    formula = f"{COL_Y} ~ {' + '.join(z_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    model = smf.ols(formula, data=d)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})

    coef = result.params.reindex(z_cols).fillna(0.0)
    vcov = result.cov_params().loc[z_cols, z_cols]

    def lag_effect(lag: float) -> float:
        kh = lag - kbar
        return float(coef["Z0"] + coef["Z1"] * kh + coef["Z2"] * kh * kh)

    def lag_se(lag: float) -> float:
        kh = lag - kbar
        g = np.array([1.0, kh, kh * kh])
        var = float(g @ vcov.values @ g.T)
        return float(np.sqrt(max(var, 0.0)))

    lag_rows = []
    for lag in range(1, LAG_MAX + 1):
        effect = lag_effect(lag)
        se = lag_se(lag)
        p_value = p_value_from_z(effect, se)
        lag_rows.append(
            {
                "lag": lag,
                "marginal_effect": effect,
                "std_error": se,
                "z": effect / se if se > 0 else np.nan,
                "p_value": p_value,
                "stars": stars(p_value),
            }
        )

    lag_df = pd.DataFrame(lag_rows)

    a0, a1, a2 = coef["Z0"], coef["Z1"], coef["Z2"]
    raw_peak = np.nan if abs(a2) < 1e-20 else kbar - a1 / (2.0 * a2)
    lag_grid = np.linspace(1.0, float(LAG_MAX), 301)
    grid_effects = np.array([lag_effect(h) for h in lag_grid])
    cumulative = float(lag_df["marginal_effect"].sum())
    direction_sign = np.sign(cumulative) if cumulative != 0 else np.sign(grid_effects[np.argmax(np.abs(grid_effects))])

    if direction_sign >= 0:
        grid_idx = int(np.argmax(grid_effects))
        integer_peak_idx = int(lag_df["marginal_effect"].idxmax())
    else:
        grid_idx = int(np.argmin(grid_effects))
        integer_peak_idx = int(lag_df["marginal_effect"].idxmin())

    peak_lag = float(lag_grid[grid_idx])
    peak_effect = float(grid_effects[grid_idx])
    peak_se = lag_se(peak_lag)
    peak_p = p_value_from_z(peak_effect, peak_se)

    joint = result.wald_test("Z0 = 0, Z1 = 0, Z2 = 0", scalar=True)
    joint_p = float(joint.pvalue)

    cumulative_cov_weights = []
    for degree in range(POLY_DEGREE + 1):
        cumulative_cov_weights.append(float(np.sum(centered_lags**degree)))
    g_sum = np.array(cumulative_cov_weights)
    cumulative_se = float(np.sqrt(max(float(g_sum @ vcov.values @ g_sum.T), 0.0)))
    cumulative_p = p_value_from_z(cumulative, cumulative_se)

    return {
        "result": result,
        "model_data": d,
        "lag_effects": lag_df,
        "coef": coef,
        "vcov": vcov,
        "raw_peak_lag": float(raw_peak) if np.isfinite(raw_peak) else np.nan,
        "peak_lag": peak_lag,
        "peak_effect": peak_effect,
        "peak_se": peak_se,
        "peak_p": peak_p,
        "integer_peak_lag": int(lag_df.loc[integer_peak_idx, "lag"]),
        "cumulative_effect": cumulative,
        "cumulative_se": cumulative_se,
        "cumulative_p": cumulative_p,
        "joint_p": joint_p,
        "nobs": int(result.nobs),
        "n_cities": int(d[COL_CITY].nunique()),
        "year_min": int(d[COL_YEAR].min()),
        "year_max": int(d[COL_YEAR].max()),
    }


def format_direction(value: float) -> str:
    if value > 0:
        return "Positive"
    if value < 0:
        return "Negative"
    return "Near zero"


def format_significance(joint_p: float, peak_p: float, cumulative_p: float) -> str:
    return (
        f"Joint p={joint_p:.3g}{stars(joint_p)}; "
        f"peak p={peak_p:.3g}{stars(peak_p)}; "
        f"cumulative p={cumulative_p:.3g}{stars(cumulative_p)}"
    )


def calculate_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = df[columns].astype(float).dropna().copy()
    x_with_const = pd.concat(
        [pd.Series(1.0, index=x.index, name="const"), x],
        axis=1,
    )

    rows = []
    for col_idx, col in enumerate(x_with_const.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(x_with_const.values, col_idx)
        tolerance = 1.0 / vif if np.isfinite(vif) and vif != 0 else np.nan
        rows.append(
            {
                "variable": col,
                "VIF": float(vif),
                "tolerance": float(tolerance) if np.isfinite(tolerance) else np.nan,
                "interpretation": (
                    "High multicollinearity"
                    if vif >= 10
                    else "Moderate multicollinearity"
                    if vif >= 5
                    else "Low multicollinearity"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_interpretation(construction: str, peak_lag: float, cumulative_p: float, joint_p: float) -> str:
    stable_peak = 1.5 <= peak_lag <= 2.6
    significant = (joint_p < 0.1) or (cumulative_p < 0.1)
    if construction == "Baseline PAO":
        return "Reference PDL specification using the domain-balanced main PAO."
    if construction == "Adding tourism/event-sensitive terms":
        if stable_peak and significant:
            return "Lag structure remains close to the baseline even after adding more volatile tourism/event attention signals."
        return "Tourism/event-sensitive terms change the lag profile, consistent with their more volatile attention signals."
    if stable_peak and significant:
        return "Lag structure and statistical evidence are broadly consistent with the baseline."
    if stable_peak:
        return "Peak timing remains close to the baseline, although statistical strength changes."
    return "Lag timing differs from the baseline and should be treated as sensitivity evidence."


def load_pao_sheet(sheet: str, pao_col: str, dimension_cols: list[str]) -> pd.DataFrame:
    cols = [COL_CITY, COL_YEAR, pao_col] + dimension_cols
    return read_pao(sheet, cols)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_model_base()
    table_b_rows = []
    lag_effect_frames = []
    model_info_rows = []

    baseline_dimensions = None

    for version in PAO_VERSIONS:
        pao = load_pao_sheet(version["sheet"], version["pao_col"], version["dimension_cols"])
        merged = base.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
        if merged.shape[0] != base.shape[0]:
            raise ValueError(
                f"{version['construction']} merge has {merged.shape[0]} rows; expected {base.shape[0]}."
            )

        merged["PAO_scaled"] = merged[version["pao_col"]] / PAO_SCALE
        fitted = fit_centered_almon_pdl(merged)

        lag_df = fitted["lag_effects"].copy()
        lag_df.insert(0, "PAO construction", version["construction"])
        lag_effect_frames.append(lag_df)

        table_b_rows.append(
            {
                "PAO construction": version["construction"],
                "Peak lag": f"{fitted['peak_lag']:.2f} years",
                "Raw quadratic vertex": (
                    f"{fitted['raw_peak_lag']:.2f} years"
                    if np.isfinite(fitted["raw_peak_lag"])
                    else "NA"
                ),
                "Cumulative-effect window": f"1-{LAG_MAX} years",
                "Marginal effect direction": format_direction(fitted["peak_effect"]),
                "Cumulative effect direction": format_direction(fitted["cumulative_effect"]),
                "Significance": format_significance(
                    fitted["joint_p"], fitted["peak_p"], fitted["cumulative_p"]
                ),
                "Interpretation": build_interpretation(
                    version["construction"],
                    fitted["peak_lag"],
                    fitted["cumulative_p"],
                    fitted["joint_p"],
                ),
            }
        )

        model_info_rows.append(
            {
                "PAO construction": version["construction"],
                "pao_column": version["pao_col"],
                "nobs_after_lags": fitted["nobs"],
                "cities": fitted["n_cities"],
                "year_min_after_lags": fitted["year_min"],
                "year_max_after_lags": fitted["year_max"],
                "peak_lag_numeric": fitted["peak_lag"],
                "peak_effect": fitted["peak_effect"],
                "peak_se": fitted["peak_se"],
                "peak_p": fitted["peak_p"],
                "cumulative_effect_lag_1_4": fitted["cumulative_effect"],
                "cumulative_se": fitted["cumulative_se"],
                "cumulative_p": fitted["cumulative_p"],
                "joint_p_Z_terms": fitted["joint_p"],
                "r_squared": float(fitted["result"].rsquared),
                "adj_r_squared": float(fitted["result"].rsquared_adj),
            }
        )

        if version["construction"] == "Baseline PAO":
            baseline_dimensions = merged[
                [COL_CITY, COL_YEAR]
                + [
                    "PAO_baseline_Economic",
                    "PAO_baseline_Ecological",
                    "PAO_baseline_Social",
                ]
            ].rename(
                columns={
                    "PAO_baseline_Economic": "Economic PAO",
                    "PAO_baseline_Ecological": "Ecological PAO",
                    "PAO_baseline_Social": "Social PAO",
                }
            )

    table_b = pd.DataFrame(table_b_rows)
    lag_effects = pd.concat(lag_effect_frames, ignore_index=True)
    model_info = pd.DataFrame(model_info_rows)

    if baseline_dimensions is None:
        raise RuntimeError("Baseline PAO dimensions were not loaded.")

    table_c_pearson = baseline_dimensions[
        ["Economic PAO", "Ecological PAO", "Social PAO"]
    ].corr(method="pearson")
    table_c_spearman = baseline_dimensions[
        ["Economic PAO", "Ecological PAO", "Social PAO"]
    ].corr(method="spearman")
    table_c_vif = calculate_vif(
        baseline_dimensions,
        ["Economic PAO", "Ecological PAO", "Social PAO"],
    )

    output_xlsx = OUTPUT_DIR / "PAO_PDL_robustness_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        table_b.to_excel(writer, sheet_name="Table_B_PDL_robustness", index=False)
        lag_effects.to_excel(writer, sheet_name="lag_marginal_effects", index=False)
        model_info.to_excel(writer, sheet_name="model_diagnostics", index=False)
        table_c_pearson.to_excel(writer, sheet_name="Table_C_Pearson")
        table_c_spearman.to_excel(writer, sheet_name="Table_C_Spearman")
        table_c_vif.to_excel(writer, sheet_name="Table_C_VIF", index=False)


    note = f"""# PAO PDL robustness results

Specification: centered quadratic Almon PDL with PAO lags 1-{LAG_MAX}, city fixed effects, year fixed effects, and city-clustered standard errors. PAO is divided by {PAO_SCALE:,.0f}, so coefficients are per 100,000 PAO units.

Inputs: `Coastal urban resilience.xlsx` and `Baidu index.xlsx` (PAO versions built in common.py).

Output: `PAO_PDL_robustness_results.xlsx` (sheet Table_B_PDL_robustness = Table S9).

SHAP interpretation note: Because the three PAO dimensions are conceptually and empirically correlated, and the VIF diagnostics indicate multicollinearity among PAO dimensions, SHAP values should be interpreted as model-attributed predictive contributions rather than independent causal effects.
"""
    (OUTPUT_DIR / "PAO_PDL_robustness_summary.md").write_text(note, encoding="utf-8")

    print(f"Saved PDL robustness outputs to: {OUTPUT_DIR}")
    print("\nTable B:")
    print(table_b.to_string(index=False))
    print("\nTable C Pearson:")
    print(table_c_pearson.to_string())
    print("\nTable C VIF:")
    print(table_c_vif.to_string(index=False))


if __name__ == "__main__":
    main()
