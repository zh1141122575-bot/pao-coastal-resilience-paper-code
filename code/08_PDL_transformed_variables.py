#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


from common import PAO_SCALE, load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("08_PDL_transformed_variables")

COL_YEAR = "年份"
COL_CITY = "城市"
COL_Y = "Resilience_Score"
PAO_COL = "PAO_baseline"
LAGS = [1, 2, 3, 4]
POLY_DEGREE = 2


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

    df = df.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)
    df["PAO_level"] = df[PAO_COL] / PAO_SCALE
    df["year_centered"] = df[COL_YEAR] - df[COL_YEAR].mean()

    detrend_model = smf.ols(f"PAO_level ~ year_centered + C({COL_CITY})", data=df).fit()
    df["PAO_detrended"] = detrend_model.resid

    df["PAO_first_difference"] = df.groupby(COL_CITY)["PAO_level"].diff()
    df["PAO_log_first_difference"] = df.groupby(COL_CITY)[PAO_COL].transform(
        lambda s: np.log1p(s).diff()
    )
    return df


def fit_almon_pdl(df_in: pd.DataFrame, pao_var: str) -> dict:
    d = df_in.sort_values([COL_CITY, COL_YEAR]).copy()
    lag_cols = []
    for lag in LAGS:
        col = f"{pao_var}_L{lag}"
        d[col] = d.groupby(COL_CITY)[pao_var].shift(lag)
        lag_cols.append(col)

    d = d.dropna(subset=[COL_Y] + lag_cols).copy()
    lag_values = np.array(LAGS, dtype=float)
    centered = lag_values - lag_values.mean()

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
    kbar = lag_values.mean()

    def lag_effect(lag: float) -> float:
        kh = lag - kbar
        return float(coef["Z0"] + coef["Z1"] * kh + coef["Z2"] * kh * kh)

    def lag_se(lag: float) -> float:
        kh = lag - kbar
        g = np.array([1.0, kh, kh * kh])
        var = float(g @ vcov.values @ g.T)
        return float(np.sqrt(max(var, 0.0)))

    lag_effects = pd.DataFrame(
        {
            "lag": LAGS,
            "marginal_effect": [lag_effect(lag) for lag in LAGS],
            "std_error": [lag_se(lag) for lag in LAGS],
        }
    )
    lag_effects["p_value"] = [
        p_value_from_z(v, se)
        for v, se in zip(lag_effects["marginal_effect"], lag_effects["std_error"])
    ]
    lag_effects["stars"] = lag_effects["p_value"].map(stars)

    grid = np.linspace(min(LAGS), max(LAGS), 301)
    grid_effects = np.array([lag_effect(lag) for lag in grid])
    cumulative_effect = float(lag_effects["marginal_effect"].sum())
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
        "lag_effects": lag_effects,
        "nobs": int(result.nobs),
        "year_min": int(d[COL_YEAR].min()),
        "year_max": int(d[COL_YEAR].max()),
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
    }


def direction(value: float) -> str:
    if value > 0:
        return "Positive"
    if value < 0:
        return "Negative"
    return "Near zero"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    transforms = [
        ("Level PAO", "PAO_level", "Baseline level specification"),
        ("Detrended PAO", "PAO_detrended", "Residual from PAO on city fixed effects and linear year trend"),
        ("First-differenced PAO", "PAO_first_difference", "Annual change in PAO"),
        ("Log first-differenced PAO", "PAO_log_first_difference", "Annual change in log(1+PAO)"),
    ]

    rows = []
    lag_frames = []
    for label, var, purpose in transforms:
        fitted = fit_almon_pdl(df, var)
        rows.append(
            {
                "Specification": label,
                "PAO variable": var,
                "Purpose": purpose,
                "N obs": fitted["nobs"],
                "Year min after lags": fitted["year_min"],
                "Year max after lags": fitted["year_max"],
                "AIC": fitted["aic"],
                "BIC": fitted["bic"],
                "Adj. R2": fitted["adj_r2"],
                "Peak lag": fitted["peak_lag"],
                "Peak effect": fitted["peak_effect"],
                "Peak p": fitted["peak_p"],
                "Cumulative effect": fitted["cumulative_effect"],
                "Cumulative p": fitted["cumulative_p"],
                "Joint p": fitted["joint_p"],
                "Cumulative direction": direction(fitted["cumulative_effect"]),
                "Interpretation": (
                    "Consistent positive lagged association"
                    if fitted["cumulative_effect"] > 0 and fitted["cumulative_p"] < 0.1
                    else "Positive but statistically weaker"
                    if fitted["cumulative_effect"] > 0
                    else "Direction differs from baseline"
                ),
            }
        )
        lag_df = fitted["lag_effects"].copy()
        lag_df.insert(0, "Specification", label)
        lag_frames.append(lag_df)

    summary = pd.DataFrame(rows)
    lag_effects = pd.concat(lag_frames, ignore_index=True)

    output_xlsx = OUTPUT_DIR / "PDL_stationarity_robustness_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="stationarity_robustness", index=False)
        lag_effects.to_excel(writer, sheet_name="lag_effects", index=False)


    note = """# PDL stationarity robustness

This robustness check responds to mixed panel unit-root evidence for PAO.
The main L1-L4 centered Almon PDL is re-estimated using level, detrended,
first-differenced, and log first-differenced PAO.
"""
    (OUTPUT_DIR / "PDL_stationarity_robustness_note.md").write_text(note, encoding="utf-8")

    print(f"Saved stationarity robustness outputs to: {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
