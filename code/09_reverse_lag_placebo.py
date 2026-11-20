#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


from common import PAO_SCALE, load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("09_reverse_lag_placebo")

COL_YEAR = "年份"
COL_CITY = "城市"
COL_Y = "Resilience_Score"
PAO_COL = "PAO_baseline"


def normalize_city(name: object) -> str:
    text = str(name).strip()
    return text if text.endswith("市") else f"{text}市"


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
    df["PAO_scaled"] = df[PAO_COL] / PAO_SCALE
    return df


def fit_reverse_lag(df: pd.DataFrame, max_lag: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values([COL_CITY, COL_YEAR]).copy()
    lag_cols = []
    for lag in range(1, max_lag + 1):
        col = f"Resilience_L{lag}"
        d[col] = d.groupby(COL_CITY)[COL_Y].shift(lag)
        lag_cols.append(col)
    d = d.dropna(subset=["PAO_scaled"] + lag_cols).copy()

    formula = f"PAO_scaled ~ {' + '.join(lag_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})

    coef_rows = []
    for col in lag_cols:
        coef_rows.append(
            {
                "Diagnostic": "Reverse-lag model",
                "Term": col,
                "Coefficient": float(result.params[col]),
                "Std. error": float(result.bse[col]),
                "p-value": float(result.pvalues[col]),
                "stars": stars(float(result.pvalues[col])),
            }
        )
    coef_table = pd.DataFrame(coef_rows)

    joint = result.wald_test(", ".join([f"{col} = 0" for col in lag_cols]), scalar=True)
    summary = pd.DataFrame(
        [
            {
                "Diagnostic": "Reverse-lag model",
                "Model": f"PAO_t on Resilience L1-L{max_lag}",
                "N obs": int(result.nobs),
                "Year min": int(d[COL_YEAR].min()),
                "Year max": int(d[COL_YEAR].max()),
                "Adj. R2": float(result.rsquared_adj),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "Joint null": "All lagged resilience coefficients = 0",
                "Joint statistic": float(joint.statistic),
                "Joint p-value": float(joint.pvalue),
                "Conclusion": (
                    "Lagged resilience predicts PAO; reverse-prediction concern remains"
                    if float(joint.pvalue) < 0.1
                    else "No strong evidence that lagged resilience predicts PAO"
                ),
            }
        ]
    )
    return summary, coef_table


def fit_placebo_leads(df: pd.DataFrame, max_lead: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values([COL_CITY, COL_YEAR]).copy()
    lead_cols = []
    for lead in range(1, max_lead + 1):
        col = f"PAO_F{lead}"
        d[col] = d.groupby(COL_CITY)["PAO_scaled"].shift(-lead)
        lead_cols.append(col)
    d = d.dropna(subset=[COL_Y] + lead_cols).copy()

    formula = f"{COL_Y} ~ {' + '.join(lead_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})

    coef_rows = []
    for col in lead_cols:
        coef_rows.append(
            {
                "Diagnostic": "Placebo lead test",
                "Term": col,
                "Coefficient": float(result.params[col]),
                "Std. error": float(result.bse[col]),
                "p-value": float(result.pvalues[col]),
                "stars": stars(float(result.pvalues[col])),
            }
        )
    coef_table = pd.DataFrame(coef_rows)

    joint = result.wald_test(", ".join([f"{col} = 0" for col in lead_cols]), scalar=True)
    summary = pd.DataFrame(
        [
            {
                "Diagnostic": "Placebo lead test",
                "Model": f"Resilience_t on future PAO F1-F{max_lead}",
                "N obs": int(result.nobs),
                "Year min": int(d[COL_YEAR].min()),
                "Year max": int(d[COL_YEAR].max()),
                "Adj. R2": float(result.rsquared_adj),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "Joint null": "All future PAO coefficients = 0",
                "Joint statistic": float(joint.statistic),
                "Joint p-value": float(joint.pvalue),
                "Conclusion": (
                    "Future PAO predicts current resilience; common-trend/reverse-causality concern remains"
                    if float(joint.pvalue) < 0.1
                    else "No strong evidence that future PAO predicts current resilience"
                ),
            }
        ]
    )
    return summary, coef_table


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    reverse_summary, reverse_coef = fit_reverse_lag(df, max_lag=4)
    placebo_summary, placebo_coef = fit_placebo_leads(df, max_lead=2)

    summary = pd.concat([reverse_summary, placebo_summary], ignore_index=True)
    coefficients = pd.concat([reverse_coef, placebo_coef], ignore_index=True)

    output_xlsx = OUTPUT_DIR / "reverse_causality_diagnostics.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="diagnostic_summary", index=False)
        coefficients.to_excel(writer, sheet_name="diagnostic_coefficients", index=False)


    note = """# Reverse-causality diagnostics

These tests are diagnostics rather than causal identification strategies.

1. Reverse-lag model: current PAO is regressed on lagged resilience scores,
   with city and year fixed effects and city-clustered standard errors.
2. Placebo lead test: current resilience is regressed on future PAO leads,
   with city and year fixed effects and city-clustered standard errors.
"""
    (OUTPUT_DIR / "reverse_causality_diagnostics_note.md").write_text(note, encoding="utf-8")

    print(f"Saved reverse-causality diagnostics to: {OUTPUT_DIR}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nCoefficients:")
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
