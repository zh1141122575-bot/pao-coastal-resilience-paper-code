#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from common import output_dir
from resilience_tools import (
    COL_CITY, COL_ORIGINAL_Y, COL_YEAR, ahp_weights, entropy_weights, fit_centered_almon_pdl,
    indicator_columns, load_and_clean_data, load_pao, normalized_indicator_matrix,
)

OUTPUT_DIR = output_dir("04_resilience_indicator_subsets")
LGGA_COL = "词频总和"
PERIOD_LABEL = "2014-2024"


def minmax(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float).reshape(-1, 1)
    return pd.Series(MinMaxScaler().fit_transform(arr).ravel(), index=values.index)


def weighted_score(x_norm: pd.DataFrame, weights: pd.Series, cols: list[str]) -> pd.Series:
    w = weights.reindex(cols).astype(float)
    w = w / w.sum()
    x_arr = np.asarray(x_norm[cols].to_numpy(dtype=float), dtype=np.float64)
    w_arr = np.asarray(w.to_numpy(dtype=float), dtype=np.float64)
    score = np.sum(x_arr * w_arr.reshape(1, -1), axis=1)
    return pd.Series(score, index=x_norm.index)


def build_scores(df: pd.DataFrame, x_norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    ahp = ahp_weights().reindex(indicator_columns())
    ewm = entropy_weights(x_norm).reindex(indicator_columns())
    ahp_ewm = (0.5 * ahp + 0.5 * ewm).rename("AHP_EWM_weight")

    all_cols = indicator_columns()
    no_lgga_cols = [col for col in all_cols if col != LGGA_COL]

    scores = df[[COL_CITY, COL_YEAR, COL_ORIGINAL_Y]].copy()
    scores["AHP_EWM_all_indicators"] = minmax(weighted_score(x_norm, ahp_ewm, all_cols))
    scores["AHP_EWM_without_LGGA"] = minmax(weighted_score(x_norm, ahp_ewm, no_lgga_cols))
    scores["AHP_EWM_LGGA_only_governance_removed"] = scores["AHP_EWM_without_LGGA"]
    return scores, ahp_ewm


def subset_uncertainty_full_period(
    df: pd.DataFrame,
    x_norm: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    cols = indicator_columns()
    rows = []

    # Exhaustive subset enumeration is feasible for 14 indicators: 2^14 - 1.
    for k in range(1, len(cols) + 1):
        for subset in combinations(cols, k):
            subset_cols = list(subset)
            score = weighted_score(x_norm, weights, subset_cols)
            national_mean = float(score.mean() * 100.0)
            rows.append(
                {
                    "period": PERIOD_LABEL,
                    "n_selected_indicators": k,
                    "subset": "|".join(subset_cols),
                    "includes_LGGA": LGGA_COL in subset_cols,
                    "national_mean_score_0_100": national_mean,
                }
            )
    return pd.DataFrame(rows)


def subset_summary(subset_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = subset_scores.groupby(["n_selected_indicators", "includes_LGGA"])[
        "national_mean_score_0_100"
    ]
    return grouped.agg(
        n_subsets="count",
        mean="mean",
        median="median",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        min="min",
        max="max",
        sd="std",
    ).reset_index()


def score_consistency(scores: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("Original BPNN-PSO", COL_ORIGINAL_Y, "AHP-EWM all indicators", "AHP_EWM_all_indicators"),
        ("Original BPNN-PSO", COL_ORIGINAL_Y, "AHP-EWM without LGGA", "AHP_EWM_without_LGGA"),
        (
            "AHP-EWM all indicators",
            "AHP_EWM_all_indicators",
            "AHP-EWM without LGGA",
            "AHP_EWM_without_LGGA",
        ),
    ]
    rows = []
    for a_name, a_col, b_name, b_col in pairs:
        rows.append(
            {
                "Score A": a_name,
                "Score B": b_name,
                "Pearson r": scores[a_col].corr(scores[b_col], method="pearson"),
                "Spearman rho": scores[a_col].corr(scores[b_col], method="spearman"),
                "Mean absolute difference": float((scores[a_col] - scores[b_col]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def yearly_trajectory(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby(COL_YEAR)[
            [COL_ORIGINAL_Y, "AHP_EWM_all_indicators", "AHP_EWM_without_LGGA"]
        ]
        .mean()
        .reset_index()
    )


def pdl_no_lgga(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pao = load_pao()
    model_df = scores.merge(
        pao[[COL_CITY, COL_YEAR, "PAO_scaled"]],
        on=[COL_CITY, COL_YEAR],
        how="inner",
    )
    if model_df.shape[0] != scores.shape[0]:
        raise ValueError(
            f"PAO merge returned {model_df.shape[0]} rows; expected {scores.shape[0]}."
        )
    summaries = []
    lag_frames = []
    for y_col in [COL_ORIGINAL_Y, "AHP_EWM_all_indicators", "AHP_EWM_without_LGGA"]:
        fitted = fit_centered_almon_pdl(model_df, y_col)
        summary = fitted["summary"].copy()
        summary["Dependent variable"] = y_col
        summaries.append(summary)
        lag_frames.append(fitted["lag_effects"])
    return pd.DataFrame(summaries), pd.concat(lag_frames, ignore_index=True)


def write_note(consistency: pd.DataFrame, pdl_summary: pd.DataFrame) -> None:
    corr_no_lgga = consistency.loc[
        consistency["Score B"].eq("AHP-EWM without LGGA"), "Spearman rho"
    ].iloc[0]
    pdl_no_lgga_row = pdl_summary.loc[
        pdl_summary["Dependent variable"].eq("AHP_EWM_without_LGGA")
    ].iloc[0]
    note = f"""# Supplementary Note 2: Resilience indicator-system sensitivity

This diagnostic analysis examines whether the resilience indicator system is
sensitive to the number of selected indicators and, in particular, to the local
government governance attention indicator (LGGA; `词频总和`) derived from
government reports.

For the indicator-number uncertainty analysis, all non-empty subsets of the
14 normalized indicators were enumerated. For each subset, AHP-EWM weights were
renormalized within the selected indicators, and the national mean score over
the full {PERIOD_LABEL} study period was calculated on a 0-100 scale. The box plots therefore describe
indicator-selection uncertainty, not sampling uncertainty.

For the LGGA robustness diagnostic, the AHP-EWM score was reconstructed after
excluding `词频总和` and renormalizing the remaining indicator weights. The
Spearman correlation between the original BPNN-PSO score and the no-LGGA
AHP-EWM score is {corr_no_lgga:.3f}. The PDL diagnostic using the no-LGGA score
has peak lag {pdl_no_lgga_row['Peak lag']:.2f}, cumulative effect
{pdl_no_lgga_row['Cumulative effect L1-L4']:.3f}, cumulative p
{pdl_no_lgga_row['Cumulative p']:.3g}, and joint p
{pdl_no_lgga_row['Joint p']:.3g}.

These results should be interpreted as sensitivity evidence for the indicator
system. They do not imply that government-report attention is unimportant;
rather, they test whether the overall resilience pattern and PAO-resilience
association are mechanically driven by this single text-based governance
indicator.
"""
    (OUTPUT_DIR / "Supplementary_Note_2_resilience_indicator_LGGA_sensitivity.md").write_text(
        note, encoding="utf-8"
    )


def main() -> None:
    df = load_and_clean_data()
    x_norm = normalized_indicator_matrix(df)
    scores, ahp_ewm = build_scores(df, x_norm)
    subset_scores = subset_uncertainty_full_period(df, x_norm, ahp_ewm)
    summary = subset_summary(subset_scores)
    consistency = score_consistency(scores)
    trajectory = yearly_trajectory(scores)
    pdl_summary, pdl_lags = pdl_no_lgga(scores)


    with pd.ExcelWriter(OUTPUT_DIR / "resilience_indicator_LGGA_sensitivity_results.xlsx", engine="openpyxl") as writer:
        subset_scores.to_excel(writer, sheet_name="subset_scores_2014_2024", index=False)
        summary.to_excel(writer, sheet_name="subset_summary", index=False)
        scores.to_excel(writer, sheet_name="scores_with_without_LGGA", index=False)
        consistency.to_excel(writer, sheet_name="score_consistency", index=False)
        trajectory.to_excel(writer, sheet_name="national_mean", index=False)
        pdl_summary.to_excel(writer, sheet_name="PDL_summary", index=False)
        pdl_lags.to_excel(writer, sheet_name="PDL_lag_effects", index=False)

    write_note(consistency, pdl_summary)

    print(f"Saved resilience indicator/LGGA sensitivity outputs to: {OUTPUT_DIR}")
    print("\\nLGGA score consistency:")
    print(consistency.to_string(index=False))
    print("\\nPDL robustness:")
    print(pdl_summary.to_string(index=False))


if __name__ == "__main__":
    main()
