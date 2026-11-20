#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import COL_Y, load_model_base, output_dir
from resilience_tools import (
    COL_CITY, COL_YEAR, INDICATOR_SYSTEM, LAGS, POLY_DEGREE, RANDOM_STATE,
    ahp_weights, entropy_weights, indicator_columns, load_and_clean_data, load_pao,
    normalized_indicator_matrix, p_value_from_z, run_bpnn_pso_version, stars,
)

OUTPUT_DIR = output_dir("03_resilience_MC_alpha")
MC_DRAWS = int(os.environ.get("RESILIENCE_MC_DRAWS", "100"))


def load_baseline_scores() -> pd.DataFrame:
    return load_model_base().rename(columns={COL_Y: "Baseline_BPNN_PSO"})[
        [COL_CITY, COL_YEAR, "Baseline_BPNN_PSO"]
    ]


def make_mc_alpha_weights(x_norm: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    ahp = ahp_weights().reindex(indicator_columns())
    ewm = entropy_weights(x_norm).reindex(indicator_columns())
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for draw in range(1, MC_DRAWS + 1):
        alpha = float(rng.uniform(0.0, 1.0))
        w = alpha * ahp + (1.0 - alpha) * ewm
        w = w / w.sum()
        rows.append({"draw": draw, "alpha_AHP": alpha, **w.to_dict()})
    return pd.DataFrame(rows), ahp, ewm


def build_mc_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_and_clean_data()
    x_norm = normalized_indicator_matrix(df)
    baseline = load_baseline_scores()
    scores = df[[COL_CITY, COL_YEAR]].merge(baseline, on=[COL_CITY, COL_YEAR], how="left")
    if scores["Baseline_BPNN_PSO"].isna().any():
        raise ValueError("Baseline score merge failed for some city-year rows.")

    mc_weights, ahp, ewm = make_mc_alpha_weights(x_norm)
    mc_score_series = []
    histories = []
    diagnostics = []
    for _, row in mc_weights.iterrows():
        draw = int(row["draw"])
        w = row.reindex(indicator_columns()).astype(float)
        score, diag, history = run_bpnn_pso_version(
            x_norm,
            w,
            f"MC_alpha_BPNN_PSO_{draw:03d}",
            seed_offset=1000 + draw,
        )
        mc_score_series.append(score.rename(f"MC_draw_{draw:03d}"))
        histories.append(history)
        diagnostics.append(diag)

    mc_score_df = pd.concat(mc_score_series, axis=1)
    scores = pd.concat([scores, mc_score_df], axis=1)
    score_arr = mc_score_df.to_numpy(dtype=float)
    scores["MC_alpha_mean_BPNN_PSO"] = score_arr.mean(axis=1)
    scores["MC_alpha_p025_BPNN_PSO"] = np.quantile(score_arr, 0.025, axis=1)
    scores["MC_alpha_p975_BPNN_PSO"] = np.quantile(score_arr, 0.975, axis=1)
    scores["MC_alpha_sd_BPNN_PSO"] = score_arr.std(axis=1, ddof=1)

    weight_summary = pd.DataFrame(
        {
            "indicator": indicator_columns(),
            "dimension": [
                dim for dim, entries in INDICATOR_SYSTEM.items() for _ in entries
            ],
            "AHP_weight": ahp.to_numpy(dtype=float),
            "EWM_weight": ewm.to_numpy(dtype=float),
        }
    )
    return scores, mc_weights, weight_summary, pd.concat(histories, ignore_index=True), pd.DataFrame(diagnostics)


def ranking_consistency(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mc_cols = [c for c in scores.columns if c.startswith("MC_draw_")]
    rows = []
    for col in ["MC_alpha_mean_BPNN_PSO"] + mc_cols:
        rows.append(
            {
                "score": col,
                "Pearson_with_baseline": scores["Baseline_BPNN_PSO"].corr(scores[col], method="pearson"),
                "Spearman_with_baseline": scores["Baseline_BPNN_PSO"].corr(scores[col], method="spearman"),
            }
        )
    all_corr = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "metric": "Pearson with baseline across MC draws",
                "mean": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Pearson_with_baseline"].mean(),
                "p025": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Pearson_with_baseline"].quantile(0.025),
                "p975": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Pearson_with_baseline"].quantile(0.975),
                "min": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Pearson_with_baseline"].min(),
                "max": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Pearson_with_baseline"].max(),
            },
            {
                "metric": "Spearman with baseline across MC draws",
                "mean": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Spearman_with_baseline"].mean(),
                "p025": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Spearman_with_baseline"].quantile(0.025),
                "p975": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Spearman_with_baseline"].quantile(0.975),
                "min": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Spearman_with_baseline"].min(),
                "max": all_corr.loc[all_corr["score"].str.startswith("MC_draw_"), "Spearman_with_baseline"].max(),
            },
        ]
    )
    return all_corr, summary


def fit_centered_almon_pdl(df_in: pd.DataFrame, y_col: str) -> tuple[dict, pd.DataFrame]:
    d = df_in.sort_values([COL_CITY, COL_YEAR]).copy()
    lag_cols = []
    for lag in LAGS:
        col = f"PAO_L{lag}"
        d[col] = d.groupby(COL_CITY)["PAO_scaled"].shift(lag)
        lag_cols.append(col)
    d = d.dropna(subset=lag_cols + [y_col]).copy()

    idx = np.array(LAGS, dtype=float)
    kbar = idx.mean()
    centered = idx - kbar
    z_cols = []
    for degree in range(POLY_DEGREE + 1):
        z_col = f"Z{degree}"
        d[z_col] = 0.0
        weights = centered**degree
        for pos, lag in enumerate(LAGS):
            d[z_col] += weights[pos] * d[f"PAO_L{lag}"]
        z_cols.append(z_col)

    formula = f"{y_col} ~ {' + '.join(z_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})
    coef = result.params.reindex(z_cols).fillna(0.0)
    vcov = result.cov_params().loc[z_cols, z_cols]

    def lag_effect(lag_value: float) -> float:
        kh = lag_value - kbar
        return float(coef["Z0"] + coef["Z1"] * kh + coef["Z2"] * kh * kh)

    def lag_se(lag_value: float) -> float:
        kh = lag_value - kbar
        g = np.array([1.0, kh, kh * kh])
        return float(np.sqrt(max(float(g @ vcov.values @ g.T), 0.0)))

    lag_rows = []
    for lag in LAGS:
        effect = lag_effect(float(lag))
        se = lag_se(float(lag))
        p_val = p_value_from_z(effect, se)
        lag_rows.append(
            {
                "dependent_variable": y_col,
                "lag": lag,
                "marginal_effect": effect,
                "std_error": se,
                "p_value": p_val,
                "stars": stars(p_val),
            }
        )
    lag_df = pd.DataFrame(lag_rows)

    lag_grid = np.linspace(min(LAGS), max(LAGS), 301)
    grid_effects = np.array([lag_effect(h) for h in lag_grid])
    cumulative = float(lag_df["marginal_effect"].sum())
    direction_sign = np.sign(cumulative) if cumulative != 0 else 1.0
    grid_idx = int(np.argmax(grid_effects) if direction_sign >= 0 else np.argmin(grid_effects))
    peak_lag = float(lag_grid[grid_idx])
    peak_effect = float(grid_effects[grid_idx])
    peak_se = lag_se(peak_lag)
    peak_p = p_value_from_z(peak_effect, peak_se)
    cumulative_weights = np.array(
        [float(np.sum(centered**degree)) for degree in range(POLY_DEGREE + 1)]
    )
    cumulative_se = float(
        np.sqrt(max(float(cumulative_weights @ vcov.values @ cumulative_weights.T), 0.0))
    )
    cumulative_p = p_value_from_z(cumulative, cumulative_se)
    joint = result.wald_test("Z0 = 0, Z1 = 0, Z2 = 0", scalar=True)

    return (
        {
            "Dependent variable": y_col,
            "Peak lag": peak_lag,
            "Peak effect": peak_effect,
            "Peak p": peak_p,
            "Cumulative effect L1-L4": cumulative,
            "Cumulative se": cumulative_se,
            "Cumulative p": cumulative_p,
            "Joint p": float(joint.pvalue),
            "N obs": int(result.nobs),
            "N cities": int(d[COL_CITY].nunique()),
            "Year range after lags": f"{int(d[COL_YEAR].min())}-{int(d[COL_YEAR].max())}",
            "Interpretation": (
                "Baseline existing BPNN-PSO resilience score."
                if y_col == "Baseline_BPNN_PSO"
                else "Monte Carlo mean resilience under random AHP/EWM mixing."
            ),
        },
        lag_df,
    )


def pdl_robustness(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = scores.merge(load_pao(), on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != scores.shape[0]:
        raise ValueError("PAO merge failed for PDL robustness.")
    summaries = []
    lag_frames = []
    for y_col in ["Baseline_BPNN_PSO", "MC_alpha_mean_BPNN_PSO"]:
        summary, lags = fit_centered_almon_pdl(df, y_col)
        summaries.append(summary)
        lag_frames.append(lags)
    return pd.DataFrame(summaries), pd.concat(lag_frames, ignore_index=True)


def write_appendix_note(
    mc_weights: pd.DataFrame, corr_summary: pd.DataFrame, pdl_summary: pd.DataFrame
) -> None:
    spearman = corr_summary.loc[
        corr_summary["metric"].eq("Spearman with baseline across MC draws")
    ].iloc[0]
    text = f"""Appendix note: Monte Carlo sensitivity of resilience weights

The baseline trajectory reports the existing BPNN-PSO resilience score used in the main analysis. To examine whether this score is sensitive to the fixed AHP-EWM mixing assumption, we randomly varied the relative contribution of AHP and EWM weights. In each Monte Carlo draw, alpha was sampled from U(0,1), and indicator weights were reconstructed as:

w_j(alpha) = alpha * w_j^AHP + (1 - alpha) * w_j^EWM.

The resulting weight vector was normalized and passed through the same BPNN-PSO pipeline to generate an alternative resilience score. The shaded band in Figure Sx reports the 95% interval of the Monte Carlo-generated national mean resilience trajectory. This interval is a weight-allocation uncertainty interval, not a sampling confidence interval.

Ranking consistency:
Across {MC_DRAWS} Monte Carlo draws, the Spearman correlation between the baseline resilience score and Monte Carlo-generated resilience scores has mean {spearman['mean']:.3f}, with a 95% interval of [{spearman['p025']:.3f}, {spearman['p975']:.3f}].

PDL robustness:
"""
    for _, row in pdl_summary.iterrows():
        text += (
            f"- {row['Dependent variable']}: peak lag = {row['Peak lag']:.2f}, "
            f"cumulative effect = {row['Cumulative effect L1-L4']:.4f}, "
            f"cumulative p = {row['Cumulative p']:.3g}, joint p = {row['Joint p']:.3g}.\n"
        )
    (OUTPUT_DIR / "appendix_note_baseline_resilience_MC_alpha.md").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    scores, mc_weights, weight_summary, histories, diagnostics = build_mc_scores()
    corr_all, corr_summary = ranking_consistency(scores)
    pdl_summary, pdl_lags = pdl_robustness(scores)
    write_appendix_note(mc_weights, corr_summary, pdl_summary)

    yearly = scores.groupby(COL_YEAR)[
        [
            "Baseline_BPNN_PSO",
            "MC_alpha_mean_BPNN_PSO",
            "MC_alpha_p025_BPNN_PSO",
            "MC_alpha_p975_BPNN_PSO",
        ]
    ].mean().reset_index()
    output_xlsx = OUTPUT_DIR / "baseline_resilience_MC_alpha_BPNN_PSO_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="scores_and_MC_interval", index=False)
        mc_weights.to_excel(writer, sheet_name="MC_alpha_weights", index=False)
        weight_summary.to_excel(writer, sheet_name="base_weights", index=False)
        diagnostics.to_excel(writer, sheet_name="BPNN_PSO_diagnostics", index=False)
        histories.to_excel(writer, sheet_name="training_history", index=False)
        corr_all.to_excel(writer, sheet_name="ranking_corr_all_draws", index=False)
        corr_summary.to_excel(writer, sheet_name="ranking_corr_summary", index=False)
        pdl_summary.to_excel(writer, sheet_name="PDL_robustness", index=False)
        pdl_lags.to_excel(writer, sheet_name="PDL_lag_effects", index=False)
        yearly.to_excel(writer, sheet_name="FigS3_yearly", index=False)


    print(f"Saved baseline resilience MC-alpha outputs to: {OUTPUT_DIR}")
    print("\nYearly figure data:")
    print(yearly.to_string(index=False))
    print("\nRanking consistency summary:")
    print(corr_summary.to_string(index=False))
    print("\nPDL robustness:")
    print(pdl_summary.to_string(index=False))


if __name__ == "__main__":
    main()
