#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import output_dir
from resilience_tools import (
    BPNN_EPOCHS, BPNN_HIDDEN_SIZE, BPNN_LEARNING_RATE, COL_CITY, COL_YEAR, INDICATOR_SYSTEM,
    LAGS, POLY_DEGREE, PSO_BOUNDS, PSO_C1, PSO_C2, PSO_INERTIA, PSO_ITERATIONS, PSO_PARTICLES,
    RANDOM_STATE, ahp_weights, build_scores, entropy_weights, indicator_columns,
    load_and_clean_data, load_pao, normalized_indicator_matrix, p_value_from_z, stars,
)

OUTPUT_DIR = output_dir("02_resilience_index")


def table_s2_weights() -> pd.DataFrame:
    df = load_and_clean_data()
    x_norm = normalized_indicator_matrix(df)
    ahp = ahp_weights()
    ewm = entropy_weights(x_norm)
    out = pd.DataFrame({
        "Dimension": [dim for dim, entries in INDICATOR_SYSTEM.items() for _ in entries],
        "Symbol": [item["symbol"] for entries in INDICATOR_SYSTEM.values() for item in entries],
        "Indicator": [item["label"] for entries in INDICATOR_SYSTEM.values() for item in entries],
        "AHP weight": ahp.reindex(indicator_columns()).to_numpy(),
        "EWM weight": ewm.reindex(indicator_columns()).to_numpy(),
    })
    out["AHP-EWM combined weight (Table S2)"] = 0.5 * out["AHP weight"] + 0.5 * out["EWM weight"]
    return out


def optimized_score_columns() -> list[str]:
    return [
        "Legacy_original_BPNN_PSO",
        "B0_equal_dimension_BPNN_PSO",
        "B1_AHP_only_BPNN_PSO",
        "B2_EWM_only_BPNN_PSO",
        "B3_AHP_EWM_BPNN_PSO",
        "B4_MC_mean_BPNN_PSO",
    ]


def score_correlations(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = optimized_score_columns()
    pearson = scores[cols].corr(method="pearson")
    spearman = scores[cols].corr(method="spearman")
    rows = []
    for col in cols:
        if col == "Legacy_original_BPNN_PSO":
            continue
        rows.append(
            {
                "Alternative BPNN-PSO resilience score": col,
                "Pearson r with legacy original": scores["Legacy_original_BPNN_PSO"].corr(
                    scores[col], method="pearson"
                ),
                "Spearman r with legacy original": scores["Legacy_original_BPNN_PSO"].corr(
                    scores[col], method="spearman"
                ),
            }
        )
    return pearson, spearman, pd.DataFrame(rows)


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
    summary = {
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
        "Interpretation": build_pdl_interpretation(
            y_col, peak_lag, cumulative, cumulative_p, float(joint.pvalue)
        ),
    }
    return summary, lag_df


def pdl_robustness(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pao = load_pao()
    df = scores.merge(pao[[COL_CITY, COL_YEAR, "PAO_scaled"]], on=[COL_CITY, COL_YEAR], how="inner")
    summaries = []
    lag_frames = []
    for y_col in optimized_score_columns():
        summary, lags = fit_centered_almon_pdl(df, y_col)
        summaries.append(summary)
        lag_frames.append(lags)
    return pd.DataFrame(summaries), pd.concat(lag_frames, ignore_index=True)


def build_pdl_interpretation(
    y_col: str, peak_lag: float, cumulative: float, cumulative_p: float, joint_p: float
) -> str:
    direction = "positive" if cumulative > 0 else "negative" if cumulative < 0 else "near-zero"
    stable_peak = 1.5 <= peak_lag <= 3.3
    significant = (cumulative_p < 0.1) or (joint_p < 0.1)
    if y_col == "Legacy_original_BPNN_PSO":
        return "Previously saved BPNN-PSO score used as legacy reference."
    if stable_peak and significant:
        return f"Robust {direction} association under the same BPNN-PSO aggregation pipeline."
    if stable_peak:
        return f"Peak timing remains similar, but statistical strength changes; cumulative direction is {direction}."
    return f"Lag profile changes under this BPNN-PSO weighting scheme; cumulative direction is {direction}."


def write_method_note(diagnostics: pd.DataFrame) -> None:
    text = f"""# BPNN-PSO Resilience Sensitivity Analysis

All optimized alternative resilience scores are generated using the same BPNN-PSO pipeline.

Model parameters:
- Input indicators: {len(indicator_columns())}
- Hidden layer: one hidden layer with {BPNN_HIDDEN_SIZE} neurons
- Output layer: one neuron
- Activation function: sigmoid in hidden and output layers
- Loss function: mean squared error between BPNN output and normalized weighted-sum target
- Learning rate: {BPNN_LEARNING_RATE}
- Training epochs after PSO initialization: {BPNN_EPOCHS}
- PSO particles: {PSO_PARTICLES}
- PSO iterations: {PSO_ITERATIONS}
- PSO bounds: {PSO_BOUNDS}
- PSO inertia: {PSO_INERTIA}
- PSO cognitive/social coefficients: c1={PSO_C1}, c2={PSO_C2}
- Random seed base: {RANDOM_STATE}; each weighting scheme uses a fixed deterministic seed offset.

Training flow:
1. Original indicators are min-max normalized.
2. A weighting scheme is applied to create weighted inputs.
3. The weighted-sum target is computed and min-max normalized.
4. PSO searches the BPNN parameter vector to minimize MSE.
5. The BPNN is initialized with the best PSO particle and trained by gradient descent.
6. The final BPNN output is min-max normalized and used as the optimized resilience score.

The previously saved BPNN-PSO score is reported only as a legacy reference because its original run did not fix a random seed in the notebook.

Final training diagnostics:
"""
    for _, row in diagnostics.iterrows():
        text += (
            f"- {row['version']}: PSO final MSE={row['pso_best_mse_final']:.6g}; "
            f"BPNN final MSE={row['bpnn_mse_final']:.6g}; seed={int(row['random_seed'])}.\n"
        )
    (OUTPUT_DIR / "bpnn_pso_training_method_note.md").write_text(text, encoding="utf-8")


def main() -> None:
    scores, weights, diagnostics, histories, mc_interval = build_scores()
    pearson, spearman, corr = score_correlations(scores)
    table_s2 = table_s2_weights()
    pdl_summary, pdl_lags = pdl_robustness(scores)
    write_method_note(diagnostics)

    output_xlsx = OUTPUT_DIR / "resilience_bpnn_pso_sensitivity_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        table_s2.to_excel(writer, sheet_name="Table_S2_weights", index=False)
        weights.to_excel(writer, sheet_name="weight_schemes", index=False)
        scores.to_excel(writer, sheet_name="scores", index=False)
        mc_interval.to_excel(writer, sheet_name="MC_additive_CI", index=False)
        diagnostics.to_excel(writer, sheet_name="BPNN_PSO_parameters", index=False)
        histories.to_excel(writer, sheet_name="training_history", index=False)
        pearson.to_excel(writer, sheet_name="score_corr_pearson")
        spearman.to_excel(writer, sheet_name="score_corr_spearman")
        corr.to_excel(writer, sheet_name="corr_with_legacy", index=False)
        pdl_summary.to_excel(writer, sheet_name="PDL_robustness", index=False)
        pdl_lags.to_excel(writer, sheet_name="PDL_lag_effects", index=False)


    print(f"Saved BPNN-PSO sensitivity outputs to: {OUTPUT_DIR}")
    print("\nBPNN-PSO training diagnostics:")
    print(diagnostics.to_string(index=False))
    print("\nCorrelations with legacy original:")
    print(corr.to_string(index=False))
    print("\nPDL robustness:")
    print(pdl_summary.to_string(index=False))


if __name__ == "__main__":
    main()
