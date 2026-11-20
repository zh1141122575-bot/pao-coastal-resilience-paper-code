#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from common import load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("13_mismatch")

COL_CITY = "城市"
COL_YEAR = "年份"
COL_Y = "Resilience_Score"
PAO_COL = "PAO_baseline"
LAG_START = 1
LAG_END = 4


TYPE_ORDER = [
    "High PAO-High resilience",
    "High PAO-Low resilience",
    "Low PAO-High resilience",
    "Low PAO-Low resilience",
]

TYPE_CN = {
    "High PAO-High resilience": "高PAO-高韧性",
    "High PAO-Low resilience": "高PAO-低韧性",
    "Low PAO-High resilience": "低PAO-高韧性",
    "Low PAO-Low resilience": "低PAO-低韧性",
}

TYPE_COLOR = {
    "High PAO-High resilience": "#2b8cbe",
    "High PAO-Low resilience": "#d95f0e",
    "Low PAO-High resilience": "#31a354",
    "Low PAO-Low resilience": "#756bb1",
}


def normalize_city(name: object) -> str:
    text = str(name).strip()
    return text if text.endswith("市") else f"{text}市"


def classify(high_pao: bool, high_resilience: bool) -> str:
    if high_pao and high_resilience:
        return "High PAO-High resilience"
    if high_pao and not high_resilience:
        return "High PAO-Low resilience"
    if not high_pao and high_resilience:
        return "Low PAO-High resilience"
    return "Low PAO-Low resilience"


def load_data() -> pd.DataFrame:
    resilience = load_model_base()
    pao = read_pao("baseline_main_balanced", [COL_CITY, COL_YEAR, PAO_COL])

    merged = resilience.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    if merged.shape[0] != resilience.shape[0]:
        raise ValueError(
            f"Merged rows {merged.shape[0]} do not match resilience rows {resilience.shape[0]}."
        )
    return merged.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def add_lag_adjusted_pao(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag_cols = []
    for lag in range(LAG_START, LAG_END + 1):
        col = f"PAO_L{lag}"
        out[col] = out.groupby(COL_CITY)[PAO_COL].shift(lag)
        lag_cols.append(col)
    out["Lag_adjusted_PAO_L1_L4_mean"] = out[lag_cols].mean(axis=1)
    out["Lag_adjusted_PAO_L1_L4_sum"] = out[lag_cols].sum(axis=1)
    return out.dropna(subset=lag_cols + [COL_Y]).copy()


def add_classification(
    df: pd.DataFrame,
    pao_threshold: float,
    resilience_threshold: float,
) -> pd.DataFrame:
    out = df.copy()
    out["PAO_group"] = np.where(
        out["Lag_adjusted_PAO_L1_L4_mean"] >= pao_threshold, "High PAO", "Low PAO"
    )
    out["Resilience_group"] = np.where(out[COL_Y] >= resilience_threshold, "High", "Low")
    out["Mismatch_type"] = [
        classify(pao >= pao_threshold, resilience >= resilience_threshold)
        for pao, resilience in zip(out["Lag_adjusted_PAO_L1_L4_mean"], out[COL_Y])
    ]
    out["Mismatch_type_CN"] = out["Mismatch_type"].map(TYPE_CN)
    return out


def summarize_types(df: pd.DataFrame, count_col: str = "observations") -> pd.DataFrame:
    counts = df["Mismatch_type"].value_counts().reindex(TYPE_ORDER, fill_value=0)
    summary = counts.rename(count_col).reset_index().rename(columns={"index": "Mismatch_type"})
    summary["Mismatch_type_CN"] = summary["Mismatch_type"].map(TYPE_CN)
    summary["share"] = summary[count_col] / summary[count_col].sum()
    return summary


def build_city_summaries(
    pooled: pd.DataFrame,
    pao_threshold: float,
    resilience_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    city_avg = (
        pooled.groupby(COL_CITY)
        .agg(
            Lag_adjusted_PAO_L1_L4_mean=("Lag_adjusted_PAO_L1_L4_mean", "mean"),
            Lag_adjusted_PAO_L1_L4_sum=("Lag_adjusted_PAO_L1_L4_sum", "mean"),
            Resilience_Score=(COL_Y, "mean"),
            years=(COL_YEAR, "nunique"),
            first_year=(COL_YEAR, "min"),
            last_year=(COL_YEAR, "max"),
        )
        .reset_index()
    )

    city_avg = add_classification(city_avg, pao_threshold, resilience_threshold)

    city_type_counts = (
        pooled.groupby([COL_CITY, "Mismatch_type"])
        .size()
        .rename("years_in_type")
        .reset_index()
    )
    dominant_idx = city_type_counts.groupby(COL_CITY)["years_in_type"].idxmax()
    dominant = city_type_counts.loc[dominant_idx].copy()
    dominant = dominant.rename(columns={"Mismatch_type": "Dominant_mismatch_type"})
    total_years = pooled.groupby(COL_CITY)[COL_YEAR].nunique().rename("total_years")
    dominant = dominant.merge(total_years, on=COL_CITY, how="left")
    dominant["dominant_share"] = dominant["years_in_type"] / dominant["total_years"]
    dominant["Dominant_mismatch_type_CN"] = dominant["Dominant_mismatch_type"].map(TYPE_CN)
    dominant = dominant.merge(
        city_avg[
            [
                COL_CITY,
                "Lag_adjusted_PAO_L1_L4_mean",
                "Resilience_Score",
                "Mismatch_type",
                "Mismatch_type_CN",
            ]
        ].rename(
            columns={
                "Mismatch_type": "Long_term_average_type",
                "Mismatch_type_CN": "Long_term_average_type_CN",
            }
        ),
        on=COL_CITY,
        how="left",
    )
    return city_avg, dominant


def main() -> None:
    data = add_lag_adjusted_pao(load_data())
    pooled_pao_threshold = float(data["Lag_adjusted_PAO_L1_L4_mean"].median())
    pooled_resilience_threshold = float(data[COL_Y].median())
    pooled = add_classification(data, pooled_pao_threshold, pooled_resilience_threshold)

    city_avg, dominant = build_city_summaries(
        pooled,
        pooled_pao_threshold,
        pooled_resilience_threshold,
    )
    city_pao_threshold = pooled_pao_threshold
    city_resilience_threshold = pooled_resilience_threshold

    type_summary_pooled = summarize_types(pooled)
    type_summary_city_avg = summarize_types(city_avg, count_col="cities")
    dominant_summary = (
        dominant["Dominant_mismatch_type"]
        .value_counts()
        .reindex(TYPE_ORDER, fill_value=0)
        .rename("cities")
        .reset_index()
        .rename(columns={"index": "Dominant_mismatch_type"})
    )
    dominant_summary["Dominant_mismatch_type_CN"] = dominant_summary[
        "Dominant_mismatch_type"
    ].map(TYPE_CN)
    dominant_summary["share"] = dominant_summary["cities"] / dominant_summary["cities"].sum()
    yearly_counts = (
        pooled.groupby([COL_YEAR, "Mismatch_type"])
        .size()
        .rename("city_years")
        .reset_index()
        .pivot(index=COL_YEAR, columns="Mismatch_type", values="city_years")
        .reindex(columns=TYPE_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    thresholds = pd.DataFrame(
        [
            {
                "classification_level": "Annual city-year, fixed full-period threshold",
                "year_range": f"{pooled[COL_YEAR].min()}-{pooled[COL_YEAR].max()}",
                "lag_window": f"L{LAG_START}-L{LAG_END}",
                "PAO_threshold_median": pooled_pao_threshold,
                "resilience_threshold_median": pooled_resilience_threshold,
                "observations": pooled.shape[0],
                "cities": pooled[COL_CITY].nunique(),
            },
            {
                "classification_level": "City long-term average, same fixed threshold",
                "year_range": f"{pooled[COL_YEAR].min()}-{pooled[COL_YEAR].max()}",
                "lag_window": f"L{LAG_START}-L{LAG_END}",
                "PAO_threshold_median": city_pao_threshold,
                "resilience_threshold_median": city_resilience_threshold,
                "observations": city_avg.shape[0],
                "cities": city_avg[COL_CITY].nunique(),
            },
        ]
    )

    output_xlsx = OUTPUT_DIR / "lag_adjusted_PAO_resilience_mismatch_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        yearly_counts.to_excel(writer, sheet_name="annual_type_counts", index=False)
        pooled.to_excel(writer, sheet_name="city_year_classification", index=False)
        dominant.to_excel(writer, sheet_name="city_dominant_type", index=False)
        dominant_summary.to_excel(writer, sheet_name="dominant_type_summary", index=False)
        type_summary_pooled.to_excel(writer, sheet_name="pooled_type_summary", index=False)
        city_avg.to_excel(writer, sheet_name="city_average_supplement", index=False)
        type_summary_city_avg.to_excel(writer, sheet_name="city_avg_type_summary", index=False)
        thresholds.to_excel(writer, sheet_name="thresholds", index=False)



    note = f"""# Lag-adjusted PAO-resilience mismatch diagnostics

This diagnostic uses the PDL-supported lag window L{LAG_START}-L{LAG_END}.

Lag-adjusted PAO is defined as the city-year mean of PAO from t-{LAG_START} to t-{LAG_END}.
The current resilience score is Resilience_Score in `Coastal urban resilience.xlsx`.

The main classification is annual city-year mismatch identification for
{pooled[COL_YEAR].min()}-{pooled[COL_YEAR].max()}. City-level dominant type and
city-average classification are reported only as supplementary stability summaries.

The four-quadrant labels are descriptive heterogeneity diagnostics. They identify
alignment or mismatch between lagged PAO signals and current resilience performance;
they should not be interpreted as causal evidence that PAO directly changes resilience.
"""
    (OUTPUT_DIR / "lag_adjusted_mismatch_summary.md").write_text(note, encoding="utf-8")

    print(f"Saved lag-adjusted mismatch outputs to: {OUTPUT_DIR}")
    print("\nThresholds:")
    print(thresholds.to_string(index=False))
    print("\nPooled city-year type summary:")
    print(type_summary_pooled.to_string(index=False))
    print("\nAnnual type counts:")
    print(yearly_counts.to_string(index=False))
    print("\nCity dominant type summary:")
    print(dominant_summary.to_string(index=False))
    print("\nCity-average type summary:")
    print(type_summary_city_avg.to_string(index=False))


if __name__ == "__main__":
    main()
