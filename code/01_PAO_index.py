#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from common import COL_CITY, COL_YEAR, COL_Y, load_model_base, output_dir, pao_versions

OUTPUT_DIR = output_dir("01_PAO_index")

PAO_COLUMNS = {
    "baseline_main_balanced": "PAO_baseline",
    "R1_raw_sum": "PAO_raw",
    "R2_no_political": "PAO_R2_no_political",
    "R3_market_sensitive": "PAO_R3_market_sensitive",
    "R4_tourism_event": "PAO_R4_tourism_event",
}


def correlations_with_baseline(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = sheets["baseline_main_balanced"][[COL_CITY, COL_YEAR, "PAO_baseline"]]
    rows = []
    for sheet, col in PAO_COLUMNS.items():
        if sheet == "baseline_main_balanced":
            continue
        merged = base.merge(sheets[sheet][[COL_CITY, COL_YEAR, col]], on=[COL_CITY, COL_YEAR])
        rows.append({
            "comparison": sheet,
            "baseline_column": "PAO_baseline",
            "robustness_column": col,
            "pearson_r": merged["PAO_baseline"].corr(merged[col], method="pearson"),
            "spearman_r": merged["PAO_baseline"].corr(merged[col], method="spearman"),
        })
    return pd.DataFrame(rows)


def descriptive_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pao = sheets["baseline_main_balanced"]
    df = load_model_base().merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    variables = {
        "PAO": df["PAO_baseline"],
        "Economic PAO": df["PAO_baseline_Economic"],
        "Ecological PAO": df["PAO_baseline_Ecological"],
        "Social PAO": df["PAO_baseline_Social"],
        "Resilience score": df[COL_Y],
    }
    rows = []
    for name, s in variables.items():
        rows.append({"Variable": name, "N": int(s.count()), "Mean": s.mean(), "SD": s.std(ddof=1),
                     "Min": s.min(), "Max": s.max()})
    return pd.DataFrame(rows)


def main() -> None:
    sheets, params = pao_versions()
    table_s7 = correlations_with_baseline(sheets)
    table_s16 = descriptive_table(sheets)

    with pd.ExcelWriter(OUTPUT_DIR / "PAO_index_results.xlsx", engine="openpyxl") as writer:
        table_s7.to_excel(writer, sheet_name="Table_S7_correlations", index=False)
        table_s16.to_excel(writer, sheet_name="Table_S16_descriptives", index=False)
        params.to_excel(writer, sheet_name="domain_parameters", index=False)
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Saved PAO outputs to: {OUTPUT_DIR}")
    print("\nTable S7:")
    print(table_s7.to_string(index=False))
    print("\nTable S16:")
    print(table_s16.to_string(index=False))


if __name__ == "__main__":
    main()
