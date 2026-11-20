#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from common import OUTPUT_ROOT, output_dir

xgb = importlib.import_module("11_XGBoost_SHAP")
COL_CITY, COL_YEAR, COL_Y = xgb.COL_CITY, xgb.COL_YEAR, xgb.COL_Y

OUTPUT_DIR = output_dir("12_XGBoost_alternative_resilience")

DEPENDENT_VARIABLES = {
    "Baseline BPNN-PSO": None,
    "MC-alpha mean BPNN-PSO": ("03_resilience_MC_alpha/baseline_resilience_MC_alpha_BPNN_PSO_results.xlsx",
                               "scores_and_MC_interval", "MC_alpha_mean_BPNN_PSO"),
    "AHP-EWM additive composite": ("02_resilience_index/resilience_bpnn_pso_sensitivity_results.xlsx",
                                   "scores", "B3_AHP_EWM_additive"),
    "AHP-EWM BPNN-PSO rerun": ("02_resilience_index/resilience_bpnn_pso_sensitivity_results.xlsx",
                               "scores", "B3_AHP_EWM_BPNN_PSO"),
}


TUNED_KEYS = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
              "min_child_weight", "reg_alpha", "reg_lambda"]


def panel_with_y(label: str) -> pd.DataFrame:
    panel = xgb.load_panel()
    spec = DEPENDENT_VARIABLES[label]
    if spec is None:
        return panel
    path, sheet, col = spec
    alt = pd.read_excel(OUTPUT_ROOT / path, sheet_name=sheet, usecols=[COL_CITY, COL_YEAR, col])
    panel = panel.drop(columns=[COL_Y]).merge(alt.rename(columns={col: COL_Y}), on=[COL_CITY, COL_YEAR], how="left")
    if panel[COL_Y].isna().any():
        raise ValueError(f"Missing alternative score for {label}.")
    return panel.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def evaluate(label: str) -> tuple[dict, dict]:
    model_df, feature_cols = xgb.add_lag_features(panel_with_y(label))
    train = model_df[model_df[COL_YEAR] <= xgb.TEMPORAL_TRAIN_END]
    test = model_df[model_df[COL_YEAR] >= xgb.TEMPORAL_TEST_START]

    tuned, _ = xgb.tune_xgb_with_group_cv(train[feature_cols], train[COL_Y].to_numpy(), train[COL_CITY])
    params = {key: value for key, value in tuned.get_params().items() if key in TUNED_KEYS}

    model = xgb.make_xgb_model(params)
    model.fit(train[feature_cols], train[COL_Y].to_numpy())
    temporal = xgb.evaluate_predictions(test[COL_Y].to_numpy(), model.predict(test[feature_cols]))

    values = xgb.load_shap().TreeExplainer(model).shap_values(test[feature_cols])
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=feature_cols)
    dims = mean_abs.groupby(mean_abs.index.str.extract(r"PAO_(Economic|Ecological|Social)_")[0].values).sum()
    shares = (dims / dims.sum()).to_dict()

    X, y = model_df[feature_cols], model_df[COL_Y].to_numpy()
    folds = []
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, model_df[COL_CITY]):
        m = xgb.make_xgb_model(params)
        m.fit(X.iloc[train_idx], y[train_idx])
        folds.append(xgb.evaluate_predictions(y[test_idx], m.predict(X.iloc[test_idx])))
    folds = pd.DataFrame(folds)

    perf = {"Dependent variable": label, **{f"Selected {k}": params[k] for k in TUNED_KEYS},
            "Temporal RMSE": temporal["RMSE"], "Temporal MAE": temporal["MAE"],
            "Temporal EVS": temporal["EVS"], "Temporal R2": temporal["R2"],
            "City-CV RMSE": folds["RMSE"].mean(), "City-CV R2": folds["R2"].mean()}
    order = sorted(shares, key=shares.get, reverse=True)
    shap_row = {"Dependent variable": label, "Social PAO": shares["Social"], "Economic PAO": shares["Economic"],
                "Ecological PAO": shares["Ecological"], "Ranking": " > ".join(order)}
    return perf, shap_row


def main() -> None:
    perf_rows, shap_rows = [], []
    for label in DEPENDENT_VARIABLES:
        perf, shap_row = evaluate(label)
        perf_rows.append(perf)
        shap_rows.append(shap_row)
    table_s24 = pd.DataFrame(perf_rows)
    table_s25 = pd.DataFrame(shap_rows)
    with pd.ExcelWriter(OUTPUT_DIR / "xgboost_alternative_resilience_results.xlsx", engine="openpyxl") as writer:
        table_s24.to_excel(writer, sheet_name="Table_S24_performance", index=False)
        table_s25.to_excel(writer, sheet_name="Table_S25_SHAP_shares", index=False)
    print("Table S24:")
    print(table_s24[[c for c in table_s24.columns if not c.startswith("Selected")]].to_string(index=False))
    print("\nTable S25:")
    print(table_s25.to_string(index=False))


if __name__ == "__main__":
    main()
