#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from common import load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("11_XGBoost_SHAP")

COL_CITY = "城市"
COL_YEAR = "年份"
COL_Y = "Resilience_Score"
LAGS = (1, 2, 3, 4)
RANDOM_STATE = 42
SKLEARN_N_JOBS = 1
XGB_N_JOBS = 1
TEMPORAL_TRAIN_END = 2022
TEMPORAL_TEST_START = 2023

DIMENSION_COLS = {
    "Economic": "PAO_baseline_Economic",
    "Ecological": "PAO_baseline_Ecological",
    "Social": "PAO_baseline_Social",
}


def normalize_city(name: object) -> str:
    text = str(name).strip()
    return text if text.endswith("市") else f"{text}市"


def load_xgboost():
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "XGBoost could not be imported. On macOS this usually means the "
            "OpenMP runtime is missing. Install it with `brew install libomp`, "
            "then rerun this script."
        ) from exc
    return XGBRegressor


def load_shap():
    try:
        import shap
    except Exception as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("SHAP could not be imported. Install `shap` and rerun.") from exc
    return shap


def load_panel() -> pd.DataFrame:
    y = load_model_base()

    pao_cols = [COL_CITY, COL_YEAR] + list(DIMENSION_COLS.values())
    pao = read_pao("baseline_main_balanced", pao_cols)

    df = y.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != y.shape[0]:
        raise ValueError(f"PAO merge returned {df.shape[0]} rows; expected {y.shape[0]}.")

    df = df.rename(columns={v: f"PAO_{k}" for k, v in DIMENSION_COLS.items()})
    return df.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    feature_cols = []
    for dim in DIMENSION_COLS:
        base_col = f"PAO_{dim}"
        for lag in LAGS:
            col = f"{base_col}_L{lag}"
            out[col] = out.groupby(COL_CITY)[base_col].shift(lag)
            feature_cols.append(col)
    model_df = out.dropna(subset=feature_cols + [COL_Y]).copy()
    return model_df, feature_cols


def add_lead_features(df: pd.DataFrame, leads: tuple[int, ...] = (1, 2)) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    feature_cols = []
    for dim in DIMENSION_COLS:
        base_col = f"PAO_{dim}"
        for lead in leads:
            col = f"{base_col}_F{lead}"
            out[col] = out.groupby(COL_CITY)[base_col].shift(-lead)
            feature_cols.append(col)
    model_df = out.dropna(subset=feature_cols + [COL_Y]).copy()
    return model_df, feature_cols


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "EVS": float(explained_variance_score(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_xgb_model(params: dict | None = None):
    XGBRegressor = load_xgboost()
    base = {
        "objective": "reg:squarederror",
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "n_jobs": XGB_N_JOBS,
    }
    if params:
        base.update(params)
    return XGBRegressor(**base)


def make_xgb_grid() -> tuple[object, dict]:
    model = make_xgb_model()
    grid = {
        "n_estimators": [200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [3, 5],
        "reg_alpha": [0.1, 1.0],
        "reg_lambda": [5.0, 10.0],
    }
    return model, grid


def benchmark_models() -> dict[str, object]:
    return {
        "Dummy mean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=[0.1, 0.5, 0.9],
                        alphas=np.logspace(-4, 2, 30),
                        cv=5,
                        random_state=RANDOM_STATE,
                        max_iter=100_000,
                    ),
                ),
            ]
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=4,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=SKLEARN_N_JOBS,
        ),
    }


def tune_xgb_with_group_cv(
    X_train: pd.DataFrame, y_train: np.ndarray, groups_train: pd.Series
) -> tuple[object, pd.DataFrame]:
    model, grid = make_xgb_grid()
    cv = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        model,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=SKLEARN_N_JOBS,
        verbose=0,
        refit=True,
        return_train_score=True,
    )
    gs.fit(X_train, y_train, groups=groups_train)
    results = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    return gs.best_estimator_, results


def temporal_holdout(
    df: pd.DataFrame, feature_cols: list[str], xgb_model: object
) -> tuple[pd.DataFrame, dict[str, object]]:
    train = df[df[COL_YEAR] <= TEMPORAL_TRAIN_END].copy()
    test = df[df[COL_YEAR] >= TEMPORAL_TEST_START].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split is empty; check year range after lag construction.")

    X_train = train[feature_cols]
    y_train = train[COL_Y].to_numpy()
    X_test = test[feature_cols]
    y_test = test[COL_Y].to_numpy()

    rows = []
    fitted_models = {}
    for name, model in benchmark_models().items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_test)
        rows.append(
            {
                "Validation": "Temporal holdout",
                "Model": name,
                "Train years": f"{int(train[COL_YEAR].min())}-{int(train[COL_YEAR].max())}",
                "Test years": f"{int(test[COL_YEAR].min())}-{int(test[COL_YEAR].max())}",
                "N train": len(train),
                "N test": len(test),
                **evaluate_predictions(y_test, pred),
            }
        )
        fitted_models[name] = fitted

    xgb_model.fit(X_train, y_train)
    pred = xgb_model.predict(X_test)
    rows.append(
        {
            "Validation": "Temporal holdout",
            "Model": "XGBoost strict L1-L4",
            "Train years": f"{int(train[COL_YEAR].min())}-{int(train[COL_YEAR].max())}",
            "Test years": f"{int(test[COL_YEAR].min())}-{int(test[COL_YEAR].max())}",
            "N train": len(train),
            "N test": len(test),
            **evaluate_predictions(y_test, pred),
        }
    )
    fitted_models["XGBoost strict L1-L4"] = xgb_model

    context = {
        "train": train,
        "test": test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "xgb_model": xgb_model,
    }
    return pd.DataFrame(rows), context


def groupkfold_city_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_factory: Callable[[], object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df[feature_cols]
    y = df[COL_Y].to_numpy()
    groups = df[COL_CITY]
    cv = GroupKFold(n_splits=5)

    model_factories: dict[str, Callable[[], object]] = {
        name: (lambda model=model: clone(model)) for name, model in benchmark_models().items()
    }
    model_factories["XGBoost strict L1-L4"] = xgb_factory

    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_cities = sorted(df.iloc[test_idx][COL_CITY].unique())

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_rows.append(
                {
                    "Validation": "GroupKFold by city",
                    "Fold": fold,
                    "Model": name,
                    "N train": len(train_idx),
                    "N test": len(test_idx),
                    "N test cities": len(test_cities),
                    "Test cities": "、".join(test_cities),
                    **evaluate_predictions(y_test, pred),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(["Validation", "Model"], as_index=False)[["RMSE", "MAE", "EVS", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        " ".join([str(x) for x in col if str(x) != ""]).strip()
        for col in summary.columns.to_flat_index()
    ]
    return fold_df, summary


def lead_placebo_diagnostic(df: pd.DataFrame, best_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_df, lead_cols = add_lead_features(df)
    train = lead_df[lead_df[COL_YEAR] <= 2020].copy()
    test = lead_df[lead_df[COL_YEAR] >= 2021].copy()

    rows = []
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    X_train = train[lead_cols]
    y_train = train[COL_Y].to_numpy()
    X_test = test[lead_cols]
    y_test = test[COL_Y].to_numpy()

    for name, model in benchmark_models().items():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        pred = fitted.predict(X_test)
        rows.append(
            {
                "Diagnostic": "Lead placebo",
                "Model": name,
                "Train years": f"{int(train[COL_YEAR].min())}-{int(train[COL_YEAR].max())}",
                "Test years": f"{int(test[COL_YEAR].min())}-{int(test[COL_YEAR].max())}",
                "N train": len(train),
                "N test": len(test),
                **evaluate_predictions(y_test, pred),
            }
        )

    xgb = make_xgb_model(best_params)
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    rows.append(
        {
            "Diagnostic": "Lead placebo",
            "Model": "XGBoost lead placebo",
            "Train years": f"{int(train[COL_YEAR].min())}-{int(train[COL_YEAR].max())}",
            "Test years": f"{int(test[COL_YEAR].min())}-{int(test[COL_YEAR].max())}",
            "N train": len(train),
            "N test": len(test),
            **evaluate_predictions(y_test, pred),
        }
    )

    feature_map = pd.DataFrame({"feature": lead_cols})
    feature_map["dimension"] = feature_map["feature"].str.extract(r"PAO_(Economic|Ecological|Social)_")
    feature_map["lead"] = feature_map["feature"].str.extract(r"_F(\d+)").astype(int)
    return pd.DataFrame(rows), feature_map


def shap_summary(context: dict[str, object], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap = load_shap()
    model = context["xgb_model"]
    X_test = context["X_test"]

    try:
        values = shap.TreeExplainer(model).shap_values(X_test)
    except NotFittedError:
        model.fit(context["X_train"], context["y_train"])
        values = shap.TreeExplainer(model).shap_values(X_test)

    shap_df = pd.DataFrame(values, columns=feature_cols, index=X_test.index)
    long = shap_df.abs().mean(axis=0).reset_index()
    long.columns = ["feature", "mean_abs_SHAP"]
    long["dimension"] = long["feature"].str.extract(r"PAO_(Economic|Ecological|Social)_")
    long["lag"] = long["feature"].str.extract(r"_L(\d+)").astype(int)
    long = long.sort_values("mean_abs_SHAP", ascending=False)
    long["share"] = long["mean_abs_SHAP"] / long["mean_abs_SHAP"].sum()

    by_dimension = (
        long.groupby("dimension", as_index=False)["mean_abs_SHAP"].sum().sort_values(
            "mean_abs_SHAP", ascending=False
        )
    )
    by_dimension["share"] = by_dimension["mean_abs_SHAP"] / by_dimension["mean_abs_SHAP"].sum()
    return long, by_dimension


def shap_by_lag(shap_feature: pd.DataFrame) -> pd.DataFrame:
    """Table S22: share of total mean |SHAP| by lag year."""
    out = shap_feature.groupby("lag", as_index=False)["mean_abs_SHAP"].sum().sort_values("lag")
    out["share"] = out["mean_abs_SHAP"] / out["mean_abs_SHAP"].sum()
    return out


def shap_cv_stability(df: pd.DataFrame, feature_cols: list[str], best_params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Table S23: SHAP shares by dimension recomputed on held-out cities in each GroupKFold fold."""
    shap = load_shap()
    X = df[feature_cols]
    y = df[COL_Y].to_numpy()
    rows = []
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(X, y, df[COL_CITY]), start=1):
        model = make_xgb_model(best_params)
        model.fit(X.iloc[train_idx], y[train_idx])
        values = shap.TreeExplainer(model).shap_values(X.iloc[test_idx])
        mean_abs = pd.Series(np.abs(values).mean(axis=0), index=feature_cols)
        dims = mean_abs.groupby(mean_abs.index.str.extract(r"PAO_(Economic|Ecological|Social)_")[0].values).sum()
        for dim, value in (dims / dims.sum()).items():
            rows.append({"Fold": fold, "dimension": dim, "share": float(value)})
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("dimension")["share"].agg(["mean", "std", "min", "max"])
        .reindex(["Social", "Economic", "Ecological"]).reset_index()
        .rename(columns={"mean": "Mean SHAP share", "std": "SD", "min": "Min", "max": "Max"})
    )
    return folds, summary


def fig3_data(context: dict[str, object], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fig. 3: SHAP values and feature values on the 2023-2024 test set, and the
    dimension-level dependence data (mean PAO over L1-L4 vs. summed SHAP over L1-L4)."""
    shap = load_shap()
    test = context["test"]
    values = shap.TreeExplainer(context["xgb_model"]).shap_values(context["X_test"])
    meta = test[[COL_CITY, COL_YEAR, COL_Y]].reset_index(drop=True)
    shap_df = pd.concat([meta, pd.DataFrame(values, columns=feature_cols)], axis=1)
    x_df = pd.concat([meta, test[feature_cols].reset_index(drop=True)], axis=1)
    dep = meta.copy()
    for dim in DIMENSION_COLS:
        lag_cols = [f"PAO_{dim}_L{lag}" for lag in LAGS]
        dep[f"PAO_{dim}_L1_L4_mean"] = test[lag_cols].mean(axis=1).to_numpy()
        dep[f"SHAP_{dim}_L1_L4_sum"] = shap_df[lag_cols].sum(axis=1).to_numpy()
    return shap_df, x_df, dep


def feature_dictionary(feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        dim = col.split("_")[1]
        lag = int(col.rsplit("_L", 1)[1])
        rows.append(
            {
                "feature": col,
                "dimension": dim,
                "lag": lag,
                "description": f"{dim} PAO at t-{lag}",
            }
        )
    return pd.DataFrame(rows)


def write_summary_note(
    best_params: dict,
    temporal_results: pd.DataFrame,
    group_summary: pd.DataFrame,
    lead_results: pd.DataFrame,
) -> None:
    xgb_temporal = temporal_results[temporal_results["Model"] == "XGBoost strict L1-L4"].iloc[0]
    note = f"""# Strict XGBoost PAO Validation

Purpose: test whether lagged PAO contains useful predictive information for subsequent resilience under stricter validation. This analysis does not claim causal effects.

Predictors: Economic, Ecological, and Social PAO lags L1-L4 only. Contemporaneous PAO, future PAO, rolling features, keyword-level PCA features, and all-sample PCA transformations are excluded.

Temporal holdout split: train on {2018}-{TEMPORAL_TRAIN_END} and test on {TEMPORAL_TEST_START}-{2024}, giving an approximately 70/30 year-based split after L1-L4 lag construction.

Best XGBoost parameters selected on the temporal-training sample using GroupKFold by city:
`{best_params}`

Temporal holdout XGBoost:
- RMSE = {xgb_temporal['RMSE']:.4f}
- MAE = {xgb_temporal['MAE']:.4f}
- EVS = {xgb_temporal['EVS']:.4f}
- R2 = {xgb_temporal['R2']:.4f}

SHAP note: SHAP values are interpreted as model-attributed predictive contributions, not independent causal effects.

Lead-placebo note: Future PAO features are retained only as a diagnostic for common trends, reverse timing, or residual information leakage; they should not be used as evidence of causality.
"""
    if not lead_results.empty:
        xgb_lead = lead_results[lead_results["Model"] == "XGBoost lead placebo"].iloc[0]
        note += (
            f"\nLead-placebo XGBoost: RMSE = {xgb_lead['RMSE']:.4f}, "
            f"MAE = {xgb_lead['MAE']:.4f}, EVS = {xgb_lead['EVS']:.4f}, "
            f"R2 = {xgb_lead['R2']:.4f}.\n"
        )

    note += "\nThe earlier exploratory notebook result (R2 around 0.94) should be treated as exploratory because it used random splitting, all-sample PCA transformations, contemporaneous PC features, rolling features, and lead-placebo diagnostics with high predictive power.\n"
    (OUTPUT_DIR / "xgboost_strict_validation_summary.md").write_text(note, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PAO/resilience panel...", flush=True)
    panel = load_panel()
    model_df, feature_cols = add_lag_features(panel)

    temporal_train = model_df[model_df[COL_YEAR] <= TEMPORAL_TRAIN_END]
    print(
        f"Tuning strict XGBoost on {len(temporal_train)} rows, "
        f"{temporal_train[COL_CITY].nunique()} cities, {len(feature_cols)} lag features...",
        flush=True,
    )
    best_xgb, xgb_cv_results = tune_xgb_with_group_cv(
        temporal_train[feature_cols],
        temporal_train[COL_Y].to_numpy(),
        temporal_train[COL_CITY],
    )
    best_params = best_xgb.get_params()
    param_names = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "tree_method",
        "random_state",
    ]
    best_param_table = pd.DataFrame(
        [{"parameter": key, "value": best_params.get(key)} for key in param_names]
    )

    print("Running temporal holdout and benchmark models...", flush=True)
    temporal_results, shap_context = temporal_holdout(model_df, feature_cols, best_xgb)
    strict_best_params = {
        key: best_params[key]
        for key in [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ]
    }
    print("Running GroupKFold-by-city validation...", flush=True)
    fold_results, fold_summary = groupkfold_city_validation(
        model_df, feature_cols, lambda: make_xgb_model(strict_best_params)
    )
    print("Running lead-placebo diagnostic...", flush=True)
    lead_results, lead_feature_map = lead_placebo_diagnostic(panel, strict_best_params)
    print("Computing SHAP predictive contributions...", flush=True)
    shap_feature, shap_dimension = shap_summary(shap_context, feature_cols)
    shap_lag = shap_by_lag(shap_feature)
    print("Computing city-grouped SHAP stability...", flush=True)
    shap_cv_folds, shap_cv_summary = shap_cv_stability(model_df, feature_cols, strict_best_params)
    fig3_shap, fig3_x, fig3_dep = fig3_data(shap_context, feature_cols)

    data_audit = pd.DataFrame(
        [
            {
                "Item": "Raw panel rows",
                "Value": len(panel),
            },
            {
                "Item": "Cities",
                "Value": panel[COL_CITY].nunique(),
            },
            {
                "Item": "Raw year range",
                "Value": f"{int(panel[COL_YEAR].min())}-{int(panel[COL_YEAR].max())}",
            },
            {
                "Item": "Model rows after L1-L4",
                "Value": len(model_df),
            },
            {
                "Item": "Model year range after L1-L4",
                "Value": f"{int(model_df[COL_YEAR].min())}-{int(model_df[COL_YEAR].max())}",
            },
            {
                "Item": "Predictors",
                "Value": len(feature_cols),
            },
            {
                "Item": "Excluded features",
                "Value": "Current PAO, future PAO, rolling features, keyword PCA features",
            },
        ]
    )

    output_xlsx = OUTPUT_DIR / "xgboost_strict_validation_results.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        data_audit.to_excel(writer, sheet_name="data_audit", index=False)
        feature_dictionary(feature_cols).to_excel(writer, sheet_name="feature_dictionary", index=False)
        best_param_table.to_excel(writer, sheet_name="xgb_best_params", index=False)
        xgb_cv_results.head(50).to_excel(writer, sheet_name="xgb_tuning_top50", index=False)
        temporal_results.to_excel(writer, sheet_name="temporal_holdout", index=False)
        fold_summary.to_excel(writer, sheet_name="groupkfold_summary", index=False)
        fold_results.to_excel(writer, sheet_name="groupkfold_folds", index=False)
        shap_feature.to_excel(writer, sheet_name="shap_by_feature", index=False)
        shap_dimension.to_excel(writer, sheet_name="shap_by_dimension", index=False)
        shap_lag.to_excel(writer, sheet_name="shap_by_lag", index=False)
        shap_cv_summary.to_excel(writer, sheet_name="shap_cv_stability", index=False)
        shap_cv_folds.to_excel(writer, sheet_name="shap_cv_folds", index=False)
        fig3_shap.to_excel(writer, sheet_name="Fig3_SHAP_values", index=False)
        fig3_x.to_excel(writer, sheet_name="Fig3_feature_values", index=False)
        fig3_dep.to_excel(writer, sheet_name="Fig3b_dependence", index=False)
        lead_results.to_excel(writer, sheet_name="lead_placebo", index=False)
        lead_feature_map.to_excel(writer, sheet_name="lead_feature_map", index=False)

    write_summary_note(strict_best_params, temporal_results, fold_summary, lead_results)

    print(f"Saved strict XGBoost outputs to: {OUTPUT_DIR}")
    print("\nBest strict XGBoost params:")
    print(best_param_table.to_string(index=False))
    print("\nTemporal holdout:")
    print(temporal_results.to_string(index=False))
    print("\nGroupKFold by city summary:")
    print(fold_summary.to_string(index=False))
    print("\nLead-placebo diagnostic:")
    print(lead_results.to_string(index=False))


if __name__ == "__main__":
    main()
