#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

from common import (
    COL_CITY, COL_YEAR, COL_Y as COL_ORIGINAL_Y, PAO_SCALE, POLY_DEGREE,
    load_resilience, p_value_from_z, read_pao, stars,
)

PAO_COL = "PAO_baseline"
LAGS = [1, 2, 3, 4]
MC_DRAWS = 1000
MC_PERTURB = 0.20
RANDOM_STATE = 42


INDICATOR_SYSTEM = OrderedDict(
    {
        "Economic": [
            {
                "indicator": "海洋相关经济活动强度（万元）",
                "label": "Marine-related economic activity intensity",
                "symbol": "MRIA",
                "capacity": "Ability to withstand and absorb economic risks",
            },
            {
                "indicator": "第三产业增加值占GDP比重(%)",
                "label": "Industrial structure diversity",
                "symbol": "ISD",
                "capacity": "Ability to maintain stability and recover",
            },
            {
                "indicator": "社会消费品零售总额(万元)",
                "label": "Market vitality",
                "symbol": "MV",
                "capacity": "Ability to support recovery",
            },
        ],
        "Social": [
            {
                "indicator": "教育支出(万元)",
                "label": "Education level",
                "symbol": "EL",
                "capacity": "Safeguarding equity and basic services",
            },
            {
                "indicator": "医院、卫生院数(个)",
                "label": "Medical resources",
                "symbol": "MR",
                "capacity": "Basic service capacity",
            },
            {
                "indicator": "失业保险参保人数(人)",
                "label": "Social security coverage",
                "symbol": "SSC",
                "capacity": "Social protection capacity",
            },
        ],
        "Infrastructure": [
            {
                "indicator": "轨道交通/公共交通支出_万元",
                "label": "Transportation infrastructure",
                "symbol": "TI",
                "capacity": "Maintaining critical systems",
            },
            {
                "indicator": "移动电话年末用户数(万户)",
                "label": "Communication infrastructure",
                "symbol": "CI",
                "capacity": "Information and response capacity",
            },
            {
                "indicator": "本年城市市政公用设施建设固定资产投资完成额_万元",
                "label": "Infrastructure completeness and disaster resistance",
                "symbol": "ICDR",
                "capacity": "Repair and recovery capacity",
            },
        ],
        "Ecology_Environment": [
            {
                "indicator": "污水处理厂集中处理率(%)",
                "label": "Ecological buffering",
                "symbol": "EB",
                "capacity": "Environmental risk buffering",
            },
            {
                "indicator": "城市全部供水-供水总量_万立方米",
                "label": "Water resource security",
                "symbol": "WRS",
                "capacity": "Resource security",
            },
            {
                "indicator": "绿化覆盖面积_公顷",
                "label": "Green coverage rate",
                "symbol": "GCR",
                "capacity": "Ecological buffering",
            },
        ],
        "Institution_Governance": [
            {
                "indicator": "词频总和",
                "label": "Local government governance attention",
                "symbol": "LGGA",
                "capacity": "Resource coordination and adaptive adjustment",
            },
            {
                "indicator": "绿色专利授权总量（个）",
                "label": "Green patents granted",
                "symbol": "GPG",
                "capacity": "Institutional innovation and green transition",
            },
        ],
    }
)

AHP_WEIGHT_BY_INDICATOR = {
    "海洋相关经济活动强度（万元）": 0.084853,
    "第三产业增加值占GDP比重(%)": 0.075220,
    "社会消费品零售总额(万元)": 0.051325,
    "教育支出(万元)": 0.060839,
    "医院、卫生院数(个)": 0.076900,
    "失业保险参保人数(人)": 0.046622,
    "轨道交通/公共交通支出_万元": 0.083175,
    "移动电话年末用户数(万户)": 0.049885,
    "本年城市市政公用设施建设固定资产投资完成额_万元": 0.071214,
    "污水处理厂集中处理率(%)": 0.097286,
    "城市全部供水-供水总量_万立方米": 0.077188,
    "绿化覆盖面积_公顷": 0.063601,
    "绿色专利授权总量（个）": 0.072611,
    "词频总和": 0.089281,
}


def indicator_columns() -> list[str]:
    return [item["indicator"] for entries in INDICATOR_SYSTEM.values() for item in entries]


def minmax_series(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=float).reshape(-1, 1)
    return pd.Series(MinMaxScaler().fit_transform(arr).ravel(), index=values.index)


def load_and_clean_data() -> pd.DataFrame:
    """Indicators + baseline score. Missing indicator values are filled with the within-city
    median and then the all-observation median (the published data file already contains
    these filled values, so this step changes nothing when that file is used)."""
    cols = [COL_CITY, COL_YEAR, COL_ORIGINAL_Y] + indicator_columns()
    df = load_resilience(cols)
    df[indicator_columns()] = df.groupby(COL_CITY)[indicator_columns()].transform(
        lambda g: g.fillna(g.median())
    )
    for col in indicator_columns():
        df[col] = df[col].fillna(df[col].median())
    if df[indicator_columns() + [COL_ORIGINAL_Y]].isna().any().any():
        raise ValueError("Cleaned resilience input still contains missing values.")
    return df.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def normalized_indicator_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[indicator_columns()].copy()
    scaled = MinMaxScaler().fit_transform(x.to_numpy(dtype=float))
    return pd.DataFrame(scaled, columns=indicator_columns(), index=df.index)


def entropy_weights(x_norm: pd.DataFrame) -> pd.Series:
    data = x_norm.to_numpy(dtype=float)
    n, _ = data.shape
    data = np.where(data <= 0, 1e-10, data)
    p = data / np.sum(data, axis=0)
    p = np.where(p <= 0, 1e-10, p)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(n)
    weights = (1.0 - entropy) / np.sum(1.0 - entropy)
    return pd.Series(weights, index=x_norm.columns, name="EWM_weight")


def ahp_weights() -> pd.Series:
    s = pd.Series(AHP_WEIGHT_BY_INDICATOR, dtype=float)
    s = s.reindex(indicator_columns())
    if s.isna().any():
        raise ValueError(f"Missing AHP weights: {s[s.isna()].index.tolist()}")
    return (s / s.sum()).rename("AHP_weight")


def equal_dimension_score(x_norm: pd.DataFrame) -> pd.Series:
    dim_scores = []
    for dim, entries in INDICATOR_SYSTEM.items():
        cols = [item["indicator"] for item in entries]
        dim_scores.append(x_norm[cols].mean(axis=1).rename(dim))
    return pd.concat(dim_scores, axis=1).mean(axis=1)


def weighted_score(x_norm: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(x_norm.columns).to_numpy(dtype=float)
    w = w / w.sum()
    score = np.sum(x_norm.to_numpy(dtype=float) * w.reshape(1, -1), axis=1)
    return pd.Series(score, index=x_norm.index)


def monte_carlo_scores(
    x_norm: pd.DataFrame, baseline_weights: pd.Series
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    base = baseline_weights.reindex(x_norm.columns).to_numpy(dtype=float)
    draws = []
    weight_rows = []
    x_arr = x_norm.to_numpy(dtype=float)
    for draw in range(1, MC_DRAWS + 1):
        eps = rng.uniform(-MC_PERTURB, MC_PERTURB, size=base.shape[0])
        w = base * (1.0 + eps)
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        draws.append(np.sum(x_arr * w.reshape(1, -1), axis=1))
        weight_rows.append({"draw": draw, **dict(zip(x_norm.columns, w))})
    draw_arr = np.column_stack(draws)
    mc_mean = pd.Series(draw_arr.mean(axis=1), index=x_norm.index, name="R5_MC_mean")
    mc_ci = pd.DataFrame(
        {
            "R5_MC_p025": np.quantile(draw_arr, 0.025, axis=1),
            "R5_MC_p975": np.quantile(draw_arr, 0.975, axis=1),
            "R5_MC_sd": draw_arr.std(axis=1, ddof=1),
        },
        index=x_norm.index,
    )
    return mc_mean, mc_ci, pd.DataFrame(weight_rows)


def build_resilience_versions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_and_clean_data()
    x_norm = normalized_indicator_matrix(df)
    ahp = ahp_weights()
    ewm = entropy_weights(x_norm)
    ahp_ewm = (0.5 * ahp + 0.5 * ewm).rename("AHP_EWM_weight")

    scores = df[[COL_CITY, COL_YEAR, COL_ORIGINAL_Y]].copy()
    scores = scores.rename(columns={COL_ORIGINAL_Y: "R0_BPNN_PSO_original"})
    scores["R1_equal_dimension"] = minmax_series(equal_dimension_score(x_norm))
    scores["R2_AHP_only"] = minmax_series(weighted_score(x_norm, ahp))
    scores["R3_EWM_only"] = minmax_series(weighted_score(x_norm, ewm))
    scores["R4_AHP_EWM_additive"] = minmax_series(weighted_score(x_norm, ahp_ewm))
    mc_mean, mc_ci, mc_weights = monte_carlo_scores(x_norm, ahp_ewm)
    scores["R5_MC_mean"] = minmax_series(mc_mean)
    scores = pd.concat([scores, mc_ci], axis=1)

    weight_table = pd.DataFrame(
        {
            "indicator": indicator_columns(),
            "dimension": [
                dim for dim, entries in INDICATOR_SYSTEM.items() for _ in entries
            ],
            "AHP_weight": ahp.reindex(indicator_columns()).to_numpy(),
            "EWM_weight": ewm.reindex(indicator_columns()).to_numpy(),
            "AHP_EWM_weight": ahp_ewm.reindex(indicator_columns()).to_numpy(),
        }
    )
    weight_table["equal_dimension_weight"] = weight_table["dimension"].map(
        {
            dim: 1.0 / len(INDICATOR_SYSTEM) / len(entries)
            for dim, entries in INDICATOR_SYSTEM.items()
        }
    )

    indicator_table = pd.DataFrame(
        [
            {"dimension": dim, **item}
            for dim, entries in INDICATOR_SYSTEM.items()
            for item in entries
        ]
    )
    return scores, weight_table, indicator_table, mc_weights


def score_columns() -> list[str]:
    return [
        "R0_BPNN_PSO_original",
        "R1_equal_dimension",
        "R2_AHP_only",
        "R3_EWM_only",
        "R4_AHP_EWM_additive",
        "R5_MC_mean",
    ]


def score_correlations(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = score_columns()
    pearson = scores[cols].corr(method="pearson")
    spearman = scores[cols].corr(method="spearman")
    rows = []
    for col in cols:
        if col == "R0_BPNN_PSO_original":
            continue
        rows.append(
            {
                "Alternative resilience score": col,
                "Pearson r with original": scores["R0_BPNN_PSO_original"].corr(
                    scores[col], method="pearson"
                ),
                "Spearman r with original": scores["R0_BPNN_PSO_original"].corr(
                    scores[col], method="spearman"
                ),
            }
        )
    return pearson, spearman, pd.DataFrame(rows)


def yearly_rank_consistency(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in score_columns():
        if col == "R0_BPNN_PSO_original":
            continue
        yearly = []
        for year, g in scores.groupby(COL_YEAR):
            rho = g["R0_BPNN_PSO_original"].corr(g[col], method="spearman")
            yearly.append(rho)
            rows.append(
                {
                    "Alternative resilience score": col,
                    "year": int(year),
                    "yearly_spearman_with_original": rho,
                }
            )
        rows.append(
            {
                "Alternative resilience score": col,
                "year": "Summary_mean",
                "yearly_spearman_with_original": float(np.nanmean(yearly)),
            }
        )
        rows.append(
            {
                "Alternative resilience score": col,
                "year": "Summary_min",
                "yearly_spearman_with_original": float(np.nanmin(yearly)),
            }
        )
    return pd.DataFrame(rows)


def load_pao() -> pd.DataFrame:
    pao = read_pao("baseline_main_balanced", [COL_CITY, COL_YEAR, PAO_COL])
    pao["PAO_scaled"] = pao[PAO_COL] / PAO_SCALE
    return pao


def fit_centered_almon_pdl(df_in: pd.DataFrame, y_col: str) -> dict:
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
        var = float(g @ vcov.values @ g.T)
        return float(np.sqrt(max(var, 0.0)))

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
    direction_sign = np.sign(cumulative) if cumulative != 0 else np.sign(grid_effects[np.argmax(np.abs(grid_effects))])
    grid_idx = int(np.argmax(grid_effects) if direction_sign >= 0 else np.argmin(grid_effects))
    peak_lag = float(lag_grid[grid_idx])
    peak_effect = float(grid_effects[grid_idx])
    peak_se = lag_se(peak_lag)
    peak_p = p_value_from_z(peak_effect, peak_se)

    cumulative_weights = np.array(
        [float(np.sum(centered**degree)) for degree in range(POLY_DEGREE + 1)]
    )
    cumulative_se = float(np.sqrt(max(float(cumulative_weights @ vcov.values @ cumulative_weights.T), 0.0)))
    cumulative_p = p_value_from_z(cumulative, cumulative_se)
    joint = result.wald_test("Z0 = 0, Z1 = 0, Z2 = 0", scalar=True)

    return {
        "summary": {
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
            "Interpretation": build_pdl_interpretation(y_col, peak_lag, cumulative, cumulative_p, float(joint.pvalue)),
        },
        "lag_effects": lag_df,
    }


def build_pdl_interpretation(
    y_col: str, peak_lag: float, cumulative: float, cumulative_p: float, joint_p: float
) -> str:
    direction = "positive" if cumulative > 0 else "negative" if cumulative < 0 else "near-zero"
    stable_peak = 1.5 <= peak_lag <= 3.3
    significant = (cumulative_p < 0.1) or (joint_p < 0.1)
    if y_col == "R0_BPNN_PSO_original":
        return "Main BPNN-PSO dependent variable."
    if stable_peak and significant:
        return f"Robust {direction} medium-term association under alternative resilience construction."
    if stable_peak:
        return f"Peak timing remains similar, but statistical strength changes; cumulative direction is {direction}."
    return f"Lag profile changes under this resilience construction; cumulative direction is {direction}."


def pdl_robustness(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pao = load_pao()
    df = scores.merge(pao[[COL_CITY, COL_YEAR, "PAO_scaled"]], on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != scores.shape[0]:
        raise ValueError(f"PAO merge returned {df.shape[0]} rows; expected {scores.shape[0]}.")
    summaries = []
    lag_frames = []
    for y_col in score_columns():
        fitted = fit_centered_almon_pdl(df, y_col)
        summaries.append(fitted["summary"])
        lag_frames.append(fitted["lag_effects"])
    return pd.DataFrame(summaries), pd.concat(lag_frames, ignore_index=True)


BPNN_HIDDEN_SIZE = 10
BPNN_LEARNING_RATE = 0.01
BPNN_EPOCHS = 1000
PSO_PARTICLES = 20
PSO_ITERATIONS = 50
PSO_BOUNDS = (-5.0, 5.0)
PSO_INERTIA = 0.7
PSO_C1 = 2.0
PSO_C2 = 2.0
FIG_DPI = 600


class BPNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        learning_rate: float = BPNN_LEARNING_RATE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        rng = rng or np.random.default_rng(RANDOM_STATE)
        self.weights1 = rng.normal(0, 0.1, (input_size, hidden_size))
        self.bias1 = rng.normal(0, 0.1, hidden_size)
        self.weights2 = rng.normal(0, 0.1, (hidden_size, output_size))
        self.bias2 = rng.normal(0, 0.1, output_size)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        return a * (1.0 - a)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = np.sum(x[:, :, None] * self.weights1[None, :, :], axis=1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.sum(self.a1[:, :, None] * self.weights2[None, :, :], axis=1) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(x)
        return float(np.mean((pred - y) ** 2))

    def backward(self, x: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
        n = x.shape[0]
        delta2 = (pred - y) * self.sigmoid_derivative(pred)
        d_w2 = np.sum(self.a1[:, :, None] * delta2[:, None, :], axis=0) / n
        d_b2 = np.sum(delta2, axis=0) / n
        back2 = np.sum(delta2[:, :, None] * self.weights2.T[None, :, :], axis=1)
        delta1 = back2 * self.sigmoid_derivative(self.a1)
        d_w1 = np.sum(x[:, :, None] * delta1[:, None, :], axis=0) / n
        d_b1 = np.sum(delta1, axis=0) / n
        self.weights2 -= self.learning_rate * d_w2
        self.bias2 -= self.learning_rate * d_b2
        self.weights1 -= self.learning_rate * d_w1
        self.bias1 -= self.learning_rate * d_b1

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int) -> list[float]:
        history = []
        for epoch in range(1, epochs + 1):
            pred = self.forward(x)
            self.backward(x, y, pred)
            if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                history.append(float(np.mean((self.forward(x) - y) ** 2)))
        return history

    def parameter_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                self.weights1.ravel(),
                self.bias1.ravel(),
                self.weights2.ravel(),
                self.bias2.ravel(),
            ]
        )

    def set_parameter_vector(self, vector: np.ndarray) -> None:
        idx = 0
        n_w1 = self.input_size * self.hidden_size
        self.weights1 = vector[idx : idx + n_w1].reshape(self.input_size, self.hidden_size)
        idx += n_w1
        self.bias1 = vector[idx : idx + self.hidden_size]
        idx += self.hidden_size
        n_w2 = self.hidden_size * self.output_size
        self.weights2 = vector[idx : idx + n_w2].reshape(self.hidden_size, self.output_size)
        idx += n_w2
        self.bias2 = vector[idx : idx + self.output_size]


class PSO:
    def __init__(
        self,
        n_particles: int,
        dims: int,
        bounds: tuple[float, float],
        rng: np.random.Generator,
    ) -> None:
        self.n_particles = n_particles
        self.dims = dims
        self.bounds = bounds
        self.rng = rng
        self.positions = rng.uniform(bounds[0], bounds[1], (n_particles, dims))
        self.velocities = rng.uniform(-1, 1, (n_particles, dims))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, np.inf)
        self.gbest_position = self.positions[0].copy()
        self.gbest_score = np.inf

    def optimize(self, bpnn: BPNN, x: np.ndarray, y: np.ndarray, iterations: int) -> list[float]:
        history = []
        for iteration in range(1, iterations + 1):
            for i in range(self.n_particles):
                bpnn.set_parameter_vector(self.positions[i])
                score = bpnn.mse(x, y)
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()

            r1 = self.rng.random((self.n_particles, self.dims))
            r2 = self.rng.random((self.n_particles, self.dims))
            self.velocities = (
                PSO_INERTIA * self.velocities
                + PSO_C1 * r1 * (self.pbest_positions - self.positions)
                + PSO_C2 * r2 * (self.gbest_position - self.positions)
            )
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.bounds[0], self.bounds[1])
            if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
                history.append(float(self.gbest_score))

        bpnn.set_parameter_vector(self.gbest_position)
        return history


def make_weight_schemes(
    x_norm: pd.DataFrame,
) -> tuple[dict[str, pd.Series], pd.DataFrame, pd.DataFrame]:
    ahp = ahp_weights()
    ewm = entropy_weights(x_norm)
    ahp_ewm = (0.5 * ahp + 0.5 * ewm).rename("AHP_EWM_weight")
    _, _, mc_weights = monte_carlo_scores(x_norm, ahp_ewm)
    mc_mean_weight = mc_weights.drop(columns=["draw"]).mean(axis=0)
    mc_mean_weight = (mc_mean_weight / mc_mean_weight.sum()).rename("MC_mean_weight")

    equal_weights = {}
    for dim, entries in INDICATOR_SYSTEM.items():
        for item in entries:
            equal_weights[item["indicator"]] = 1.0 / len(INDICATOR_SYSTEM) / len(entries)
    equal = pd.Series(equal_weights).reindex(indicator_columns()).rename("Equal_dimension_weight")

    schemes = {
        "B0_equal_dimension_BPNN_PSO": equal,
        "B1_AHP_only_BPNN_PSO": ahp,
        "B2_EWM_only_BPNN_PSO": ewm,
        "B3_AHP_EWM_BPNN_PSO": ahp_ewm,
        "B4_MC_mean_BPNN_PSO": mc_mean_weight,
    }
    weights = pd.DataFrame({"indicator": indicator_columns()})
    weights["dimension"] = [
        dim for dim, entries in INDICATOR_SYSTEM.items() for _ in entries
    ]
    for name, w in schemes.items():
        weights[name] = w.reindex(indicator_columns()).to_numpy(dtype=float)
    return schemes, weights, mc_weights


def monte_carlo_additive_interval(
    x_norm: pd.DataFrame, mc_weights: pd.DataFrame, index_df: pd.DataFrame
) -> pd.DataFrame:
    x_arr = x_norm.to_numpy(dtype=float)
    draws = []
    for _, row in mc_weights.drop(columns=["draw"]).iterrows():
        w = row.reindex(x_norm.columns).to_numpy(dtype=float)
        w = w / w.sum()
        score = np.sum(x_arr * w.reshape(1, -1), axis=1)
        score = MinMaxScaler().fit_transform(score.reshape(-1, 1)).ravel()
        draws.append(score)
    draw_arr = np.column_stack(draws)
    out = index_df[[COL_CITY, COL_YEAR]].copy()
    out["MC_additive_mean"] = draw_arr.mean(axis=1)
    out["MC_additive_p025"] = np.quantile(draw_arr, 0.025, axis=1)
    out["MC_additive_p975"] = np.quantile(draw_arr, 0.975, axis=1)
    out["MC_additive_sd"] = draw_arr.std(axis=1, ddof=1)
    return out


def run_bpnn_pso_version(
    x_norm: pd.DataFrame, weights: pd.Series, version: str, seed_offset: int
) -> tuple[pd.Series, dict, pd.DataFrame]:
    w = weights.reindex(x_norm.columns).to_numpy(dtype=float)
    w = w / w.sum()
    x_weighted = x_norm.to_numpy(dtype=float) * w.reshape(1, -1)
    y = np.sum(x_weighted, axis=1).reshape(-1, 1)
    y = MinMaxScaler().fit_transform(y)

    rng = np.random.default_rng(RANDOM_STATE + seed_offset)
    bpnn = BPNN(
        input_size=x_weighted.shape[1],
        hidden_size=BPNN_HIDDEN_SIZE,
        output_size=1,
        learning_rate=BPNN_LEARNING_RATE,
        rng=rng,
    )
    dims = bpnn.parameter_vector().shape[0]
    pso = PSO(PSO_PARTICLES, dims, PSO_BOUNDS, rng)
    pso_history = pso.optimize(bpnn, x_weighted, y, PSO_ITERATIONS)
    bpnn_history = bpnn.train(x_weighted, y, BPNN_EPOCHS)
    pred = bpnn.forward(x_weighted).ravel()
    score = minmax_series(pd.Series(pred, index=x_norm.index))

    diagnostics = {
        "version": version,
        "input_size": x_weighted.shape[1],
        "hidden_size": BPNN_HIDDEN_SIZE,
        "output_size": 1,
        "learning_rate": BPNN_LEARNING_RATE,
        "epochs": BPNN_EPOCHS,
        "pso_particles": PSO_PARTICLES,
        "pso_iterations": PSO_ITERATIONS,
        "pso_bounds": str(PSO_BOUNDS),
        "pso_inertia": PSO_INERTIA,
        "pso_c1": PSO_C1,
        "pso_c2": PSO_C2,
        "random_seed": RANDOM_STATE + seed_offset,
        "pso_best_mse_initial_record": pso_history[0],
        "pso_best_mse_final": pso_history[-1],
        "bpnn_mse_initial_record": bpnn_history[0],
        "bpnn_mse_final": bpnn_history[-1],
    }
    history = pd.DataFrame(
        {
            "version": version,
            "stage": ["PSO"] * len(pso_history) + ["BPNN"] * len(bpnn_history),
            "record": list(range(1, len(pso_history) + 1))
            + list(range(1, len(bpnn_history) + 1)),
            "mse": pso_history + bpnn_history,
        }
    )
    return score.rename(version), diagnostics, history


def build_scores() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_and_clean_data()
    x_norm = normalized_indicator_matrix(df)
    schemes, weights, mc_weights = make_weight_schemes(x_norm)

    scores = df[[COL_CITY, COL_YEAR, COL_ORIGINAL_Y]].rename(
        columns={COL_ORIGINAL_Y: "Legacy_original_BPNN_PSO"}
    )
    diagnostics = []
    histories = []
    for offset, (version, w) in enumerate(schemes.items(), start=1):
        score, diag, history = run_bpnn_pso_version(x_norm, w, version, offset)
        scores[version] = score
        diagnostics.append(diag)
        histories.append(history)

    # Keep transparent additive scores in the same file for reference only.
    scores["A0_equal_dimension_additive"] = minmax_series(equal_dimension_score(x_norm))
    for name, w in schemes.items():
        additive_name = name.replace("_BPNN_PSO", "_additive")
        scores[additive_name] = minmax_series(weighted_score(x_norm, w))

    mc_interval = monte_carlo_additive_interval(x_norm, mc_weights, df)
    return (
        scores,
        weights,
        pd.DataFrame(diagnostics),
        pd.concat(histories, ignore_index=True),
        mc_interval,
    )


