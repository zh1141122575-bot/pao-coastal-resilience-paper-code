#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


from common import PAO_SCALE, load_model_base, output_dir, read_pao

OUTPUT_DIR = output_dir("10_spatial_autocorrelation")

COL_YEAR = "年份"
COL_CITY = "城市"
COL_Y = "Resilience_Score"
PAO_COL = "PAO_baseline"
LAGS = [1, 2, 3, 4]
K_NEIGHBORS = 5
N_PERMUTATIONS = 999
RANDOM_SEED = 20260523


CITY_COORDS = {
    "丹东市": (40.0005, 124.3547),
    "大连市": (38.9140, 121.6147),
    "盘锦市": (41.1199, 122.0707),
    "营口市": (40.6674, 122.2352),
    "锦州市": (41.0951, 121.1270),
    "葫芦岛市": (40.7110, 120.8369),
    "秦皇岛市": (39.9354, 119.6005),
    "唐山市": (39.6305, 118.1802),
    "沧州市": (38.3044, 116.8388),
    "天津市": (39.3434, 117.3616),
    "东营市": (37.4346, 118.6746),
    "滨州市": (37.3835, 117.9717),
    "潍坊市": (36.7069, 119.1618),
    "烟台市": (37.4638, 121.4479),
    "威海市": (37.5131, 122.1204),
    "青岛市": (36.0671, 120.3826),
    "日照市": (35.4164, 119.5269),
    "连云港市": (34.5967, 119.2216),
    "盐城市": (33.3474, 120.1636),
    "南通市": (31.9802, 120.8943),
    "上海市": (31.2304, 121.4737),
    "嘉兴市": (30.7461, 120.7555),
    "杭州市": (30.2741, 120.1551),
    "绍兴市": (30.0303, 120.5802),
    "宁波市": (29.8683, 121.5440),
    "舟山市": (29.9853, 122.2072),
    "台州市": (28.6564, 121.4208),
    "温州市": (27.9938, 120.6994),
    "宁德市": (26.6657, 119.5482),
    "福州市": (26.0745, 119.2965),
    "莆田市": (25.4541, 119.0076),
    "泉州市": (24.8741, 118.6759),
    "厦门市": (24.4798, 118.0894),
    "漳州市": (24.5130, 117.6471),
    "汕头市": (23.3541, 116.6819),
    "潮州市": (23.6567, 116.6226),
    "揭阳市": (23.5497, 116.3727),
    "汕尾市": (22.7862, 115.3753),
    "惠州市": (23.1115, 114.4168),
    "东莞市": (23.0207, 113.7518),
    "广州市": (23.1291, 113.2644),
    "深圳市": (22.5431, 114.0579),
    "中山市": (22.5176, 113.3928),
    "珠海市": (22.2707, 113.5767),
    "江门市": (22.5787, 113.0819),
    "阳江市": (21.8579, 111.9826),
    "茂名市": (21.6629, 110.9255),
    "湛江市": (21.2707, 110.3594),
    "北海市": (21.4813, 109.1201),
    "钦州市": (21.9797, 108.6544),
    "防城港市": (21.6869, 108.3547),
    "海口市": (20.0440, 110.1999),
    "三亚市": (18.2528, 109.5119),
}


# Coastal regions as defined in Table S1.
REGION_BY_PROVINCE = {
    "辽宁": "Bohai_Rim",
    "河北": "Bohai_Rim",
    "天津": "Bohai_Rim",
    "山东": "Bohai_Rim",
    "江苏": "Yangtze_River_Delta",
    "上海": "Yangtze_River_Delta",
    "浙江": "Yangtze_River_Delta",
    "福建": "Pearl_River_Delta_and_adjacent_coast",
    "广东": "Pearl_River_Delta_and_adjacent_coast",
    "广西": "Pearl_River_Delta_and_adjacent_coast",
    "海南": "Pearl_River_Delta_and_adjacent_coast",
}


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

    pao = read_pao("baseline_main_balanced", ["省份", COL_CITY, COL_YEAR, PAO_COL])

    df = y.merge(pao, on=[COL_CITY, COL_YEAR], how="inner")
    if df.shape[0] != y.shape[0]:
        raise ValueError(f"Merged rows {df.shape[0]} do not match resilience rows {y.shape[0]}.")

    missing_coords = sorted(set(df[COL_CITY]) - set(CITY_COORDS))
    if missing_coords:
        raise ValueError(f"Missing coordinates for cities: {missing_coords}")

    df["PAO_scaled"] = df[PAO_COL] / PAO_SCALE
    df["region"] = df["省份"].map(REGION_BY_PROVINCE)
    if df["region"].isna().any():
        missing = sorted(df.loc[df["region"].isna(), "省份"].unique())
        raise ValueError(f"Missing region mapping for provinces: {missing}")
    df["region_year"] = df["region"] + "_" + df[COL_YEAR].astype(str)
    return df.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_weights(cities: list[str], k_neighbors: int = K_NEIGHBORS) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(cities)
    distance = np.zeros((n, n), dtype=float)
    for i, city_i in enumerate(cities):
        lat_i, lon_i = CITY_COORDS[city_i]
        for j, city_j in enumerate(cities):
            if i == j:
                continue
            lat_j, lon_j = CITY_COORDS[city_j]
            distance[i, j] = haversine_km(lat_i, lon_i, lat_j, lon_j)

    weights = np.zeros((n, n), dtype=float)
    neighbor_rows = []
    for i, city in enumerate(cities):
        nearest = np.argsort(distance[i])
        nearest = [idx for idx in nearest if idx != i][:k_neighbors]
        raw = np.array([1.0 / distance[i, idx] for idx in nearest], dtype=float)
        raw = raw / raw.sum()
        for idx, weight in zip(nearest, raw):
            weights[i, idx] = weight
            neighbor_rows.append(
                {
                    "city": city,
                    "neighbor": cities[idx],
                    "distance_km": distance[i, idx],
                    "row_standardized_weight": weight,
                }
            )

    return (
        pd.DataFrame(weights, index=cities, columns=cities),
        pd.DataFrame(neighbor_rows),
    )


def moran_i(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    z = np.asarray(values, dtype=float)
    z = z - np.nanmean(z)
    denom = float(z @ z)
    s0 = float(w.sum())
    if denom == 0 or s0 == 0 or not np.isfinite(denom) or not np.isfinite(s0):
        return np.nan
    n = len(z)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        numerator = float(np.dot(z, np.dot(w, z)))
        value = float((n / s0) * numerator / denom)
    return value if np.isfinite(value) else np.nan


def moran_permutation(values: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> dict:
    observed = moran_i(values, weights)
    simulated = np.array(
        [moran_i(rng.permutation(values), weights) for _ in range(N_PERMUTATIONS)],
        dtype=float,
    )
    simulated = simulated[np.isfinite(simulated)]
    p_greater = float((np.sum(simulated >= observed) + 1) / (N_PERMUTATIONS + 1))
    p_two_sided = float((np.sum(np.abs(simulated) >= abs(observed)) + 1) / (N_PERMUTATIONS + 1))
    return {
        "Moran I": observed,
        "Expected I": -1.0 / (len(values) - 1),
        "Permutation p (greater)": p_greater,
        "Permutation p (two-sided)": p_two_sided,
    }


def add_almon_terms(df: pd.DataFrame, pao_col: str = "PAO_scaled") -> tuple[pd.DataFrame, list[str]]:
    d = df.sort_values([COL_CITY, COL_YEAR]).copy()
    lag_cols = []
    for lag in LAGS:
        col = f"PAO_L{lag}"
        d[col] = d.groupby(COL_CITY)[pao_col].shift(lag)
        lag_cols.append(col)
    d = d.dropna(subset=[COL_Y] + lag_cols).copy()

    lag_values = np.array(LAGS, dtype=float)
    centered = lag_values - lag_values.mean()
    z_cols = []
    for degree in range(3):
        z_col = f"Z{degree}"
        z_cols.append(z_col)
        d[z_col] = 0.0
        weights = centered**degree
        for pos, col in enumerate(lag_cols):
            d[z_col] += weights[pos] * d[col]
    return d, z_cols


def fit_baseline_pdl(df: pd.DataFrame) -> pd.DataFrame:
    d, z_cols = add_almon_terms(df)
    formula = f"{COL_Y} ~ {' + '.join(z_cols)} + C({COL_CITY}) + C({COL_YEAR})"
    result = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})
    d = d.copy()
    d["pdl_residual"] = result.resid
    return d


def p_value_from_z(value: float, se: float) -> float:
    if not np.isfinite(value) or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.0 * (1.0 - stats.norm.cdf(abs(value / se)))


def fit_region_year_pdl(df: pd.DataFrame) -> pd.DataFrame:
    d, z_cols = add_almon_terms(df)
    formula = f"{COL_Y} ~ {' + '.join(z_cols)} + C({COL_CITY}) + C(region_year)"
    # City dummies and region-by-year dummies are partly collinear (each region's level is
    # already absorbed by its cities). Redundant dummy columns are dropped before fitting so
    # that the city-clustered covariance matrix is well defined.
    y_mat, x_mat = patsy.dmatrices(formula, d, return_type="dataframe")
    keep = []
    for col in x_mat.columns:
        trial = x_mat[keep + [col]].to_numpy(float)
        if np.linalg.matrix_rank(trial) == len(keep) + 1:
            keep.append(col)
    if not all(z in keep for z in z_cols):
        raise ValueError("Almon terms are collinear with the fixed effects.")
    result = sm.OLS(y_mat, x_mat[keep]).fit(cov_type="cluster", cov_kwds={"groups": d[COL_CITY]})

    coef = result.params.reindex(z_cols).fillna(0.0)
    vcov = result.cov_params().loc[z_cols, z_cols]
    lag_values = np.array(LAGS, dtype=float)
    centered = lag_values - lag_values.mean()
    kbar = lag_values.mean()

    def lag_effect(lag: float) -> float:
        kh = lag - kbar
        return float(coef["Z0"] + coef["Z1"] * kh + coef["Z2"] * kh * kh)

    def lag_se(lag: float) -> float:
        kh = lag - kbar
        g = np.array([1.0, kh, kh * kh])
        var = float(g @ vcov.values @ g.T)
        return float(np.sqrt(max(var, 0.0)))

    effects = np.array([lag_effect(lag) for lag in LAGS])
    cumulative = float(effects.sum())
    cumulative_weights = np.array([float(np.sum(centered**degree)) for degree in range(3)])
    cumulative_se = float(np.sqrt(max(float(cumulative_weights @ vcov.values @ cumulative_weights.T), 0.0)))
    cumulative_p = p_value_from_z(cumulative, cumulative_se)

    grid = np.linspace(min(LAGS), max(LAGS), 301)
    grid_effects = np.array([lag_effect(lag) for lag in grid])
    direction = np.sign(cumulative) if cumulative != 0 else np.sign(grid_effects[np.argmax(np.abs(grid_effects))])
    peak_idx = int(np.argmax(grid_effects) if direction >= 0 else np.argmin(grid_effects))
    peak_lag = float(grid[peak_idx])
    peak_effect = float(grid_effects[peak_idx])
    peak_p = p_value_from_z(peak_effect, lag_se(peak_lag))
    joint_p = float(result.wald_test("Z0 = 0, Z1 = 0, Z2 = 0", scalar=True).pvalue)

    return pd.DataFrame(
        [
            {
                "Model": "L1-L4 Almon PDL with region-year FE",
                "N obs": int(result.nobs),
                "City FE": "Yes",
                "Year FE": "Replaced by coastal-region-by-year FE",
                "Cluster SE": "City",
                "Adj. R2": float(result.rsquared_adj),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "Peak lag": peak_lag,
                "Peak effect": peak_effect,
                "Peak p": peak_p,
                "Cumulative effect": cumulative,
                "Cumulative SE": cumulative_se,
                "Cumulative p": cumulative_p,
                "Joint p": joint_p,
                "Conclusion": (
                    "Main lag pattern remains positive after absorbing region-year shocks"
                    if cumulative > 0 and cumulative_p < 0.1
                    else "Lag direction remains positive but statistical strength weakens"
                    if cumulative > 0
                    else "Lag direction differs after absorbing region-year shocks"
                ),
            }
        ]
    )


def annual_moran_table(df: pd.DataFrame, weights: pd.DataFrame, variable: str, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + sum(ord(ch) for ch in variable))
    rows = []
    cities = list(weights.index)
    w = weights.values
    for year, group in df.groupby(COL_YEAR):
        group = group.set_index(COL_CITY).reindex(cities)
        if group[variable].isna().any():
            continue
        result = moran_permutation(group[variable].to_numpy(dtype=float), w, rng)
        rows.append(
            {
                "Variable": label,
                "Year": int(year),
                **result,
                "Significance": stars(result["Permutation p (greater)"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_moran(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, group in annual.groupby("Variable"):
        rows.append(
            {
                "Variable": variable,
                "Years tested": int(group["Year"].nunique()),
                "Mean Moran I": float(group["Moran I"].mean()),
                "Median Moran I": float(group["Moran I"].median()),
                "Min Moran I": float(group["Moran I"].min()),
                "Max Moran I": float(group["Moran I"].max()),
                "Share positive": float((group["Moran I"] > 0).mean()),
                "Share p<0.05": float((group["Permutation p (greater)"] < 0.05).mean()),
                "Conclusion": (
                    "Persistent positive spatial autocorrelation"
                    if (group["Moran I"] > 0).mean() >= 0.8
                    and (group["Permutation p (greater)"] < 0.05).mean() >= 0.5
                    else "Limited residual spatial autocorrelation"
                    if "residual" in variable.lower()
                    and (group["Permutation p (greater)"] < 0.05).mean() < 0.5
                    else "Mixed spatial autocorrelation evidence"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    cities = sorted(df[COL_CITY].unique())
    weights, neighbors = build_weights(cities)
    residual_df = fit_baseline_pdl(df)
    region_year_summary = fit_region_year_pdl(df)

    annual_tables = [
        annual_moran_table(df, weights, "PAO_scaled", "PAO"),
        annual_moran_table(df, weights, COL_Y, "Resilience score"),
        annual_moran_table(residual_df, weights, "pdl_residual", "Baseline PDL residuals"),
    ]
    annual_moran = pd.concat(annual_tables, ignore_index=True)
    moran_summary = summarize_moran(annual_moran)

    city_coordinates = pd.DataFrame(
        [
            {"城市": city, "latitude": CITY_COORDS[city][0], "longitude": CITY_COORDS[city][1]}
            for city in cities
        ]
    )

    output_xlsx = OUTPUT_DIR / "spatial_autocorrelation_diagnostics.xlsx"
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        moran_summary.to_excel(writer, sheet_name="Moran_summary", index=False)
        annual_moran.to_excel(writer, sheet_name="Annual_Moran_I", index=False)
        region_year_summary.to_excel(writer, sheet_name="Region_year_FE_PDL", index=False)
        neighbors.to_excel(writer, sheet_name="Spatial_neighbors", index=False)
        city_coordinates.to_excel(writer, sheet_name="City_coordinates", index=False)
        weights.to_excel(writer, sheet_name="Weight_matrix")


    note = f"""# Spatial autocorrelation diagnostics

Spatial weights: row-standardized inverse-distance weights among each city's {K_NEIGHBORS} nearest coastal-city neighbors.

Moran's I p-values are based on {N_PERMUTATIONS} random permutations. The tests are diagnostic evidence rather than a full spatial econometric identification strategy.

The region-year FE robustness model absorbs broad coastal-region shocks by replacing common year fixed effects with coastal-region-by-year fixed effects.
"""
    (OUTPUT_DIR / "spatial_autocorrelation_diagnostics_note.md").write_text(note, encoding="utf-8")

    print(f"Saved spatial diagnostics to: {OUTPUT_DIR}")
    print("\nMoran summary:")
    print(moran_summary.to_string(index=False))
    print("\nRegion-year FE PDL:")
    print(region_year_summary.to_string(index=False))


if __name__ == "__main__":
    main()
