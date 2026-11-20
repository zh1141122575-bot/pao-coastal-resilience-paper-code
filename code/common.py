#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent
BAIDU_FILE = DATA_DIR / "Baidu index.xlsx"
RESILIENCE_FILE = DATA_DIR / "Coastal urban resilience.xlsx"
OUTPUT_ROOT = Path(os.environ.get("PAO_OUTPUT_DIR", CODE_DIR / "outputs"))

COL_PROVINCE = "省份"
COL_CITY = "城市"
COL_YEAR = "年份"
COL_Y = "Resilience_Score"

# PAO (Eq. 2 scale) is divided by this constant before entering the regression models, so
# PDL coefficients are expressed per 100,000 PAO units.
PAO_SCALE = 100_000.0
LAG_MAX = 4
POLY_DEGREE = 2

PROVINCE_CN = {
    "Tianjin": "天津", "Hebei": "河北", "Liaoning": "辽宁", "Shandong": "山东",
    "Shanghai": "上海", "Jiangsu": "江苏", "Zhejiang": "浙江", "Fujian": "福建",
    "Guangdong": "广东", "Guangxi": "广西", "Hainan": "海南",
}

# Resilience indicators (Table S2): symbol -> column name used in the original dataset.
INDICATORS = {
    "MRIA": "海洋相关经济活动强度（万元）",
    "ISD": "第三产业增加值占GDP比重(%)",
    "MV": "社会消费品零售总额(万元)",
    "EL": "教育支出(万元)",
    "MR": "医院、卫生院数(个)",
    "SSC": "失业保险参保人数(人)",
    "TI": "轨道交通/公共交通支出_万元",
    "CI": "移动电话年末用户数(万户)",
    "ICDR": "本年城市市政公用设施建设固定资产投资完成额_万元",
    "EB": "污水处理厂集中处理率(%)",
    "WRS": "城市全部供水-供水总量_万立方米",
    "GCR": "绿化覆盖面积_公顷",
    "LGGA": "词频总和",
    "GPG": "绿色专利授权总量（个）",
}

# ---------------------------------------------------------------- PAO keywords
BASELINE_DOMAINS = {
    "Economic": ["蓝色经济", "海洋经济", "渔业", "渔业资源", "水产养殖", "港口", "航运",
                 "海上风电", "潮汐能", "滨海旅游", "海岛旅游"],
    "Ecological": ["海洋污染", "赤潮", "海洋环境保护", "海洋环境保护法", "红树林", "珊瑚礁",
                   "海洋生物", "海洋生物多样性", "风暴潮", "海平面上升"],
    "Social": ["海洋知识", "海洋馆", "渔民", "海滨社区", "台风预警", "领海", "海警", "海洋权益"],
}
POLITICAL_TERMS = ["领海", "海警", "海洋权益"]
MARKET_TERMS = ["港口吞吐量", "海运价格", "航运指数", "集装箱运价", "港口物流"]
TOURISM_CONSUMPTION_TERMS = ["休闲渔业", "邮轮旅游", "海洋公园", "沙滩旅游", "海景房"]
EVENT_SENSITIVE_TERMS = ["台风", "厄尔尼诺", "死鱼", "白色污染"]

# PAO versions used in the robustness analyses (Supplementary Note 4).
PAO_VERSIONS = [
    {"construction": "Baseline PAO", "sheet": "baseline_main_balanced", "pao_col": "PAO_baseline",
     "dimension_cols": ["PAO_baseline_Economic", "PAO_baseline_Ecological", "PAO_baseline_Social"]},
    {"construction": "Raw-sum PAO", "sheet": "R1_raw_sum", "pao_col": "PAO_raw", "dimension_cols": []},
    {"construction": "Excluding national-political terms", "sheet": "R2_no_political",
     "pao_col": "PAO_R2_no_political",
     "dimension_cols": ["PAO_R2_no_political_Economic", "PAO_R2_no_political_Ecological",
                        "PAO_R2_no_political_Social"]},
    {"construction": "Adding market-sensitive terms", "sheet": "R3_market_sensitive",
     "pao_col": "PAO_R3_market_sensitive",
     "dimension_cols": ["PAO_R3_market_sensitive_Economic", "PAO_R3_market_sensitive_Ecological",
                        "PAO_R3_market_sensitive_Social"]},
    {"construction": "Adding tourism/event-sensitive terms", "sheet": "R4_tourism_event",
     "pao_col": "PAO_R4_tourism_event",
     "dimension_cols": ["PAO_R4_tourism_event_Economic", "PAO_R4_tourism_event_Ecological",
                        "PAO_R4_tourism_event_Social"]},
]


def output_dir(name: str) -> Path:
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _city_key(city_cn: pd.Series) -> pd.Series:
    return city_cn.astype(str).str.strip().map(lambda s: s if s.endswith("市") else f"{s}市")


# ---------------------------------------------------------------- raw data
@lru_cache(maxsize=1)
def _load_baidu_cached() -> pd.DataFrame:
    raw = pd.read_excel(BAIDU_FILE)
    out = pd.DataFrame({
        COL_PROVINCE: raw["Province"].map(PROVINCE_CN),
        COL_CITY: _city_key(raw["City_CN"]),
        COL_YEAR: raw["Year"].astype(int),
    })
    for col in raw.columns:
        match = re.search(r"\(([^()]*)\)\s*$", str(col))
        if match:
            out[match.group(1)] = pd.to_numeric(raw[col], errors="coerce")
    return out


def load_baidu() -> pd.DataFrame:
    """Keyword-level annual Baidu Index; columns: 省份, 城市, 年份, <44 Chinese keywords>."""
    return _load_baidu_cached().copy()


@lru_cache(maxsize=1)
def _load_resilience_cached() -> pd.DataFrame:
    raw = pd.read_excel(RESILIENCE_FILE)
    out = pd.DataFrame({
        COL_PROVINCE: raw["Province"].map(PROVINCE_CN),
        COL_CITY: _city_key(raw["City_CN"]),
        COL_YEAR: raw["Year"].astype(int),
    })
    for symbol, cn_name in INDICATORS.items():
        col = next(c for c in raw.columns if str(c).startswith(f"{symbol}:"))
        out[cn_name] = pd.to_numeric(raw[col], errors="coerce")
    out[COL_Y] = pd.to_numeric(raw["Resilience_Score"], errors="coerce")
    return out.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)


def load_resilience(columns: list[str] | None = None) -> pd.DataFrame:
    """Resilience indicators (original column names) and the baseline Resilience_Score."""
    df = _load_resilience_cached().copy()
    return df if columns is None else df[columns].copy()


# ---------------------------------------------------------------- PAO construction
def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _round16(values: pd.Series) -> pd.Series:
    """Round to 16 significant digits, the precision at which the PAO values used in the
    reported analyses were stored. This only affects the last binary digit, but it makes
    tree-based models (random forest, XGBoost) reproduce the reported results exactly."""
    return values.map(lambda v: float(f"{v:.16g}"))


def domain_balanced_index(df: pd.DataFrame, domains: dict[str, list[str]], pao_col: str):
    """Domain-balanced PAO (Eqs. 1-2).

    PAO_d = (1 / |K_d|) * sum of keyword BI in domain d          (Eq. 1)
    PAO   = mean of the three domain scores                      (Eq. 2)
    """
    keywords = _unique([kw for kws in domains.values() for kw in kws])
    result = df[[COL_PROVINCE, COL_CITY, COL_YEAR] + keywords].copy()
    counts = {d: len(k) for d, k in domains.items()}
    k_bar = sum(counts.values()) / len(counts)
    params = []
    unrounded = {}
    for domain, kws in domains.items():
        scale = 1.0 / counts[domain]
        result[f"{pao_col}_{domain}_raw_sum"] = result[kws].sum(axis=1, skipna=False)
        unrounded[domain] = result[f"{pao_col}_{domain}_raw_sum"] * scale
        result[f"{pao_col}_{domain}"] = _round16(unrounded[domain])
        params.append({"version": pao_col, "domain": domain, "keyword_count_K_d": counts[domain],
                       "average_keyword_count_K_bar": k_bar, "domain_weight_1_over_K_d": scale,
                       "keywords": "、".join(kws)})
    result[pao_col] = _round16(sum(unrounded[d] for d in domains) / len(domains))
    result[f"{pao_col}_raw_sum_unbalanced"] = result[[f"{pao_col}_{d}_raw_sum" for d in domains]].sum(
        axis=1, skipna=False)
    return result.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True), pd.DataFrame(params)


@lru_cache(maxsize=1)
def _pao_versions_cached():
    df = load_baidu()
    if df.drop(columns=[COL_PROVINCE, COL_CITY, COL_YEAR]).isna().any().any():
        raise ValueError("Baidu Index data contain missing values.")
    sheets, params = {}, []

    baseline, p = domain_balanced_index(df, BASELINE_DOMAINS, "PAO_baseline")
    sheets["baseline_main_balanced"], _ = baseline, params.append(p)

    base_kw = _unique([kw for kws in BASELINE_DOMAINS.values() for kw in kws])
    r1 = df[[COL_PROVINCE, COL_CITY, COL_YEAR] + base_kw].copy()
    r1["PAO_raw"] = r1[base_kw].sum(axis=1, skipna=False)
    sheets["R1_raw_sum"] = r1.sort_values([COL_CITY, COL_YEAR]).reset_index(drop=True)

    variants = {
        "R2_no_political": ("PAO_R2_no_political", {
            "Economic": BASELINE_DOMAINS["Economic"], "Ecological": BASELINE_DOMAINS["Ecological"],
            "Social": [k for k in BASELINE_DOMAINS["Social"] if k not in POLITICAL_TERMS]}),
        "R3_market_sensitive": ("PAO_R3_market_sensitive", {
            "Economic": _unique(BASELINE_DOMAINS["Economic"] + MARKET_TERMS),
            "Ecological": BASELINE_DOMAINS["Ecological"], "Social": BASELINE_DOMAINS["Social"]}),
        "R4_tourism_event": ("PAO_R4_tourism_event", {
            "Economic": _unique(BASELINE_DOMAINS["Economic"] + TOURISM_CONSUMPTION_TERMS),
            "Ecological": _unique(BASELINE_DOMAINS["Ecological"] + EVENT_SENSITIVE_TERMS),
            "Social": BASELINE_DOMAINS["Social"]}),
    }
    for sheet, (col, domains) in variants.items():
        frame, p = domain_balanced_index(df, domains, col)
        sheets[sheet] = frame
        params.append(p)
    return sheets, pd.concat(params, ignore_index=True)


def pao_versions() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sheets, params = _pao_versions_cached()
    return {k: v.copy() for k, v in sheets.items()}, params.copy()


def read_pao(sheet: str = "baseline_main_balanced", columns: list[str] | None = None) -> pd.DataFrame:
    df = pao_versions()[0][sheet]
    return df if columns is None else df[columns].copy()


def load_model_base() -> pd.DataFrame:
    """城市, 年份, Resilience_Score (baseline)."""
    return load_resilience([COL_YEAR, COL_CITY, COL_Y])


# ---------------------------------------------------------------- statistics helpers
def p_value_from_z(value: float, se: float) -> float:
    if not np.isfinite(value) or not np.isfinite(se) or se <= 0:
        return np.nan
    return 2.0 * (1.0 - stats.norm.cdf(abs(value / se)))


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
