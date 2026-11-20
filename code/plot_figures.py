#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.interpolate import make_interp_spline
from statsmodels.gam.api import BSplines, GLMGam

from common import CODE_DIR, COL_CITY, COL_YEAR, COL_Y, OUTPUT_ROOT, load_model_base, read_pao

FIG_DIR = Path(os.environ.get("PAO_FIGURE_DIR", CODE_DIR.parent / "Figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Standard map of China with city polygons and boundary lines (GeoJSON or shapefile; not redistributed).
BASEMAP_FILE = Path(os.environ.get("PAO_BASEMAP", CODE_DIR / "basemap" / "中国_市.geojson"))
ALBERS = "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

DIMENSIONS = ["Economic", "Ecological", "Social"]
MAIN_KEYWORDS = 29                 # keywords in the main PAO index (Table 1)
K_BAR = MAIN_KEYWORDS / 3          # average number of keywords per domain

TYPE_ORDER = ["High PAO-High resilience", "High PAO-Low resilience",
              "Low PAO-High resilience", "Low PAO-Low resilience"]
TYPE_SHORT = dict(zip(TYPE_ORDER, ["H-H", "H-L", "L-H", "L-L"]))
SCATTER_COLORS = dict(zip(TYPE_ORDER, ["#f76c6c", "#e9b665", "#79cbe0", "#9ee6a3"]))   # Fig. 2a-c
BAR_COLORS = dict(zip(TYPE_ORDER, ["#ef6a67", "#e8b45f", "#76c7d8", "#98df9a"]))       # Fig. 2d
SHAP_CMAP = LinearSegmentedColormap.from_list("pao", ["#4c78a8", "#f58518", "#e45756"])
GG_LW = 72.27 / 25.4 * 0.75        # ggplot2 linewidth (mm-based) -> points
GG_PT = 72.27 / 25.4               # ggplot2 text/point size (mm) -> points

# ---------------------------------------------------------------- font: Arial everywhere
if os.environ.get("PAO_FONT_DIR"):
    for font_file in Path(os.environ["PAO_FONT_DIR"]).glob("*.tt[fc]"):
        font_manager.fontManager.addfont(str(font_file))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


def out(name: str) -> dict:
    return pd.read_excel(OUTPUT_ROOT / name, sheet_name=None)


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("saved", name)


def smooth_xy(x, y, n: int = 300):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xs = np.linspace(x.min(), x.max(), n)
    return xs, make_interp_spline(x, y, k=min(3, len(x) - 1))(xs)


def padded_ylim(values, pad: float) -> tuple[float, float]:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    spread = (hi - lo) * pad if hi > lo else 1.0
    return lo - spread, hi + spread


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")


def label_formatter(factor: float, decimals: int) -> FuncFormatter:
    """Tick label = tick position x factor (used to label the Fig. 1 PAO axes in Eq. 1-2 units)."""
    return FuncFormatter(lambda v, _pos: f"{v * factor:.{decimals}f}")


def gg_formatter(ticks) -> FuncFormatter:
    """ggplot2-style labels: all tick labels share the number of decimals of the most precise one."""
    decimals = max(len(f"{t:.6f}".rstrip("0").split(".")[1]) for t in ticks)
    return FuncFormatter(lambda v, _pos: f"{v:.{decimals}f}")


# ---------------------------------------------------------------- Fig. 1
# The PAO axes of Fig. 1a and 1c keep the tick positions of the original figure, which was drawn
# on the domain-summed scale in units of 100,000 (= Eq. 2 x 29 for the total index and Eq. 1 x 29/3
# for the dimensions); tick labels show the Eq. 1-2 values in thousands
# (e.g. 1.3, 1.4, ... -> 4.5, 4.8, ...; 0.2, 0.4, ... -> 2.1, 4.1, ...).
TOTAL_TO_PLOT = MAIN_KEYWORDS / 100_000      # Eq. 2 value -> plotted value
DIM_TO_PLOT = K_BAR / 100_000                # Eq. 1 value -> plotted value
TOTAL_LABEL = 100 / MAIN_KEYWORDS            # plotted value -> Eq. 2 value in thousands
DIM_LABEL = 100 / K_BAR                      # plotted value -> Eq. 1 value in thousands


def annual_data() -> pd.DataFrame:
    pao = read_pao("baseline_main_balanced", [COL_CITY, COL_YEAR, "PAO_baseline", "PAO_baseline_Economic",
                                              "PAO_baseline_Ecological", "PAO_baseline_Social"])
    df = load_model_base().merge(pao, on=[COL_CITY, COL_YEAR])
    annual = df.groupby(COL_YEAR, as_index=False).mean(numeric_only=True).sort_values(COL_YEAR)
    annual["PAO_plot"] = annual["PAO_baseline"] * TOTAL_TO_PLOT
    years = annual[COL_YEAR].astype(int).tolist()
    lookup = dict(zip(years, annual["PAO_plot"]))
    annual["PAO_4yr_plot"] = [sum(lookup.get(y - k, 0.0) for k in range(1, 5)) for y in years]
    for dim in DIMENSIONS:
        annual[f"{dim}_plot"] = annual[f"PAO_baseline_{dim}"] * DIM_TO_PLOT
    return annual


def fig1a(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    x = annual[COL_YEAR].to_numpy()
    y_res = annual[COL_Y].to_numpy()
    bars = ax.bar(x, y_res, width=0.68, color="#94c6cd", edgecolor="white", linewidth=0.8,
                  label="Resilience score", zorder=2)
    ax.set_ylabel("Resilience score", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    lo, hi = float(np.nanmin(y_res)), float(np.nanmax(y_res))
    pad = (hi - lo) * 0.12 if hi > lo else 0.05
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 2))

    ax2 = ax.twinx()
    line_pao, = ax2.plot(*smooth_xy(x, annual["PAO_plot"]), color="#ff9aa2", linewidth=2.2, label="PAO", zorder=4)
    ax2.set_ylabel("PAO (1,000)", fontsize=11)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.set_ylim(*padded_ylim(annual["PAO_plot"], 0.16))
    ax2.yaxis.set_major_formatter(label_formatter(TOTAL_LABEL, 1))

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 48))
    line_4yr, = ax3.plot(*smooth_xy(x, annual["PAO_4yr_plot"]), color="#b9b4f0", linewidth=2.2,
                         label="4-yr PAO sum", zorder=3)
    ax3.set_ylabel("4-yr PAO sum (1,000)", fontsize=11)
    ax3.tick_params(axis="y", labelsize=10)
    ax3.set_ylim(*padded_ylim(annual["PAO_4yr_plot"], 0.12))
    ax3.yaxis.set_major_formatter(label_formatter(TOTAL_LABEL, 0))

    ax.legend([bars, line_pao, line_4yr], ["Resilience score", "PAO", "4-yr PAO sum"],
              frameon=False, loc="upper left", fontsize=10)
    style_axes(ax)
    save(fig, "Fig1a_annual_resilience_PAO")


def fig1b() -> None:
    res = out("05_PDL_lag_selection/PDL_lag_selection_results.xlsx")
    marginal, cumulative = res["Fig1b_marginal_curve"], res["Fig1b_cumulative_curve"]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    peak = marginal.loc[marginal["effect"].idxmax()]
    ax.fill_between(marginal["lag"], marginal["ci_low"], marginal["ci_high"], color="#1f77b4", alpha=0.18,
                    linewidth=0, label="95% CI")
    ax.plot(marginal["lag"], marginal["effect"], color="#1f77b4", linewidth=1.8, label="Marginal association w(h)")
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvline(peak["lag"], color="#e45756", linewidth=0.9, linestyle=":")
    ax.set_title("PDL Marginal Associations (L=4)", fontsize=11)
    ax.set_ylabel("Marginal association with resilience", fontsize=9)
    ax.text(peak["lag"] + 0.04, peak["effect"], f"Peak {peak['lag']:.2f}y", color="#e45756", fontsize=8,
            ha="left", va="center")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(1, 4)
    ax.tick_params(axis="both", labelsize=8)
    style_axes(ax)
    save(fig, "Fig1b_PDL_marginal")

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    peak = cumulative.loc[cumulative["effect"].idxmax()]
    ax.fill_between(cumulative["lag"], cumulative["ci_low"], cumulative["ci_high"], color="#1f77b4", alpha=0.18,
                    linewidth=0, label="95% CI")
    ax.plot(cumulative["lag"], cumulative["effect"], color="#1f77b4", linewidth=1.8, label="Cumulative association")
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.axvline(peak["lag"], color="#e45756", linewidth=0.9, linestyle=":")
    ax.set_title("PDL Cumulative Association (L=4)", fontsize=11)
    ax.set_xlabel("Lag (years)", fontsize=9)
    ax.set_ylabel("Cumulative association with resilience", fontsize=9)
    ax.text(peak["lag"] - 0.04, peak["effect"], f"Peak {peak['lag']:.2f}y", color="#e45756", fontsize=8,
            ha="right", va="center")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(1, 4)
    ax.tick_params(axis="both", labelsize=8)
    style_axes(ax)
    save(fig, "Fig1b_PDL_cumulative")


def fig1c(annual: pd.DataFrame) -> None:
    colors = {"Economic": "#cb9475", "Ecological": "#b8dbb3", "Social": "#4a5f7e"}
    fig, ax = plt.subplots(figsize=(3.0, 5.5))
    years = annual[COL_YEAR].to_numpy()
    for dim in DIMENSIONS:
        ax.plot(*smooth_xy(years, annual[f"{dim}_plot"]), label=f"PAO_{dim.lower()}", color=colors[dim],
                linewidth=2.0)
    ax.set_xticks([int(years.min()), int(years.max())])
    ax.set_xticklabels([str(int(years.min())), str(int(years.max()))], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel("PAO (1,000)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(label_formatter(DIM_LABEL, 1))
    style_axes(ax)
    save(fig, "Fig1c_PAO_dimensions")


# ---------------------------------------------------------------- Fig. 2 (ggplot2 style of the original R code)
def gg_frame(ax: plt.Axes, border: float, tick_width: float) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(border * GG_LW)
    ax.tick_params(axis="both", color="black", width=tick_width * GG_LW, length=3.5, labelcolor="#4d4d4d")
    ax.set_axisbelow(True)


def draw_scatter(ax: plt.Axes, mismatch: pd.DataFrame, year: int, pao_thr: float, res_thr: float) -> None:
    d = mismatch[mismatch[COL_YEAR] == year]
    x = d["Lag_adjusted_PAO_L1_L4_mean"].to_numpy(float) / 1000
    x_max = max(950 / MAIN_KEYWORDS, float(x.max()) * 1.08)
    ax.set_facecolor("white")
    ax.scatter(x, d[COL_Y], c=d["Mismatch_type"].map(SCATTER_COLORS), s=(1.35 * GG_PT) ** 2 * 0.75,
               alpha=0.92, edgecolors="none", zorder=3)
    ax.axvline(pao_thr / 1000, color="black", linewidth=0.70 * GG_LW, zorder=2)
    ax.axhline(res_thr, color="black", linewidth=0.70 * GG_LW, zorder=2)
    label_box = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none")
    for xf, yv, lab, ha in [(0.01, 0.92, "(L-H)", "left"), (0.60, 0.92, "(H-H)", "center"),
                            (0.01, 0.05, "(L-L)", "left"), (0.60, 0.05, "(H-L)", "center")]:
        ax.text(xf * x_max, yv, lab, fontsize=3.4 * GG_PT, ha=ha, va="center", clip_on=True, zorder=4,
                bbox=label_box if ha == "left" else None)
    ax.set_xlim(0 - 0.01 * x_max, x_max * 1.02)
    ax.set_ylim(-0.01, 1.03)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(gg_formatter([t for t in ax.get_xticks() if 0 <= t <= x_max]))
    ax.yaxis.set_major_formatter(gg_formatter([0, 0.25, 0.5, 0.75, 1.0]))
    ax.set_xlabel("Lag-adjusted PAO (1,000)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Resilience score", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(color="#dbdbdb", linewidth=0.35 * GG_LW)
    gg_frame(ax, border=0.85, tick_width=0.55)


def load_basemap():
    """City polygons and boundary lines (incl. the South China Sea lines) of the standard map of China."""
    import geopandas as gpd
    from pyproj import Transformer

    raw = gpd.read_file(BASEMAP_FILE)
    name_col = "name" if "name" in raw.columns else "地名"
    raw = raw.rename(columns={name_col: "city"})
    raw["city"] = raw["city"].astype(str).str.strip()
    poly = raw[raw.geom_type.isin(["Polygon", "MultiPolygon"])].to_crs(ALBERS)
    lines = raw[raw.geom_type.isin(["LineString", "MultiLineString"])].to_crs(ALBERS)
    to_albers = Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)
    xs, ys = to_albers.transform([73, 175, 175, 73], [-5, -5, 54, 54])
    return poly, lines, (min(xs), max(xs), min(ys), max(ys))


def draw_north_arrow(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> None:
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_axis_off()
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Polygon([[0, 1], [-0.42, 0.0], [0, 0.22]], closed=True, facecolor="black",
                             edgecolor="black", linewidth=0.45 * GG_LW))
    ax.add_patch(plt.Polygon([[0, 1], [0.42, 0.0], [0, 0.22]], closed=True, facecolor="white",
                             edgecolor="black", linewidth=0.45 * GG_LW))
    ax.text(0, 0.36, "N", ha="center", va="center", fontsize=9,
            path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(
                linewidth=2, foreground="white")])


def draw_scale_bar(ax: plt.Axes, bbox, fig_w_cm: float, fig_h_cm: float) -> None:
    """ggspatial-style bar (4 alternating segments) at the bottom-left, 1 cm / 0.38 cm from the corner."""
    x0, x1, y0, y1 = bbox
    length = 1_000_000.0                                   # 1000 km in projected metres
    seg = length / 4
    bx = x0 + (x1 - x0) * 1.0 / fig_w_cm
    by = y0 + (y1 - y0) * 0.38 / fig_h_cm
    bh = (y1 - y0) * 0.22 / fig_h_cm
    for i in range(4):
        ax.add_patch(plt.Rectangle((bx + i * seg, by), seg, bh, facecolor="black" if i % 2 == 0 else "white",
                                   edgecolor="black", linewidth=0.45 * GG_LW, zorder=6))
    ax.text(bx + length + (x1 - x0) * 0.012, by + bh / 2, "1000 km", ha="left", va="center", fontsize=0.7 * 11,
            zorder=6)


def fig2_panel(tag: str, year: int, mismatch: pd.DataFrame, pao_thr: float, res_thr: float, basemap) -> None:
    poly, lines, bbox = basemap
    fig_w, fig_h = 6.6, 4.2                                # one panel of the 13.2 x 8.4 in 2 x 2 layout
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    d = mismatch[mismatch[COL_YEAR] == year][[COL_CITY, "Mismatch_type"]].rename(columns={COL_CITY: "city"})
    data = poly.merge(d, on="city", how="left")
    data[data["Mismatch_type"].isna()].plot(ax=ax, color="#f5f5f5", edgecolor="#bfbfbf", linewidth=0.11 * GG_LW)
    for t in TYPE_ORDER:
        sub = data[data["Mismatch_type"] == t]
        if not sub.empty:
            sub.plot(ax=ax, color=SCATTER_COLORS[t], edgecolor="#bfbfbf", linewidth=0.11 * GG_LW)
    if not lines.empty:
        lines.plot(ax=ax, color="#595959", linewidth=0.28 * GG_LW)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect("equal", adjustable="box", anchor="C")     # coord_sf(expand = FALSE)
    ax.set_axis_off()
    ax.apply_aspect()
    px, py, pw, ph = ax.get_position().bounds                # map panel inside the figure

    counts = d["Mismatch_type"].value_counts()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SCATTER_COLORS[t], edgecolor="none") for t in TYPE_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#f5f5f5", edgecolor="none"))
    labels = [f"{TYPE_SHORT[t]} ({100 * counts.get(t, 0) / len(d):.1f}%)" for t in TYPE_ORDER] + ["No data"]
    leg = fig.legend(handles, labels, title="Spatial matching types", loc="lower left",
                     bbox_to_anchor=(px + 0.055 * pw, py + 0.1 * ph), frameon=True, fontsize=7.5, title_fontsize=9,
                     handlelength=1.55, handleheight=1.0, borderpad=0.3, labelspacing=0.25)
    leg.get_title().set_fontweight("bold")
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.78)
    leg.get_frame().set_edgecolor("none")

    cm = 1 / 2.54
    draw_scale_bar(ax, (*ax.get_xlim(), *ax.get_ylim()), pw * fig_w / cm, ph * fig_h / cm)
    arrow_w, arrow_h = 0.58 * cm / fig_w, 0.78 * cm / fig_h
    arrow_left = px + pw - 11.2 * cm / fig_w - arrow_w
    arrow_bottom = py + ph - 2.0 * cm / fig_h - arrow_h
    draw_north_arrow(fig, arrow_left, arrow_bottom, arrow_w, arrow_h)
    fig.text(arrow_left + arrow_w / 2, arrow_bottom - 0.02, str(year), color="red", fontsize=14,
             fontweight="bold", ha="center", va="top")

    # inset_element(left = 0.535, bottom = 0.01, right = 0.915, top = 0.7), axis titles included
    l, bt, r, tp = px + 0.535 * pw, py + 0.01 * ph, px + 0.915 * pw, py + 0.7 * ph
    inset = fig.add_axes([l + 0.085, bt + 0.135, (r - l) - 0.085, (tp - bt) - 0.135])
    draw_scatter(inset, mismatch, year, pao_thr, res_thr)
    save(fig, f"Fig2{tag}_{year}_map_scatter")


def fig2_scatter(mismatch: pd.DataFrame, year: int, pao_thr: float, res_thr: float) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    draw_scatter(ax, mismatch, year, pao_thr, res_thr)
    save(fig, f"Fig2_{year}_scatter")


def fig2() -> None:
    res = out("13_mismatch/lag_adjusted_PAO_resilience_mismatch_results.xlsx")
    mismatch = res["city_year_classification"]
    thr = res["thresholds"].iloc[0]
    basemap = load_basemap() if BASEMAP_FILE.exists() else None
    for tag, year in (("a", 2018), ("b", 2021), ("c", 2024)):
        if basemap is not None:
            fig2_panel(tag, year, mismatch, thr["PAO_threshold_median"], thr["resilience_threshold_median"], basemap)
        else:
            print(f"Basemap not found ({BASEMAP_FILE}); drawing the Fig. 2 scatter plots only.")
            fig2_scatter(mismatch, year, thr["PAO_threshold_median"], thr["resilience_threshold_median"])

    counts = mismatch.groupby([COL_YEAR, "Mismatch_type"]).size().unstack(fill_value=0).reindex(
        columns=TYPE_ORDER, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(6.7, 3.45))
    bottom = np.zeros(len(share))
    positions = np.arange(len(share))
    for t in TYPE_ORDER:                                   # H-H at the bottom, L-L on top
        vals = share[t].to_numpy()
        ax.bar(positions, vals, bottom=bottom, width=0.64, color=BAR_COLORS[t], edgecolor="white",
               linewidth=0.38 * GG_LW, label=TYPE_SHORT[t], zorder=2)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(positions[i], bottom[i] + v / 2, f"{v:.1f}", ha="center", va="center",
                        fontsize=3.2 * GG_PT, fontweight="bold", color="#1a1a1a", zorder=3)
        bottom += vals
    ax.set_xticks(positions)
    ax.set_xticklabels(share.index.astype(int).astype(str))
    ax.set_xlim(-0.6, len(share) - 0.4)
    ax.set_ylim(0, 101)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.8,
              handlelength=1.2, handleheight=0.6, columnspacing=1.6, borderaxespad=0.2)
    ax.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cities (%)", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9, labelcolor="black")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.28 * GG_LW)
    gg_frame(ax, border=1.0, tick_width=0.38)
    ax.tick_params(labelcolor="black")
    save(fig, "Fig2d_annual_mismatch_composition")


# ---------------------------------------------------------------- Fig. 3
def fig3() -> None:
    import shap

    res = out("11_XGBoost_SHAP/xgboost_strict_validation_results.xlsx")
    meta = [COL_CITY, COL_YEAR, COL_Y]
    shap_df, x_df = res["Fig3_SHAP_values"], res["Fig3_feature_values"]
    features = [c for c in shap_df.columns if c not in meta]
    labels = [f"{f.split('_')[1]} PAO {f.split('_')[2]}" for f in features]

    plt.figure(figsize=(7.2, 5.2))
    shap.summary_plot(shap_df[features].to_numpy(), x_df[features].set_axis(labels, axis=1), feature_names=labels,
                      cmap=SHAP_CMAP, show=False, max_display=10)
    plt.xlabel("SHAP Value", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    save(plt.gcf(), "Fig3a_SHAP_summary")

    share = res["shap_by_dimension"].set_index("dimension")["share"].reindex(["Ecological", "Economic", "Social"])
    vals = share.to_numpy()
    plt.figure(figsize=(4, 4))
    bars = plt.bar(["ECO", "ECON", "SOC"], vals, color="lightblue", edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom", fontsize=18)
    plt.ylabel("Share of mean |SHAP|", fontsize=20)
    plt.title("Grouped Contributions (L1-L4)", fontsize=20)
    plt.yticks([0, 0.2, 0.4, 0.6], fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylim(0, max(0.65, vals.max() + 0.12))
    plt.tight_layout()
    save(plt.gcf(), "Fig3a_grouped_SHAP_share")

    dep = res["Fig3b_dependence"]
    for dim in DIMENSIONS:
        x = dep[f"PAO_{dim}_L1_L4_mean"].to_numpy(float) / 1000
        y = dep[f"SHAP_{dim}_L1_L4_sum"].to_numpy(float)
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        grid = np.linspace(float(xs.min()), float(xs.max()), 200)
        n_splines = min(8, max(5, len(np.unique(xs)) // 8))
        gam = GLMGam(ys, exog=np.ones((len(xs), 1)), smoother=BSplines(xs[:, None], df=[n_splines], degree=[3])).fit()
        pred = gam.get_prediction(exog=np.ones((len(grid), 1)), exog_smooth=grid[:, None]).summary_frame(alpha=0.05)

        fig, ax = plt.subplots(figsize=(3.07, 3.25))
        sc = ax.scatter(x, y, c=x, cmap=SHAP_CMAP, s=13, alpha=0.88, edgecolors="none", rasterized=True)
        ax.plot(grid, pred["mean"], color="#222222", linewidth=1.8)
        ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"], color="#222222", alpha=0.16, linewidth=0)
        ax.axhline(0, color="#8a8a8a", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel(f"PAO_{dim.lower()} (1,000)", fontsize=14, fontweight="bold")
        ax.set_ylabel("SHAP value" if dim == "Economic" else "", fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=12, width=1.0, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#555555")
        ax.grid(False)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label(f"PAO_{dim.lower()} (1,000)", fontsize=14, fontweight="bold")
        cbar.ax.tick_params(labelsize=12, width=0.8, length=2)
        fig.tight_layout()
        save(fig, f"Fig3b_SHAP_dependence_{dim.lower()}")


# ---------------------------------------------------------------- supplementary figures
def fig_s3() -> None:
    yearly = out("03_resilience_MC_alpha/baseline_resilience_MC_alpha_BPNN_PSO_results.xlsx")["FigS3_yearly"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = yearly[COL_YEAR].to_numpy(float)
    ax.fill_between(x, yearly["MC_alpha_p025_BPNN_PSO"], yearly["MC_alpha_p975_BPNN_PSO"], color="#9E9E9E",
                    alpha=0.25, linewidth=0, label="Monte Carlo 95% interval")
    ax.plot(yearly[COL_YEAR], yearly["MC_alpha_mean_BPNN_PSO"], color="#666666", linestyle="--", linewidth=1.4,
            label="Monte Carlo mean")
    ax.plot(yearly[COL_YEAR], yearly["Baseline_BPNN_PSO"], color="#222222", marker="o", markersize=4,
            linewidth=1.8, label="Baseline BPNN-PSO")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("National mean resilience score", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xticks(yearly[COL_YEAR].tolist())
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save(fig, "FigS3_MC_alpha_resilience")


def _boxplot(ax: plt.Axes, data: pd.DataFrame, title: str, x_max: int) -> None:
    positions, means = [], []
    for k in range(1, x_max + 1):
        vals = data.loc[data["n_selected_indicators"].eq(k), "national_mean_score_0_100"]
        if vals.empty:
            continue
        positions.append(k)
        q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
        lower = max(vals.min(), q1 - 1.5 * (q3 - q1))
        upper = min(vals.max(), q3 + 1.5 * (q3 - q1))
        means.append(vals.mean())
        ax.add_patch(plt.Rectangle((k - 0.28, q1), 0.56, q3 - q1, facecolor="white", edgecolor="#4D4D4D",
                                   linewidth=1.2, zorder=2))
        ax.plot([k - 0.28, k + 0.28], [med, med], color="#222222", linewidth=1.4, zorder=3)
        ax.plot([k, k], [q3, upper], color="#E53935", linewidth=1.2, zorder=2)
        ax.plot([k, k], [q1, lower], color="#43A047", linewidth=1.2, zorder=2)
        ax.plot([k - 0.12, k + 0.12], [upper, upper], color="#E53935", linewidth=1.2, zorder=2)
        ax.plot([k - 0.12, k + 0.12], [lower, lower], color="#43A047", linewidth=1.2, zorder=2)
    if positions:
        ax.plot(positions, means, color="#F2A000", marker="o", markersize=2.8, linewidth=1.3)
    ax.set_xlim(0.4, x_max + 0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of selected indicators", fontsize=11, fontweight="bold")
    ax.set_ylabel("Resilience score", fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9, width=1.0, direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_s4() -> None:
    subsets = out("04_resilience_indicator_subsets/resilience_indicator_LGGA_sensitivity_results.xlsx")[
        "subset_scores_2014_2024"]
    panels = [
        ("a", "a  All indicator subsets", subsets, 14),
        ("b", "b  Subsets including LGGA",
         subsets[subsets["includes_LGGA"] & subsets["n_selected_indicators"].ge(2)], 14),
        ("c", "c  Subsets excluding LGGA", subsets[~subsets["includes_LGGA"]], 13),
    ]
    for tag, title, data, x_max in panels:
        fig, ax = plt.subplots(figsize=(3.83, 3.8))
        _boxplot(ax, data, title, x_max)
        save(fig, f"FigS4{tag}_indicator_subsets")


def fig_s5() -> None:
    res = out("05_PDL_lag_selection/PDL_lag_selection_results.xlsx")
    unrestricted = res["Unrestricted_L1_L5_effects"].rename(columns={"coefficient": "estimate", "std_error": "se"})
    almon = res["Almon_lag_effects"]
    almon = almon[almon["Lag window"] == "L1-L5"].rename(columns={"marginal_effect": "estimate", "std_error": "se"})
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10, "legend.fontsize": 9}):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.axhline(0, color="#6b7280", linewidth=0.9, linestyle="--", zorder=1)
        ax.fill_between(almon["lag"].to_numpy(float), almon["estimate"] - 1.96 * almon["se"],
                        almon["estimate"] + 1.96 * almon["se"], color="#1f77b4", alpha=0.16, linewidth=0,
                        label="Centered Almon PDL 95% CI", zorder=2)
        ax.plot(almon["lag"], almon["estimate"], color="#1f77b4", linewidth=2.4, marker="o", markersize=5,
                label="Centered Almon PDL", zorder=4)
        ax.errorbar(unrestricted["lag"], unrestricted["estimate"], yerr=1.96 * unrestricted["se"], color="#d95f02",
                    linewidth=1.6, marker="s", markersize=4.8, capsize=3, linestyle="--", label="Unrestricted DL",
                    zorder=5)
        ax.set_xlabel("Lag year")
        ax.set_ylabel("Marginal association between PAO and resilience")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.75, 5.25)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        save(fig, "FigS5_unrestricted_vs_Almon_PDL")


def main() -> None:
    annual = annual_data()
    fig1a(annual)
    fig1b()
    fig1c(annual)
    fig2()
    fig3()
    fig_s3()
    fig_s4()
    fig_s5()
    print(f"All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
