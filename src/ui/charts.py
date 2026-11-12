"""Plotly chart system — one template so every chart reads as part of the same app.

Rules (dataviz method): thin marks, recessive grid, ≤ one y-axis, hover on every
mark, legends for ≥ 2 series, survived = blue / did not survive = orange.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from src.ui.theme import AQUA, BRASS, CATEGORICAL, GOLD, INK, INK_2, MUTED, PERISHED, SURVIVED, VIOLET

GRID = "rgba(255,255,255,0.06)"
AXIS = "rgba(255,255,255,0.14)"
FONT = "Inter, 'Source Sans', system-ui, sans-serif"
OUTCOME_COLORS = {"Survived": SURVIVED, "Did not survive": PERISHED}
SEQ_BLUE = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4"]
DIVERGING = [[0.0, PERISHED], [0.5, "#2a3550"], [1.0, SURVIVED]]

pio.templates["titanic"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, color=INK_2, size=13),
    title=dict(font=dict(family="'Space Grotesk', Inter, sans-serif", color=INK, size=16), x=0.0, xanchor="left", y=0.97),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=CATEGORICAL,
    margin=dict(l=10, r=14, t=52, b=52),
    xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2), automargin=True),
    yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, tickfont=dict(color=MUTED), title_font=dict(color=INK_2), automargin=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
    hoverlabel=dict(bgcolor="#0d1a33", bordercolor="rgba(227,184,102,0.5)", font=dict(family=FONT, color=INK, size=13)),
    bargap=0.28, bargroupgap=0.08,
    coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))),
))
pio.templates.default = "titanic"
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "toImageButtonOptions": {"format": "png", "scale": 2}}


def show(fig: go.Figure, key: str | None = None, **kwargs):
    """Render with our template (theme=None stops Streamlit overriding colours)."""
    import streamlit as st

    return st.plotly_chart(fig, theme=None, width="stretch", config=CONFIG, key=key, **kwargs)


def _finish(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(template="titanic", height=height)
    if title:
        fig.update_layout(title=title)
    return fig


def survival_rate_bar(df: pd.DataFrame, col: str, title: str, label: str | None = None, overall: float | None = None) -> go.Figure:
    """Survival rate per group (one series → no legend), passengers count in hover."""
    fig = go.Figure(go.Bar(
        x=df[col].astype(str), y=df["survival_rate"], marker=dict(color=SURVIVED, line=dict(width=0), cornerradius=5),
        text=[f"{v:.0%}" for v in df["survival_rate"]], textposition="outside", textfont=dict(color=INK, size=12),
        customdata=np.stack([df["passengers"], df["survivors"]], axis=1),
        hovertemplate=f"<b>%{{x}}</b><br>Survival rate %{{y:.1%}}<br>%{{customdata[1]}} of %{{customdata[0]}} survived<extra></extra>",
    ))
    if overall is not None:
        fig.add_hline(y=overall, line=dict(color=BRASS, dash="dot", width=1.5),
                      annotation_text=f"overall {overall:.0%}", annotation_font_color=BRASS, annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", range=[0, max(1.0, float(df["survival_rate"].max()) * 1.15)], title=None)
    fig.update_xaxes(title=label or None)
    return _finish(fig, title)


def outcome_stack(df: pd.DataFrame, col: str, title: str, label: str | None = None) -> go.Figure:
    """Counts of survivors vs non-survivors per group (two series → legend)."""
    fig = go.Figure()
    for name, ycol, color in (("Did not survive", "non_survivors", PERISHED), ("Survived", "survivors", SURVIVED)):
        fig.add_bar(name=name, x=df[col].astype(str), y=df[ycol], marker=dict(color=color, line=dict(color="#0b1630", width=2), cornerradius=4),
                    hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="Passengers")
    fig.update_xaxes(title=label or None)
    return _finish(fig, title)


def donut(values: list[float], labels: list[str], colors: list[str], title: str | None = None, center: str | None = None, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Pie(values=values, labels=labels, hole=0.68, sort=False, direction="clockwise",
                           marker=dict(colors=colors, line=dict(color="#050b1a", width=3)),
                           textinfo="percent", textfont=dict(color="#ffffff", size=13),
                           hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>"))
    if center:
        fig.add_annotation(text=center, showarrow=False, font=dict(size=20, color=INK, family="'Space Grotesk', Inter"))
    fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", yanchor="top"))
    return _finish(fig, title, height)


def histogram_by_outcome(df: pd.DataFrame, col: str, title: str, nbins: int = 30, log_x: bool = False) -> go.Figure:
    fig = go.Figure()
    for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
        sub = df[df["Survived"] == val][col].dropna()
        fig.add_histogram(x=sub, name=name, nbinsx=nbins, marker=dict(color=color, line=dict(color="#050b1a", width=1)), opacity=0.85,
                          hovertemplate=f"{name}<br>{col}: %{{x}}<br>Passengers: %{{y}}<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=col, type="log" if log_x else "linear")
    fig.update_yaxes(title="Passengers")
    return _finish(fig, title)


def histogram(series: pd.Series, title: str, color: str = SURVIVED, nbins: int = 30, xlabel: str | None = None) -> go.Figure:
    fig = go.Figure(go.Histogram(x=series.dropna(), nbinsx=nbins, marker=dict(color=color, line=dict(color="#050b1a", width=1)),
                                 hovertemplate="%{x}<br>Count: %{y}<extra></extra>"))
    fig.update_xaxes(title=xlabel or series.name)
    fig.update_yaxes(title="Count")
    return _finish(fig, title)


def heatmap(corr: pd.DataFrame, title: str, height: int = 470) -> go.Figure:
    z = corr.values
    fig = go.Figure(go.Heatmap(z=z, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=np.round(z, 2), texttemplate="%{text}", textfont=dict(size=11, color="#ffffff"),
                               xgap=2, ygap=2, hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                               colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=MUTED))))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _finish(fig, title, height)


def hbar(labels, values, title: str, color: str = SURVIVED, fmt: str = ".3f", error=None, xlabel: str | None = None, height: int | None = None) -> go.Figure:
    labels, values = list(labels)[::-1], list(values)[::-1]
    err = None if error is None else dict(type="data", array=list(error)[::-1], color=MUTED, thickness=1.2, width=3)
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", error_x=err, marker=dict(color=color, cornerradius=4),
                           hovertemplate=f"<b>%{{y}}</b><br>%{{x:{fmt}}}<extra></extra>"))
    fig.update_xaxes(title=xlabel)
    return _finish(fig, title, height or max(260, 34 * len(labels) + 80))


def signed_hbar(labels, values, title: str, xlabel: str = "", height: int | None = None, fmt: str = "+.1f", scale: float = 100) -> go.Figure:
    labels, values = list(labels)[::-1], list(values)[::-1]
    colors = [SURVIVED if v > 0 else PERISHED for v in values]
    fig = go.Figure(go.Bar(y=labels, x=[v * scale for v in values], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v * scale:{fmt}}" for v in values], textposition="outside", textfont=dict(color=INK, size=11),
                           hovertemplate=f"<b>%{{y}}</b><br>%{{x:{fmt}}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    xs = [v * scale for v in values] or [0]
    lo, hi = min(min(xs), 0), max(max(xs), 0)
    pad = 0.28 * max(hi - lo, 1e-9)
    fig.update_xaxes(title=xlabel, range=[lo - (pad if lo < 0 else 0.02 * pad), hi + (pad if hi > 0 else 0.02 * pad)])
    return _finish(fig, title, height or max(280, 32 * len(labels) + 90))


def waterfall(base: float, contrib: pd.DataFrame, final: float, title: str, top: int = 10) -> go.Figure:
    c = contrib.head(top)
    rest = contrib["contribution"].iloc[top:].sum() if len(contrib) > top else 0.0
    labels = ["Average passenger"] + [f"{r.label} = {r.value}" for r in c.itertuples()] + (["Other features"] if abs(rest) > 1e-6 else []) + ["This passenger"]
    vals = [base] + list(c["contribution"]) + ([rest] if abs(rest) > 1e-6 else []) + [final]
    measure = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels[::-1], x=[v * 100 for v in vals][::-1], measure=measure[::-1],
        increasing=dict(marker=dict(color=SURVIVED)), decreasing=dict(marker=dict(color=PERISHED)), totals=dict(marker=dict(color=BRASS)),
        connector=dict(line=dict(color=AXIS, width=1, dash="dot")),
        text=[f"{v * 100:+.1f}" if m == "relative" else f"{v * 100:.0f}%" for v, m in zip(vals, measure)][::-1], textposition="outside",
        textfont=dict(color=INK, size=11),
        hovertemplate="%{y}<br>%{x:.1f}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted survival probability (%)", range=[0, 105])
    return _finish(fig, title, max(340, 34 * len(labels) + 70))


def roc_curves(curves: dict[str, dict], title: str = "ROC curve", highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1), name="Random guess", hoverinfo="skip")
    hl = [highlight] if isinstance(highlight, str) else (list(highlight) if highlight else list(curves)[:3])
    ordered = [n for n in curves if n not in hl] + [n for n in curves if n in hl]  # muted first, highlighted on top
    for name in ordered:
        c = curves[name]
        hi = name in hl
        color = CATEGORICAL[hl.index(name) % 3] if hi else "rgba(180,194,217,0.22)"
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=f"{name} (AUC {c['auc']:.3f})",
                        line=dict(color=color, width=2.6 if hi else 1.2),
                        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra>" + name + "</extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    fig.update_layout(legend=dict(orientation="v", x=0.98, y=0.04, xanchor="right", yanchor="bottom", bgcolor="rgba(5,11,26,0.6)"))
    return _finish(fig, title, 420)


def pr_curve(prec, rec, ap: float, base_rate: float, title: str = "Precision–recall curve") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=rec, y=prec, mode="lines", name=f"Model (AP {ap:.3f})", line=dict(color=SURVIVED, width=2.5), fill="tozeroy",
                    fillcolor="rgba(58,142,230,0.10)", hovertemplate="Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>")
    fig.add_hline(y=base_rate, line=dict(color=MUTED, dash="dot", width=1), annotation_text=f"base rate {base_rate:.0%}", annotation_font_color=MUTED)
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _finish(fig, title, 420)


def calibration_plot(mean_pred, frac_pos, title: str = "Calibration (reliability) curve") -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1), name="Perfectly calibrated", hoverinfo="skip")
    fig.add_scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Model", line=dict(color=BRASS, width=2.5),
                    marker=dict(size=9, color=BRASS, line=dict(color="#050b1a", width=2)),
                    hovertemplate="Predicted %{x:.0%} → actually survived %{y:.0%}<extra></extra>")
    fig.update_xaxes(title="Mean predicted probability", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Observed survival rate", range=[0, 1.02], tickformat=".0%")
    return _finish(fig, title, 420)


def confusion(cm, title: str = "Confusion matrix", normalize: bool = False) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    labels = ["Did not survive", "Survived"]
    z = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"<b>{(z[i, j] * 100 if normalize else z[i, j]):.0f}{'%' if normalize else ''}</b><br><span style='font-size:11px'>{names[i][j]}</span>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=[f"Pred: {l}" for l in labels], y=[f"Actual: {l}" for l in labels], text=text, texttemplate="%{text}",
                               textfont=dict(color="#ffffff", size=15), colorscale=[[0, "#0d1a33"], [1, "#2a78d6"]], showscale=False, xgap=4, ygap=4,
                               hovertemplate="%{y}<br>%{x}<br>%{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    return _finish(fig, title, 360)


def metric_bars(df: pd.DataFrame, metrics: list[str], labels: dict, title: str) -> go.Figure:
    """Grouped bars: models × metrics (≤ 5 series, categorical order fixed)."""
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_bar(name=labels[m], x=df["Model"], y=df[m], marker=dict(color=CATEGORICAL[i % len(CATEGORICAL)], cornerradius=3),
                    hovertemplate=f"<b>%{{x}}</b><br>{labels[m]}: %{{y:.3f}}<extra></extra>")
    lo = float(np.nanmin(df[metrics].to_numpy())) if len(df) else 0
    fig.update_yaxes(range=[max(0, lo - 0.08), 1.0], title="Score")
    fig.update_layout(barmode="group")
    return _finish(fig, title, 420)


def cv_box(fold_scores: dict[str, list[float]], title: str, metric_label: str = "ROC-AUC") -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(fold_scores.items()):
        fig.add_box(y=vals, name=name, boxpoints="all", jitter=0.35, pointpos=0, marker=dict(color=SURVIVED, size=8, line=dict(color="#050b1a", width=1.5)),
                    line=dict(color="rgba(180,194,217,0.6)", width=1.2), fillcolor="rgba(58,142,230,0.12)",
                    hovertemplate=f"<b>{name}</b><br>{metric_label}: %{{y:.3f}}<extra></extra>")
    fig.update_yaxes(title=f"{metric_label} per fold")
    fig.update_layout(showlegend=False)
    return _finish(fig, title, 400)


def line_ice(df: pd.DataFrame, x: str, title: str, current: float | None = None, xlabel: str | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df["prob_survived"], mode="lines", line=dict(color=SURVIVED, width=2.5, shape="spline"),
                               fill="tozeroy", fillcolor="rgba(58,142,230,0.08)",
                               hovertemplate=f"{xlabel or x}: %{{x}}<br>Survival estimate %{{y:.0%}}<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dot", width=1))
    if current is not None:
        fig.add_vline(x=current, line=dict(color=BRASS, width=1.5, dash="dash"), annotation_text="this passenger", annotation_font_color=BRASS)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Survival estimate")
    fig.update_xaxes(title=xlabel or x)
    fig.update_layout(showlegend=False)
    return _finish(fig, title, 300)


def gauge(prob: float, title: str = "") -> go.Figure:
    color = SURVIVED if prob >= 0.5 else PERISHED
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=40, color=INK, family="'Space Grotesk'")),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)), bar=dict(color=color, thickness=0.3),
                                            bgcolor="rgba(255,255,255,0.04)", borderwidth=0,
                                            steps=[dict(range=[0, 50], color="rgba(232,105,58,0.10)"), dict(range=[50, 100], color="rgba(58,142,230,0.10)")],
                                            threshold=dict(line=dict(color=BRASS, width=2), thickness=0.8, value=50))))
    return _finish(fig, title, 250)


def scatter3d_galaxy(coords: np.ndarray, survived: np.ndarray, hover: list[str], title: str, highlight=None, neighbors=None) -> go.Figure:
    fig = go.Figure()
    for name, val, color in (("Did not survive", 0, PERISHED), ("Survived", 1, SURVIVED)):
        m = survived == val
        fig.add_scatter3d(x=coords[m, 0], y=coords[m, 1], z=coords[m, 2], mode="markers", name=name,
                          marker=dict(size=3.4, color=color, opacity=0.8, line=dict(width=0)), text=np.asarray(hover)[m],
                          hovertemplate="%{text}<extra>" + name + "</extra>")
    if highlight is not None:
        fig.add_scatter3d(x=[highlight[0]], y=[highlight[1]], z=[highlight[2]], mode="markers+text", name="Your passenger",
                          marker=dict(size=11, color=BRASS, symbol="diamond", line=dict(color="#ffffff", width=2)),
                          text=["  Your passenger"], textfont=dict(color=BRASS, size=13), hovertemplate="Your passenger<extra></extra>")
        if neighbors is not None:
            for p in neighbors:
                fig.add_scatter3d(x=[highlight[0], p[0]], y=[highlight[1], p[1]], z=[highlight[2], p[2]], mode="lines",
                                  line=dict(color="rgba(227,184,102,0.6)", width=3), showlegend=False, hoverinfo="skip")
    axis = dict(showbackground=False, gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)", showticklabels=False, title="")
    fig.update_layout(scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.5, y=1.4, z=0.8))),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"), margin=dict(l=0, r=0, t=40, b=0))
    return _finish(fig, title, 560)


def surface3d(ages, fares, Z, title: str) -> go.Figure:
    fig = go.Figure(go.Surface(x=ages, y=fares, z=Z, colorscale=DIVERGING, cmin=0, cmax=1, opacity=0.95,
                               contours=dict(z=dict(show=True, usecolormap=True, highlightcolor=BRASS, project=dict(z=True))),
                               colorbar=dict(title="P(survive)", thickness=10, tickformat=".0%", tickfont=dict(color=MUTED)),
                               hovertemplate="Age %{x:.0f}<br>Fare £%{y:.0f}<br>Survival estimate %{z:.0%}<extra></extra>"))
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.1)", color=MUTED)
    fig.update_layout(scene=dict(xaxis=dict(title="Age", **axis), yaxis=dict(title="Fare (£)", **axis), zaxis=dict(title="P(survive)", range=[0, 1], tickformat=".0%", **axis),
                                 bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)), aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=0.7)),
                      margin=dict(l=0, r=0, t=40, b=0))
    return _finish(fig, title, 560)


def learning_curve_plot(lc: pd.DataFrame, title: str = "Learning curve") -> go.Figure:
    fig = go.Figure()
    for name, col, color in (("Training score", "train", BRASS), ("Validation score", "val", SURVIVED)):
        m, s = lc[f"{col}_mean"], lc[f"{col}_std"]
        fig.add_scatter(x=lc["train_size"], y=m + s, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=lc["train_size"], y=m - s, mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(227,184,102,0.12)" if col == "train" else "rgba(58,142,230,0.14)", showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=lc["train_size"], y=m, mode="lines+markers", name=name, line=dict(color=color, width=2.5),
                        marker=dict(size=8, line=dict(color="#050b1a", width=1.5)), hovertemplate=f"{name}<br>n=%{{x}}: %{{y:.3f}}<extra></extra>")
    fig.update_xaxes(title="Training passengers")
    fig.update_yaxes(title="ROC-AUC")
    return _finish(fig, title, 380)


__all__ = [n for n in dir() if not n.startswith("_")] + ["CONFIG", "OUTCOME_COLORS", "AQUA", "VIOLET", "GOLD"]
