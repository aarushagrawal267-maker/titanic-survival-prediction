"""Reusable HTML components (rendered with st.markdown/st.html)."""
from __future__ import annotations

import html
import math

import streamlit as st

from src.ui.theme import PERISHED, SURVIVED


def _md(markup: str) -> None:
    # Collapse indentation/blank lines so Markdown never turns HTML into a code block.
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


def icon(name: str, cls: str = "") -> str:
    return f'<span class="msr {cls}">{name}</span>'


def page_header(eyebrow: str, title: str, subtitle: str = "", icon_name: str = "sailing") -> None:
    _md(f"""
    <div class="tml-page-head">
      <div class="tml-eyebrow"><span class="dot"></span>{esc(eyebrow)}</div>
      <h1>{esc(title)}</h1>
      {f'<div class="sub">{subtitle}</div>' if subtitle else ''}
    </div>""")


def section(title: str, num: str | None = None, icon_name: str | None = None) -> None:
    _md(f"""<div class="tml-section">{f'<span class="num">{esc(num)}</span>' if num else ''}
        {icon(icon_name) if icon_name else ''}<h3>{esc(title)}</h3><span class="line"></span></div>""")


def kpi_tiles(items: list[dict]) -> None:
    """items: {label, value, sub?, icon?, accent?, count_to? (int → animated count-up)}"""
    cards = []
    for i, it in enumerate(items):
        accent = it.get("accent", "var(--brass)")
        if it.get("count_to") is not None:
            prefix, suffix = it.get("prefix", ""), it.get("suffix", "")
            val = f'{esc(prefix)}<span class="tml-count" style="--to:{int(it["count_to"])}"></span>{esc(suffix)}'
        else:
            val = esc(it["value"])
        long_cls = " small" if it.get("count_to") is None and len(str(it.get("value", ""))) >= 7 else ""
        cards.append(f"""
        <div class="tml-kpi" style="--accent:{accent}; animation-delay:{i * 70}ms">
          <div class="label">{icon(it.get('icon', 'insights'))}{esc(it['label'])}</div>
          <div class="value{long_cls}">{val}</div>
          {f'<div class="sub">{it["sub"]}</div>' if it.get('sub') else ''}
        </div>""")
    _md(f'<div class="tml-grid">{"".join(cards)}</div>')


def insight_cards(insights: list[dict]) -> None:
    cards = "".join(
        f"""<div class="tml-insight" style="animation-delay:{i * 60}ms"><div class="ic">{icon(it.get('icon', 'lightbulb'))}</div>
        <div><h5>{esc(it['title'])}</h5><p>{esc(it['text'])}</p></div></div>"""
        for i, it in enumerate(insights)
    )
    _md(f'<div class="tml-insights">{cards}</div>')


def card(title: str, body: str, icon_name: str | None = None) -> None:
    _md(f"""<div class="tml-card"><h4>{icon(icon_name) + ' ' if icon_name else ''}{esc(title)}</h4><p>{body}</p></div>""")


def note(text: str, icon_name: str = "info") -> None:
    _md(f'<div class="tml-note">{icon(icon_name)}<div>{text}</div></div>')


def disclaimer() -> None:
    note("<b>Statistical estimate, not a fact.</b> Predictions come from patterns in historical passenger data "
         "(891 labelled records). They describe how similar passengers fared on average and say nothing certain about any individual.",
         "balance")


def causation_note() -> None:
    note("<b>Correlation ≠ causation.</b> These patterns are statistical associations in the historical data. "
         "Many factors (class, cabin location, sex, age, crew decisions) are intertwined, so no single feature should be read as a cause.",
         "hub")


def learn(title: str, body: str) -> None:
    """Educational callout — shown only when Learn mode is on (sidebar toggle)."""
    if st.session_state.get("learn_mode", True):
        _md(f'<div class="tml-learn">{icon("school")}<div><b>{esc(title)}</b><p>{body}</p></div></div>')


def pill(text: str, kind: str = "") -> str:
    return f'<span class="tml-pill {kind}">{esc(text)}</span>'


def flow(steps: list[tuple[str, str, str]]) -> None:
    """Pipeline diagram: list of (title, description, icon)."""
    parts = "".join(
        f'<div class="step">{icon(ic)}<div class="n">STEP {i + 1:02d}</div><div class="t">{esc(t)}</div><div class="d">{d}</div></div>'
        for i, (t, d, ic) in enumerate(steps)
    )
    _md(f'<div class="tml-flow">{parts}</div>')


def prob_color(p: float) -> str:
    return SURVIVED if p >= 0.5 else PERISHED


def ring(p: float, label: str = "survival", size: int = 150) -> str:
    r = size / 2 - 10
    circ = 2 * math.pi * r
    off = circ * (1 - max(0.0, min(1.0, p)))
    color = prob_color(p)
    return f"""
    <div class="tml-ring" style="--ring:{color}; --circ:{circ:.1f}; width:{size}px; height:{size}px">
      <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle class="track" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}" fill="none" stroke-width="11"/>
        <circle class="bar" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}" fill="none" stroke-width="11"
          stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}"/>
      </svg>
      <div class="center"><div><div class="pct">{p * 100:.0f}%</div><div class="lbl">{esc(label)}</div></div></div>
    </div>"""


def boarding_pass(details: dict, prob: float, confidence: str, factors: list[dict], model_name: str,
                  empty_note: str | None = None) -> None:
    """The signature prediction card. Hover/tap flips it to show the top factors."""
    color = prob_color(prob)
    outcome = "Likely survived" if prob >= 0.5 else "Likely did not survive"
    fields = "".join(f'<div class="f"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in details["fields"].items())
    maxabs = max([abs(f["contribution"]) for f in factors] + [0.01])
    rows = []
    for f in factors:
        w = abs(f["contribution"]) / maxabs * 50
        c = SURVIVED if f["contribution"] > 0 else PERISHED
        left = 50 if f["contribution"] > 0 else 50 - w
        rows.append(
            f'<div class="tml-factor"><div>{esc(f["label"])} <span style="color:var(--muted)">= {esc(f["value"])}</span></div>'
            f'<div class="bar"><i style="left:{left:.1f}%;width:{w:.1f}%;background:{c}"></i>'
            f'<i style="left:50%;width:1px;background:rgba(255,255,255,.3)"></i></div>'
            f'<div class="val">{f["contribution"] * 100:+.1f} pts</div></div>'
        )
    if rows:
        back = "".join(rows) + (
            '<p style="margin-top:.7rem;font-size:.75rem;color:var(--muted)">Contributions in percentage points of predicted survival '
            f'probability (SHAP). <span style="color:{SURVIVED}">Blue</span> pushes the estimate up, <span style="color:{PERISHED}">orange</span> '
            'pushes it down. Statistical associations, not causes.</p>')
    else:
        back = f"<p>{empty_note or 'No explanation available.'}</p>"
    _md(f"""
    <div class="tml-pass-wrap">
      <div class="tml-pass" tabindex="0">
        <div class="face front">
          <div class="main">
            <div class="top"><span class="brand">TITANIC ML LAB</span><span class="kind">SURVIVAL ESTIMATE · {esc(model_name.upper())}</span></div>
            <div class="stamp" style="--stamp:{color}">Est. survival {prob * 100:.0f}%</div>
            <div class="who">{esc(details.get('title', 'Passenger'))}</div>
            <div class="fields">{fields}</div>
            <div class="route"><span>{esc(details.get('from', 'SOUTHAMPTON'))}</span><span class="lineb"></span>{icon('directions_boat')}<span class="lineb"></span><span>NEW YORK</span></div>
            <div class="hint">{icon('touch_app')} hover or tap to flip</div>
          </div>
          <div class="stub">
            {ring(prob)}
            <div class="outcome" style="--ring:{color}">{outcome}</div>
            <div class="conf">Confidence: {esc(confidence)}</div>
          </div>
        </div>
        <div class="face back">
          <h4>{icon('psychology')} Why the model estimated {prob * 100:.0f}%</h4>
          {back}
        </div>
      </div>
    </div>""")


def sidebar_brand() -> None:
    st.sidebar.markdown("""
    <div class="tml-brand">
      <div class="logo"><svg width="26" height="26" viewBox="0 0 64 64" fill="none">
        <path d="M6 40 L58 40 L52 50 Q32 54 12 50 Z" fill="#e3b866"/>
        <rect x="16" y="32" width="32" height="8" rx="1" fill="#eaf0fa"/>
        <rect x="20" y="20" width="5" height="12" rx="1" fill="#c9a96e"/><rect x="20" y="20" width="5" height="3" fill="#111"/>
        <rect x="28" y="20" width="5" height="12" rx="1" fill="#c9a96e"/><rect x="28" y="20" width="5" height="3" fill="#111"/>
        <rect x="36" y="20" width="5" height="12" rx="1" fill="#c9a96e"/><rect x="36" y="20" width="5" height="3" fill="#111"/>
        <rect x="44" y="22" width="4" height="10" rx="1" fill="#c9a96e"/><rect x="44" y="22" width="4" height="3" fill="#111"/>
        <path d="M4 56 Q14 52 24 56 T44 56 T62 56" stroke="#3a8ee6" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      </svg></div>
      <div class="name">TITANIC ML LAB<span>SURVIVAL PREDICTION</span></div>
    </div>""", unsafe_allow_html=True)
