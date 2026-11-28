"""Global "Midnight Atlantic" theme: quiet ink chrome with brass as the single accent. Injected once per page run."""
from __future__ import annotations

import streamlit as st

# Design tokens (data colours validated for colour-blind separation on the navy surface)
INK = "#e8ebf0"
INK_2 = "#aab2bf"
MUTED = "#7c8594"
SURFACE = "#131823"
BG = "#0a0d13"
BRASS = "#e3b866"
SURVIVED = "#3a8ee6"
PERISHED = "#e8693a"
AQUA = "#1fa883"
VIOLET = "#8a7ae6"
GOLD = "#c08a1c"
MAGENTA = "#d55181"
CATEGORICAL = [SURVIVED, PERISHED, AQUA, VIOLET, GOLD, MAGENTA]
CLASS_COLORS = {1: GOLD, 2: VIOLET, 3: AQUA}

_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap');

/* Midnight Atlantic, quiet chrome: one ink ramp, hairlines, brass as the single fitting */
:root {
  --bg: #0a0d13; --bg-2: #0e121a; --surface: #131823; --surface-2: #1a202c;
  --line: rgba(232,236,242,.08); --line-2: rgba(232,236,242,.14);
  --ink: #e8ebf0; --ink-2: #aab2bf; --muted: #7c8594;
  --brass: #e3b866; --brass-2: #f0cd8c; --brass-ink: #1a1408;
  --survived: #3a8ee6; --perished: #e8693a; --aqua: #1fa883; --violet: #8a7ae6; --gold: #c08a1c;
  --radius: 14px; --radius-sm: 10px; --radius-xs: 8px;
  --ease: cubic-bezier(.2,.8,.2,1);
  --font-display: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --font-body: 'Inter', 'Source Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'Source Code Pro', ui-monospace, monospace;
}

/* ── canvas ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] { background: var(--bg) !important; }
[data-testid="stApp"] { color: var(--ink); font-family: var(--font-body); }
[data-testid="stHeader"] { background: var(--bg) !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1320px; padding-top: 32px !important; padding-bottom: 64px !important;
}

/* ── typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h1 *, h2 *, h3 *, h4 *, h5 *, [data-testid="stHeading"] * { font-family: var(--font-display) !important; font-weight: 500 !important; letter-spacing: 0; color: var(--ink); }
p, li, label, span, div { font-family: var(--font-body); }
code, pre, .tml-mono { font-family: var(--font-mono) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--ink-2); line-height: 1.6; }
[data-testid="stMarkdownContainer"] strong { color: var(--ink); font-weight: 500; }
i, em { font-style: normal; color: var(--ink); }
[data-testid="stTable"] th { font-weight: 500; color: var(--ink-2); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
a { color: var(--brass-2) !important; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"], .tml-num { font-variant-numeric: tabular-nums; }
.msr, h1 .msr, h2 .msr, h3 .msr, h4 .msr, h5 .msr, [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal; line-height: 1;
       letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
       -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }

/* ── sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: transparent; }
[data-testid="stSidebarNav"] { padding-top: 4px; }
[data-testid="stNavSectionHeader"] {
  font-family: var(--font-body); text-transform: none; letter-spacing: 0; font-size: 12px !important; font-weight: 500;
  color: var(--muted) !important; margin-top: 20px; margin-bottom: 4px; padding-left: 12px;
}
[data-testid="stSidebarNavLink"] {
  border-radius: var(--radius-xs) !important; margin: 2px 8px; padding: 6px 12px !important; min-height: 32px;
  transition: background-color .12s var(--ease), color .12s var(--ease); border: 0;
}
[data-testid="stSidebarNavLink"]:hover { background: var(--surface) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: var(--surface-2) !important; box-shadow: inset 2px 0 0 var(--brass);
}
[data-testid="stSidebarNavLink"] span { color: var(--ink-2) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--ink) !important; font-weight: 500; }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line) !important; }

/* ── containers: flat surfaces with hairlines ──────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-glass"] {
  background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
}
div[class*="st-key-glass"] { padding: 20px; }
[data-testid="stForm"] {
  background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  padding: 24px 24px 16px !important;
}
[data-testid="stExpander"] details { background: var(--surface); border: 1px solid var(--line) !important; border-radius: var(--radius-sm) !important; }
[data-testid="stExpander"] summary:hover { color: var(--ink) !important; }

/* ── buttons ───────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  border-radius: var(--radius-sm) !important; font-family: var(--font-body) !important; font-weight: 500 !important; letter-spacing: 0;
  transition: background-color .12s var(--ease), border-color .12s var(--ease), color .12s var(--ease), transform .12s var(--ease) !important;
}
.stButton > button:active, .stDownloadButton > button:active, .stFormSubmitButton > button:active { transform: scale(.98); }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--surface) !important; border: 1px solid var(--line-2) !important; color: var(--ink) !important;
}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  background: var(--surface-2) !important; border-color: rgba(232,236,242,.22) !important; color: var(--ink) !important;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--brass) !important; color: var(--brass-ink) !important; border: 0 !important; box-shadow: none !important;
}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: var(--brass-ink) !important; font-weight: 500; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { background: var(--brass-2) !important; }
button:focus-visible, a:focus-visible, [role="tab"]:focus-visible { outline: 2px solid var(--brass) !important; outline-offset: 2px; }
[data-testid="stPageLink"] a {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 8px 12px;
  transition: background-color .12s var(--ease), border-color .12s var(--ease); text-decoration: none;
}
[data-testid="stPageLink"] a:hover { background: var(--surface-2); border-color: var(--line-2); }

/* ── inputs ────────────────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-testid="stNumberInputContainer"] {
  background: var(--bg) !important; border-color: var(--line-2) !important; border-radius: var(--radius-sm) !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--ink-2) !important; box-shadow: none; }
[data-testid="stSlider"] [role="slider"] { background: var(--ink) !important; box-shadow: none !important; }
[data-testid="stSliderThumbValue"] { color: var(--ink) !important; font-family: var(--font-mono) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface) !important; border: 1px dashed var(--line-2) !important; border-radius: var(--radius) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--ink-2) !important; font-weight: 500; }

/* ── tabs (react-aria in new Streamlit, baseweb in older versions) ── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px; background: var(--bg-2); border: 1px solid var(--line) !important; border-radius: var(--radius-sm); padding: 4px;
  width: fit-content; max-width: 100%; overflow-x: auto; box-shadow: none !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 6px !important; padding: 6px 16px !important; height: auto !important; background: transparent; margin: 0 !important;
  font-family: var(--font-body); transition: background-color .12s var(--ease); border: 0 !important; cursor: pointer;
}
[data-testid="stTabs"] [role="tab"]:hover { background: var(--surface); }
[data-testid="stTabs"] [role="tab"] p { color: var(--ink-2); font-weight: 400; margin: 0; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--surface-2); box-shadow: inset 0 0 0 1px var(--line-2); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: var(--ink); font-weight: 500; }
[data-testid="stTabs"] .react-aria-SelectionIndicator, [data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 16px; }

/* ── metrics, alerts, tables ───────────────────────────────────────── */
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--ink) !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; text-transform: none; letter-spacing: 0; font-size: 13px !important; }
[data-testid="stAlert"] { border-radius: var(--radius-sm) !important; border: 1px solid var(--line) !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--line); }
[data-testid="stProgress"] > div > div > div > div { background: var(--ink-2) !important; }
hr { border-color: var(--line) !important; }
[data-testid="stToolbar"] { opacity: .6; }
[data-testid="stPlotlyChart"] { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 8px 8px 0; overflow: visible; }

/* ══ custom components ═══════════════════════════════════════════════ */
.tml-eyebrow { font-family: var(--font-mono); text-transform: none; letter-spacing: 0; font-size: 12px; color: var(--muted);
  display: inline-flex; align-items: center; gap: 8px; }
.tml-eyebrow .dot { display: none; }
.tml-page-head { margin: 4px 0 24px; }
.tml-page-head h1 { font-size: clamp(32px, 3.2vw, 44px) !important; font-weight: 500 !important; margin: 4px 0 8px !important; padding: 0 !important; line-height: 1.12 !important; color: var(--ink) !important; }
.tml-page-head .sub { color: var(--ink-2); font-size: 16px; max-width: 760px; line-height: 1.6; }
.tml-page-head .icon { font-size: 18px; }

.tml-section { display: flex; align-items: center; gap: 12px; margin: 32px 0 16px; }
.tml-section .num { font-family: var(--font-mono); font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.tml-section h3 { margin: 0 !important; padding: 0 !important; font-size: 22px !important; font-weight: 500 !important; }
.tml-section .line { flex: 1; height: 1px; background: var(--line); }
.tml-section .msr { color: var(--muted); font-size: 18px; }

.tml-grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(144px, 1fr)); margin: 8px 0 16px; }
.tml-kpi { display: flex; flex-direction: column; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 16px 20px; }
.tml-kpi .label { color: var(--muted); font-size: 13px; font-weight: 400; }
.tml-kpi .value { margin-top: auto !important; font-family: var(--font-display); font-size: 30px; font-weight: 500; color: var(--ink); margin-top: 4px; line-height: 1.15; font-variant-numeric: tabular-nums lining-nums; }
.tml-kpi .value.small { font-size: 22px; margin-top: 8px; overflow-wrap: anywhere; }
.tml-kpi .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }

.tml-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px; }
.tml-card h4 { margin: 0 0 4px !important; font-size: 17px !important; padding: 0 !important; }
.tml-card p { margin: 0; color: var(--ink-2); font-size: 14px; line-height: 1.55; }

.tml-insights { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.tml-insight { display: flex; gap: 12px; align-items: flex-start; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 16px; }
.tml-insight .ic { flex: 0 0 32px; height: 32px; border-radius: var(--radius-xs); display: grid; place-items: center; background: var(--surface-2); }
.tml-insight .ic .msr { color: var(--ink-2); font-size: 18px; }
.tml-insight h5 { margin: 0 0 4px; font-family: var(--font-body); font-size: 15px; color: var(--ink); font-weight: 500; }
.tml-insight p { margin: 0; color: var(--ink-2); font-size: 14px; line-height: 1.5; }

.tml-learn { display: flex; gap: 12px; background: var(--bg-2); border: 1px solid var(--line); border-left: 2px solid var(--line-2);
  border-radius: var(--radius-sm); padding: 12px 16px; margin: 8px 0 16px; }
.tml-learn .msr { color: var(--muted); font-size: 18px; margin-top: 2px; }
.tml-learn b { color: var(--ink); font-weight: 500; }
.tml-learn p, .tml-learn div { color: var(--ink-2); font-size: 14px; line-height: 1.55; margin: 0; }

.tml-note { display: flex; gap: 12px; align-items: flex-start; background: var(--surface); border: 1px solid var(--line);
  border-radius: var(--radius-sm); padding: 12px 16px; color: var(--ink-2); font-size: 14px; margin: 8px 0 16px; }
.tml-note b { color: var(--ink); font-weight: 500; }
.tml-note .msr { color: var(--muted); font-size: 18px; margin-top: 1px; }

.tml-pill { display: inline-flex; align-items: center; gap: 4px; border-radius: 6px; padding: 2px 8px; font-size: 12px; font-weight: 500;
  border: 1px solid var(--line-2); color: var(--ink-2); background: transparent; }
.tml-pill.survived { color: #9cc4ee; border-color: rgba(58,142,230,.35); }
.tml-pill.perished { color: #efae90; border-color: rgba(232,105,58,.35); }
.tml-pill.brass { color: var(--brass-2); border-color: rgba(227,184,102,.4); }
.tml-pill.ok { color: #8fd3b9; border-color: rgba(31,168,131,.35); }

/* hero */
.tml-hero { position: relative; padding: 8px 0 0; }
.tml-hero h1 { font-size: clamp(40px, 5vw, 64px) !important; line-height: 1.05 !important; font-weight: 500 !important; margin: 8px 0 12px !important; padding: 0 !important;
  letter-spacing: -0.01em; color: var(--ink) !important; }
.tml-hero .tag { font-size: 17px; color: var(--ink-2); max-width: 720px; line-height: 1.6; }
.tml-hero .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }

[data-testid="stSidebarLogo"], [data-testid="stLogo"] { height: 40px !important; max-width: 240px !important; }
[data-testid="stSidebarHeader"] { padding-top: 16px; padding-bottom: 8px; }
/* sidebar brand */
.tml-brand { display: flex; gap: 12px; align-items: center; padding: 4px 4px 16px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }
.tml-brand .logo { width: 40px; height: 40px; border-radius: var(--radius-sm); display: grid; place-items: center; background: var(--surface-2); border: 1px solid var(--line-2); }
.tml-brand .name { font-family: var(--font-display); font-weight: 500; font-size: 16px; color: var(--ink); line-height: 1.2; }
.tml-brand .name span { display: block; color: var(--muted); font-family: var(--font-body); font-size: 12px; font-weight: 400; margin-top: 2px; }
.tml-status { font-size: 13px; color: var(--ink-2); background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 8px 12px; line-height: 1.6; font-variant-numeric: tabular-nums; }
.tml-status b { color: var(--ink); font-weight: 500; }

/* boarding pass: the survival-probability readout */
.tml-pass-wrap { perspective: 1600px; margin: 4px 0 16px; }
.tml-pass { position: relative; display: grid; grid-template-columns: 1fr 260px; min-height: 280px; max-width: 980px; transform-style: preserve-3d;
  transition: transform .6s var(--ease); animation: tml-pass-in .3s var(--ease) both; outline: none; cursor: pointer; }
.tml-pass-wrap:hover .tml-pass, .tml-pass:focus { transform: rotateY(180deg); }
.tml-pass:focus-visible .face { box-shadow: 0 0 0 2px var(--brass); }
@keyframes tml-pass-in { from { opacity: 0; transform: translateY(8px) } to { opacity: 1; transform: none } }
.tml-pass .face { grid-area: 1 / 1 / 2 / 3; display: grid; grid-template-columns: 1fr 260px; backface-visibility: hidden; -webkit-backface-visibility: hidden;
  border-radius: 16px; overflow: hidden; border: 1px solid var(--line-2); box-shadow: 0 16px 40px -24px rgba(0,0,0,.8); }
.tml-pass .back { transform: rotateY(180deg); grid-template-columns: 1fr; background: var(--surface); padding: 20px 24px; }
.tml-pass .main { position: relative; padding: 20px 24px; background: var(--surface); }
.tml-pass .stub { position: relative; padding: 20px 16px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
  background: var(--surface-2); border-left: 1px dashed var(--line-2); }
.tml-pass .stub::before, .tml-pass .stub::after { content: ""; position: absolute; left: -12px; width: 24px; height: 24px; border-radius: 50%; background: var(--bg); box-shadow: inset 0 0 0 1px var(--line-2); }
.tml-pass .stub::before { top: -12px; } .tml-pass .stub::after { bottom: -12px; }
.tml-pass .top { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--line-2); padding-bottom: 8px; margin-bottom: 12px; }
.tml-pass .top .brand { font-family: var(--font-display); font-weight: 500; font-size: 15px; color: var(--ink-2); }
.tml-pass .top .kind { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
.tml-pass .who { font-family: var(--font-display); font-size: 24px; color: var(--ink); font-weight: 500; margin-bottom: 12px; }
.tml-pass .fields { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px 16px; }
.tml-pass .f .k { font-size: 12px; color: var(--muted); }
.tml-pass .f .v { font-family: var(--font-mono); font-size: 15px; color: var(--ink); margin-top: 2px; }
.tml-pass .route { display: flex; align-items: center; gap: 8px; margin-top: 16px; font-family: var(--font-mono); font-size: 12px; color: var(--ink-2); }
.tml-pass .route .lineb { flex: 1; height: 1px; background: repeating-linear-gradient(90deg, var(--line-2) 0 6px, transparent 6px 12px); }
.tml-pass .route .msr { color: var(--muted); font-size: 18px; }
.tml-pass .hint { position: absolute; right: 16px; bottom: 12px; font-size: 12px; color: var(--muted); display: flex; gap: 4px; align-items: center; }
.tml-pass .hint .msr { font-size: 16px; }
.tml-pass .stamp { position: absolute; top: 56px; right: 24px; transform: rotate(-6deg); border: 2px solid var(--stamp); color: var(--stamp);
  border-radius: 6px; padding: 4px 8px; font-family: var(--font-mono); font-weight: 500; font-size: 12px; opacity: .9; }
.tml-ring { position: relative; width: 150px; height: 150px; }
.tml-ring svg { transform: rotate(-90deg); }
.tml-ring .track { stroke: var(--line); }
.tml-ring .bar { stroke: var(--ring); stroke-linecap: round; animation: tml-ring .8s .1s var(--ease) both; }
@keyframes tml-ring { from { stroke-dashoffset: var(--circ) } }
.tml-ring .center { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.tml-ring .pct { font-family: var(--font-display); font-weight: 500; font-size: 36px; color: var(--ink); line-height: 1; font-variant-numeric: tabular-nums lining-nums; }
.tml-ring .lbl { font-size: 12px; color: var(--muted); margin-top: 4px; }
.tml-pass .outcome { font-weight: 500; font-size: 15px; color: var(--ring); text-align: center; }
.tml-pass .conf { font-size: 12px; color: var(--muted); text-align: center; }
.tml-pass .back h4 { margin: 0 0 12px !important; padding: 0 !important; font-size: 17px !important; color: var(--ink); }
.tml-pass .back h4 .msr { color: var(--muted); font-size: 18px; }
.tml-factor { display: grid; grid-template-columns: 170px 1fr 72px; gap: 8px; align-items: center; margin: 4px 0; font-size: 14px; color: var(--ink-2); }
.tml-factor .bar { height: 8px; border-radius: 4px; background: var(--line); position: relative; overflow: hidden; }
.tml-factor .bar i { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.tml-factor .val { font-family: var(--font-mono); text-align: right; color: var(--ink); font-variant-numeric: tabular-nums; }

@media (max-width: 760px) {
  .tml-pass, .tml-pass .face { grid-template-columns: 1fr; }
  .tml-pass .stub { border-left: 0; border-top: 1px dashed var(--line-2); }
  .tml-pass .stub::before, .tml-pass .stub::after { display: none; }
  .tml-pass .fields { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .tml-factor { grid-template-columns: 112px 1fr 56px; }
  .tml-pass .stamp { position: static; display: inline-block; margin: 0 0 12px; }
  .tml-pass .hint { position: static; margin-top: 12px; }
  .tml-pass .top { flex-wrap: wrap; gap: 4px; }
}

/* quiz */
.tml-quiz-card { background: var(--surface); border: 1px solid var(--line-2); border-radius: 16px; padding: 20px 24px; }
.tml-score { display: flex; gap: 12px; justify-content: center; }
.tml-score .s { text-align: center; padding: 8px 20px; border-radius: var(--radius-sm); background: var(--surface); border: 1px solid var(--line); min-width: 128px; }
.tml-score .s .n { font-family: var(--font-display); font-size: 32px; font-weight: 500; color: var(--ink); font-variant-numeric: tabular-nums lining-nums; }
.tml-score .s .t { font-size: 13px; color: var(--muted); }

/* timeline of steps (pipeline diagram) */
.tml-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 8px; align-items: stretch; margin: 8px 0 16px; }
.tml-flow .step { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 12px 16px; position: relative; }
.tml-flow .step .n { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
.tml-flow .step .t { color: var(--ink); font-weight: 500; margin: 4px 0; }
.tml-flow .step .d { font-size: 13px; color: var(--ink-2); line-height: 1.45; }
.tml-flow .step .msr { position: absolute; right: 12px; top: 12px; color: var(--muted); font-size: 18px; }

@media (max-width: 760px) {
  .tml-score { gap: 8px; }
  .tml-score .s { min-width: 0; flex: 1; padding: 8px; }
  .tml-quiz-card .fields { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
}

@media (prefers-reduced-motion: reduce) { .tml-pass, .tml-ring .bar { animation: none !important; transition: none !important; } }

/* hide default "Deploy"/hamburger clutter but keep the sidebar toggle */
[data-testid="stAppDeployButton"], #MainMenu { visibility: hidden; }
footer { visibility: hidden; }
"""


def inject_theme() -> None:
    # st.markdown (not st.html) so the rules apply to the whole page, not an isolated container.
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
