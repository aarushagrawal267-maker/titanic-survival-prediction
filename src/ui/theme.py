"""Global "Midnight Atlantic" theme — injected once per page run."""
from __future__ import annotations

import streamlit as st

# Design tokens (data colours validated for colour-blind separation on the navy surface)
INK = "#eaf0fa"
INK_2 = "#b4c2d9"
MUTED = "#7d8ca8"
SURFACE = "#0d1a33"
BG = "#050b1a"
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg: #050b1a; --bg-2: #081127; --surface: #0d1a33; --surface-2: #12244a;
  --glass: rgba(13, 26, 51, 0.58); --glass-strong: rgba(13, 26, 51, 0.82);
  --line: rgba(255, 255, 255, 0.08); --line-2: rgba(255, 255, 255, 0.14);
  --ink: #eaf0fa; --ink-2: #b4c2d9; --muted: #7d8ca8;
  --brass: #e3b866; --brass-2: #f5dca6; --brass-deep: #b98a37;
  --survived: #3a8ee6; --perished: #e8693a; --aqua: #1fa883; --violet: #8a7ae6; --gold: #c08a1c;
  --radius: 18px; --radius-sm: 12px;
  --shadow: 0 10px 40px -12px rgba(0, 0, 0, 0.65), 0 0 0 1px rgba(255,255,255,0.04) inset;
  --font-display: 'Space Grotesk', 'Inter', 'Source Sans', system-ui, sans-serif;
  --font-body: 'Inter', 'Source Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', 'Source Code Pro', ui-monospace, monospace;
}

/* ── canvas: deep ocean night, aurora and stars ───────────────────────── */
html, body, [data-testid="stApp"] { background: var(--bg) !important; }
[data-testid="stApp"] {
  background:
    radial-gradient(1100px 620px at 8% -8%, rgba(58,142,230,0.20), transparent 60%),
    radial-gradient(900px 520px at 100% 0%, rgba(138,122,230,0.15), transparent 62%),
    radial-gradient(1000px 700px at 50% 115%, rgba(31,168,131,0.10), transparent 60%),
    linear-gradient(180deg, #050b1a 0%, #071028 45%, #040914 100%) !important;
  background-attachment: fixed !important;
  color: var(--ink); font-family: var(--font-body);
}
[data-testid="stApp"]::before, [data-testid="stApp"]::after {
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
}
[data-testid="stApp"]::before {   /* star field (two tiled layers) */
  background-image:
    radial-gradient(1px 1px at 23px 41px, rgba(255,255,255,.85), transparent 60%),
    radial-gradient(1px 1px at 132px 97px, rgba(255,255,255,.6), transparent 60%),
    radial-gradient(1.5px 1.5px at 301px 211px, rgba(207,224,255,.9), transparent 60%),
    radial-gradient(1px 1px at 411px 23px, rgba(255,255,255,.55), transparent 60%),
    radial-gradient(1px 1px at 487px 341px, rgba(255,255,255,.7), transparent 60%),
    radial-gradient(1.2px 1.2px at 77px 389px, rgba(243,215,155,.8), transparent 60%),
    radial-gradient(1px 1px at 213px 299px, rgba(255,255,255,.5), transparent 60%),
    radial-gradient(1px 1px at 361px 457px, rgba(255,255,255,.65), transparent 60%),
    radial-gradient(1.4px 1.4px at 529px 133px, rgba(255,255,255,.8), transparent 60%);
  background-size: 560px 520px;
  opacity: .55; animation: tml-twinkle 9s ease-in-out infinite alternate;
}
[data-testid="stApp"]::after {    /* slow aurora */
  background:
    radial-gradient(40% 30% at 30% 20%, rgba(58,142,230,.10), transparent 70%),
    radial-gradient(35% 25% at 70% 10%, rgba(31,168,131,.08), transparent 70%);
  filter: blur(30px); animation: tml-aurora 26s ease-in-out infinite alternate;
}
@keyframes tml-twinkle { 0% { opacity: .35 } 50% { opacity: .6 } 100% { opacity: .45 } }
@keyframes tml-aurora { 0% { transform: translate3d(-4%, 0, 0) scale(1) } 100% { transform: translate3d(5%, 3%, 0) scale(1.15) } }
[data-testid="stMain"], [data-testid="stSidebar"], [data-testid="stHeader"] { position: relative; z-index: 1; }

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1320px; padding-top: 2.2rem !important; padding-bottom: 4rem !important;
}
[data-testid="stMainBlockContainer"] > div { animation: tml-fade-up .55s cubic-bezier(.2,.7,.2,1) both; }
@keyframes tml-fade-up { from { opacity: 0; transform: translateY(10px) } to { opacity: 1; transform: none } }

/* ── typography ─────────────────────────────────────────────────────── */
h1, h2, h3, h4, [data-testid="stHeading"] * { font-family: var(--font-display) !important; letter-spacing: -0.01em; color: var(--ink); }
h2 { font-weight: 600 !important; }
p, li, label, span, div { font-family: var(--font-body); }
code, pre, .tml-mono { font-family: var(--font-mono) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--ink-2); line-height: 1.65; }
[data-testid="stMarkdownContainer"] strong { color: var(--ink); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
a { color: var(--brass) !important; }
.msr { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
       letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap;
       -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased; }

/* ── sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(9,19,43,.94) 0%, rgba(5,11,26,.97) 100%) !important;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: transparent; }
[data-testid="stSidebarNav"] { padding-top: .25rem; }
[data-testid="stNavSectionHeader"] {
  font-family: var(--font-display); text-transform: uppercase; letter-spacing: .16em; font-size: .68rem !important;
  color: var(--brass) !important; opacity: .85; margin-top: .9rem; padding-left: .75rem;
}
[data-testid="stSidebarNavLink"] {
  border-radius: 10px !important; margin: 1px 6px; padding: .38rem .7rem !important;
  transition: background .2s, box-shadow .2s, transform .2s; border: 1px solid transparent;
}
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,.05) !important; transform: translateX(2px); }
[data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, rgba(227,184,102,.16), rgba(227,184,102,.02)) !important;
  border: 1px solid rgba(227,184,102,.25);
  box-shadow: inset 3px 0 0 var(--brass), 0 0 24px -8px rgba(227,184,102,.45);
}
[data-testid="stSidebarNavLink"] span { color: var(--ink-2) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: var(--ink) !important; font-weight: 600; }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line) !important; }

/* ── bordered containers become glass cards ────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-glass"] {
  background: var(--glass) !important; border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow); backdrop-filter: blur(14px) saturate(1.2); -webkit-backdrop-filter: blur(14px) saturate(1.2);
}
div[class*="st-key-glass"] { padding: 1.1rem 1.2rem; }
[data-testid="stForm"] {
  background: var(--glass) !important; border: 1px solid var(--line-2) !important; border-radius: var(--radius) !important;
  box-shadow: var(--shadow); backdrop-filter: blur(14px); padding: 1.4rem 1.4rem 1rem !important;
}
[data-testid="stExpander"] details {
  background: rgba(13,26,51,.45); border: 1px solid var(--line) !important; border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--brass) !important; }

/* ── buttons ───────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  border-radius: 12px !important; font-family: var(--font-display) !important; font-weight: 600 !important;
  letter-spacing: .01em; transition: transform .15s ease, box-shadow .2s ease, background .2s ease !important;
}
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  background: rgba(255,255,255,.04) !important; border: 1px solid var(--line-2) !important; color: var(--ink) !important;
}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  border-color: rgba(227,184,102,.55) !important; color: var(--brass-2) !important; transform: translateY(-1px);
  box-shadow: 0 8px 24px -10px rgba(227,184,102,.45);
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, #f5dca6 0%, #e3b866 45%, #c9953f 100%) !important;
  color: #1a1305 !important; border: 0 !important;
  box-shadow: 0 10px 30px -10px rgba(227,184,102,.65), inset 0 1px 0 rgba(255,255,255,.5);
}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: #1a1305 !important; font-weight: 700; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform: translateY(-2px); box-shadow: 0 16px 36px -10px rgba(227,184,102,.75), inset 0 1px 0 rgba(255,255,255,.6);
}
[data-testid="stPageLink"] a {
  background: rgba(255,255,255,.04); border: 1px solid var(--line-2); border-radius: 12px; padding: .5rem .9rem;
  transition: all .2s; text-decoration: none;
}
[data-testid="stPageLink"] a:hover { border-color: var(--brass); box-shadow: 0 0 22px -8px rgba(227,184,102,.6); }

/* ── inputs ────────────────────────────────────────────────────────── */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-testid="stNumberInputContainer"] {
  background: rgba(5,11,26,.55) !important; border-color: var(--line-2) !important; border-radius: 10px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--brass) !important; box-shadow: 0 0 0 3px rgba(227,184,102,.15); }
[data-testid="stSlider"] [role="slider"] { background: var(--brass) !important; box-shadow: 0 0 0 4px rgba(227,184,102,.2), 0 0 18px rgba(227,184,102,.6); }
[data-testid="stSliderThumbValue"] { color: var(--brass) !important; font-family: var(--font-mono) !important; }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(13,26,51,.45) !important; border: 1.5px dashed rgba(227,184,102,.35) !important; border-radius: var(--radius) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--ink-2) !important; font-weight: 500; }

/* ── tabs (react-aria in new Streamlit, baseweb in older versions) ── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px; background: rgba(13,26,51,.55); border: 1px solid var(--line) !important; border-radius: 14px; padding: 4px;
  width: fit-content; max-width: 100%; overflow-x: auto; box-shadow: none !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 10px !important; padding: .45rem .95rem !important; height: auto !important; background: transparent; margin: 0 !important;
  font-family: var(--font-display); transition: background .2s; border: 0 !important; cursor: pointer;
}
[data-testid="stTabs"] [role="tab"]:hover { background: rgba(255,255,255,.05); }
[data-testid="stTabs"] [role="tab"] p { color: var(--ink-2); font-weight: 500; margin: 0; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: linear-gradient(135deg, rgba(227,184,102,.22), rgba(227,184,102,.08)); box-shadow: inset 0 0 0 1px rgba(227,184,102,.35); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: var(--brass-2); font-weight: 600; }
[data-testid="stTabs"] .react-aria-SelectionIndicator, [data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: .9rem; }

/* ── metrics, alerts, tables ───────────────────────────────────────── */
[data-testid="stMetric"] {
  background: var(--glass); border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem; box-shadow: var(--shadow);
}
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--ink) !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; text-transform: uppercase; letter-spacing: .08em; font-size: .72rem !important; }
[data-testid="stAlert"] { border-radius: 14px !important; border: 1px solid var(--line) !important; backdrop-filter: blur(10px); }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--survived), var(--brass)) !important; }
hr { border-color: var(--line) !important; }
[data-testid="stToolbar"] { opacity: .6; }
[data-testid="stPlotlyChart"] { background: var(--glass); border: 1px solid var(--line); border-radius: var(--radius); padding: .4rem .5rem .2rem; box-shadow: var(--shadow); }

/* ══ custom components ═══════════════════════════════════════════════ */
.tml-eyebrow { font-family: var(--font-display); text-transform: uppercase; letter-spacing: .22em; font-size: .72rem;
  color: var(--brass); display: inline-flex; align-items: center; gap: .5rem; }
.tml-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--brass); box-shadow: 0 0 12px var(--brass); }
.tml-page-head { margin: .2rem 0 1.4rem; }
.tml-page-head h1 { font-size: clamp(2rem, 3.4vw, 2.9rem) !important; font-weight: 700 !important; margin: .35rem 0 .35rem !important; padding: 0 !important; line-height: 1.08 !important;
  background: linear-gradient(95deg, #ffffff 0%, #d8e6ff 45%, #f3d79b 100%); -webkit-background-clip: text; background-clip: text; color: transparent !important; }
.tml-page-head .sub { color: var(--ink-2); font-size: 1.04rem; max-width: 820px; line-height: 1.6; }
.tml-page-head .icon { font-size: 1.1rem; }

.tml-section { display: flex; align-items: center; gap: .75rem; margin: 1.8rem 0 .9rem; }
.tml-section .num { font-family: var(--font-mono); font-size: .72rem; color: var(--brass); border: 1px solid rgba(227,184,102,.35);
  border-radius: 999px; padding: .1rem .55rem; }
.tml-section h3 { margin: 0 !important; padding: 0 !important; font-size: 1.28rem !important; font-weight: 600 !important; }
.tml-section .line { flex: 1; height: 1px; background: linear-gradient(90deg, var(--line-2), transparent); }
.tml-section .msr { color: var(--brass); font-size: 1.3rem; }

.tml-grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: .4rem 0 1rem; }
.tml-kpi { position: relative; overflow: hidden; background: var(--glass); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.1rem 1.05rem; box-shadow: var(--shadow); backdrop-filter: blur(14px); transition: transform .25s, border-color .25s, box-shadow .25s;
  animation: tml-fade-up .6s cubic-bezier(.2,.7,.2,1) both; }
.tml-kpi:hover { transform: translateY(-3px); border-color: rgba(227,184,102,.35); box-shadow: 0 18px 40px -16px rgba(0,0,0,.8), 0 0 30px -12px rgba(227,184,102,.35); }
.tml-kpi::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 2px; background: linear-gradient(90deg, transparent, var(--accent, var(--brass)), transparent); opacity: .9; }
.tml-kpi::after { content: ""; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px; border-radius: 50%;
  background: radial-gradient(circle, color-mix(in srgb, var(--accent, var(--brass)) 22%, transparent), transparent 70%); }
.tml-kpi .label { display: flex; align-items: center; gap: .45rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-size: .7rem; font-weight: 600; }
.tml-kpi .label .msr { font-size: 1.05rem; color: var(--accent, var(--brass)); }
.tml-kpi .value { font-family: var(--font-display); font-size: 1.95rem; font-weight: 600; color: var(--ink); margin-top: .35rem; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tml-kpi .value.small { font-size: 1.32rem; margin-top: .55rem; overflow-wrap: anywhere; }
.tml-kpi .sub { color: var(--ink-2); font-size: .82rem; margin-top: .25rem; }

@property --tml-n { syntax: '<integer>'; initial-value: 0; inherits: false; }
.tml-count { animation: tml-count 1.6s cubic-bezier(.2,.8,.2,1) forwards; counter-reset: n var(--tml-n); }
.tml-count::after { content: counter(n); }
@keyframes tml-count { from { --tml-n: 0 } to { --tml-n: var(--to) } }

.tml-card { background: var(--glass); border: 1px solid var(--line); border-radius: var(--radius); padding: 1.1rem 1.2rem; box-shadow: var(--shadow); backdrop-filter: blur(14px); }
.tml-card h4 { margin: 0 0 .35rem !important; font-size: 1.02rem !important; padding: 0 !important; }
.tml-card p { margin: 0; color: var(--ink-2); font-size: .92rem; line-height: 1.55; }

.tml-insights { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.tml-insight { display: flex; gap: .85rem; align-items: flex-start; background: var(--glass); border: 1px solid var(--line); border-radius: 16px;
  padding: .95rem 1rem; transition: border-color .25s, transform .25s; animation: tml-fade-up .6s both; }
.tml-insight:hover { border-color: rgba(58,142,230,.4); transform: translateY(-2px); }
.tml-insight .ic { flex: 0 0 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
  background: linear-gradient(135deg, rgba(58,142,230,.25), rgba(138,122,230,.15)); border: 1px solid rgba(58,142,230,.3); }
.tml-insight .ic .msr { color: #cfe0ff; font-size: 1.25rem; }
.tml-insight h5 { margin: 0 0 .2rem; font-family: var(--font-display); font-size: .98rem; color: var(--ink); font-weight: 600; }
.tml-insight p { margin: 0; color: var(--ink-2); font-size: .88rem; line-height: 1.5; }

.tml-learn { display: flex; gap: .8rem; background: linear-gradient(135deg, rgba(31,168,131,.10), rgba(58,142,230,.06)); border: 1px solid rgba(31,168,131,.28);
  border-radius: 14px; padding: .85rem 1rem; margin: .4rem 0 1rem; }
.tml-learn .msr { color: #5fd3ad; font-size: 1.3rem; }
.tml-learn b { color: #9de8cf; font-family: var(--font-display); }
.tml-learn p, .tml-learn div { color: var(--ink-2); font-size: .9rem; line-height: 1.55; margin: 0; }

.tml-note { display: flex; gap: .7rem; align-items: flex-start; background: rgba(227,184,102,.06); border: 1px solid rgba(227,184,102,.22);
  border-radius: 12px; padding: .7rem .9rem; color: var(--ink-2); font-size: .86rem; margin: .5rem 0 1rem; }
.tml-note .msr { color: var(--brass); font-size: 1.15rem; }

.tml-pill { display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px; padding: .18rem .65rem; font-size: .76rem; font-weight: 600;
  border: 1px solid var(--line-2); color: var(--ink-2); background: rgba(255,255,255,.04); }
.tml-pill.survived { color: #9cc8f6; border-color: rgba(58,142,230,.45); background: rgba(58,142,230,.12); }
.tml-pill.perished { color: #f6b08f; border-color: rgba(232,105,58,.45); background: rgba(232,105,58,.12); }
.tml-pill.brass { color: var(--brass-2); border-color: rgba(227,184,102,.45); background: rgba(227,184,102,.10); }
.tml-pill.ok { color: #8ee0c0; border-color: rgba(31,168,131,.45); background: rgba(31,168,131,.12); }

/* hero */
.tml-hero { position: relative; padding: .4rem 0 0; }
.tml-hero h1 { font-size: clamp(2.6rem, 6vw, 4.6rem) !important; line-height: 1 !important; font-weight: 700 !important; margin: .5rem 0 .6rem !important; padding: 0 !important;
  letter-spacing: -0.035em; background: linear-gradient(100deg, #ffffff 0%, #cfe0ff 38%, #f3d79b 72%, #e3b866 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent !important; filter: drop-shadow(0 6px 30px rgba(58,142,230,.25)); }
.tml-hero .tag { font-size: 1.15rem; color: var(--ink-2); max-width: 760px; line-height: 1.6; }
.tml-hero .chips { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1rem; }

[data-testid="stSidebarLogo"], [data-testid="stLogo"] { height: 2.6rem !important; max-width: 15rem !important; }
[data-testid="stSidebarHeader"] { padding-top: 1.1rem; padding-bottom: .4rem; }
/* sidebar brand */
.tml-brand { display: flex; gap: .7rem; align-items: center; padding: .2rem .2rem .9rem; border-bottom: 1px solid var(--line); margin-bottom: .4rem; }
.tml-brand .logo { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
  background: radial-gradient(circle at 30% 25%, #1d3a73, #0a1733 70%); border: 1px solid rgba(227,184,102,.4); box-shadow: 0 0 24px -6px rgba(227,184,102,.55); }
.tml-brand .name { font-family: var(--font-display); font-weight: 700; letter-spacing: .14em; font-size: .92rem; color: var(--ink); line-height: 1.15; }
.tml-brand .name span { display: block; color: var(--brass); font-size: .62rem; letter-spacing: .24em; font-weight: 500; margin-top: 2px; }
.tml-status { font-size: .78rem; color: var(--ink-2); background: rgba(255,255,255,.03); border: 1px solid var(--line); border-radius: 12px; padding: .6rem .7rem; line-height: 1.6; }
.tml-status b { color: var(--ink); font-family: var(--font-display); }

/* boarding pass */
.tml-pass-wrap { perspective: 1600px; margin: .3rem 0 1rem; }
.tml-pass { position: relative; display: grid; grid-template-columns: 1fr 260px; min-height: 280px; max-width: 980px; transform-style: preserve-3d;
  transition: transform .9s cubic-bezier(.2,.8,.2,1); animation: tml-pass-in .9s cubic-bezier(.2,.8,.2,1) both; outline: none; cursor: pointer; }
.tml-pass-wrap:hover .tml-pass, .tml-pass:focus { transform: rotateY(180deg); }
@keyframes tml-pass-in { from { opacity: 0; transform: translateY(26px) rotateX(18deg) } to { opacity: 1; transform: none } }
.tml-pass .face { grid-area: 1 / 1 / 2 / 3; display: grid; grid-template-columns: 1fr 260px; backface-visibility: hidden; -webkit-backface-visibility: hidden;
  border-radius: 20px; overflow: hidden; box-shadow: 0 30px 60px -24px rgba(0,0,0,.85), 0 0 0 1px rgba(227,184,102,.28); }
.tml-pass .back { transform: rotateY(180deg); grid-template-columns: 1fr; background: linear-gradient(135deg, #0f1f40, #0a1530); padding: 1.3rem 1.5rem; }
.tml-pass .main { position: relative; padding: 1.2rem 1.5rem 1.2rem; background:
  linear-gradient(135deg, rgba(243,215,155,.10), transparent 40%), repeating-linear-gradient(0deg, rgba(255,255,255,.012) 0 2px, transparent 2px 4px),
  linear-gradient(160deg, #13284f 0%, #0c1a38 60%, #0a1630 100%); }
.tml-pass .stub { position: relative; padding: 1.2rem 1.1rem; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .5rem;
  background: linear-gradient(160deg, #182f5c, #0d1d3f); border-left: 2px dashed rgba(227,184,102,.38); }
.tml-pass .stub::before, .tml-pass .stub::after { content: ""; position: absolute; left: -13px; width: 24px; height: 24px; border-radius: 50%; background: var(--bg); box-shadow: inset 0 0 0 1px rgba(227,184,102,.28); }
.tml-pass .stub::before { top: -12px; } .tml-pass .stub::after { bottom: -12px; }
.tml-pass .top { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(227,184,102,.25); padding-bottom: .55rem; margin-bottom: .8rem; }
.tml-pass .top .brand { font-family: var(--font-display); font-weight: 700; letter-spacing: .26em; font-size: .78rem; color: var(--brass-2); }
.tml-pass .top .kind { font-family: var(--font-mono); font-size: .7rem; color: var(--muted); letter-spacing: .14em; }
.tml-pass .who { font-family: var(--font-display); font-size: 1.45rem; color: var(--ink); font-weight: 600; margin-bottom: .75rem; }
.tml-pass .fields { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem 1rem; }
.tml-pass .f .k { font-size: .62rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); }
.tml-pass .f .v { font-family: var(--font-mono); font-size: .98rem; color: var(--ink); margin-top: 2px; }
.tml-pass .route { display: flex; align-items: center; gap: .6rem; margin-top: .95rem; font-family: var(--font-mono); font-size: .78rem; color: var(--ink-2); }
.tml-pass .route .lineb { flex: 1; height: 1px; background: repeating-linear-gradient(90deg, rgba(227,184,102,.55) 0 6px, transparent 6px 11px); }
.tml-pass .hint { position: absolute; right: 1rem; bottom: .7rem; font-size: .68rem; color: var(--muted); letter-spacing: .08em; display: flex; gap: .3rem; align-items: center; }
.tml-pass .stamp { position: absolute; top: 3.3rem; right: 1.6rem; transform: rotate(-9deg); border: 2.5px solid var(--stamp); color: var(--stamp);
  border-radius: 10px; padding: .25rem .6rem; font-family: var(--font-display); font-weight: 700; letter-spacing: .14em; font-size: .72rem; text-transform: uppercase;
  opacity: .92; animation: tml-stamp .7s .55s cubic-bezier(.2,1.6,.4,1) both; mix-blend-mode: screen; }
@keyframes tml-stamp { 0% { opacity: 0; transform: rotate(-9deg) scale(2.4) } 100% { opacity: .92; transform: rotate(-9deg) scale(1) } }
.tml-ring { position: relative; width: 150px; height: 150px; }
.tml-ring svg { transform: rotate(-90deg); }
.tml-ring .track { stroke: rgba(255,255,255,.08); }
.tml-ring .bar { stroke: var(--ring); stroke-linecap: round; filter: drop-shadow(0 0 8px var(--ring)); animation: tml-ring 1.4s .3s cubic-bezier(.2,.8,.2,1) both; }
@keyframes tml-ring { from { stroke-dashoffset: var(--circ) } }
.tml-ring .center { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.tml-ring .pct { font-family: var(--font-display); font-weight: 700; font-size: 2.1rem; color: var(--ink); line-height: 1; }
.tml-ring .lbl { font-size: .6rem; color: var(--muted); letter-spacing: .16em; text-transform: uppercase; margin-top: 4px; }
.tml-pass .outcome { font-family: var(--font-display); font-weight: 600; font-size: .95rem; color: var(--ring); text-align: center; }
.tml-pass .conf { font-size: .72rem; color: var(--muted); text-align: center; }
.tml-pass .back h4 { margin: 0 0 .6rem !important; padding: 0 !important; font-size: 1.05rem !important; color: var(--brass-2); }
.tml-factor { display: grid; grid-template-columns: 170px 1fr 70px; gap: .6rem; align-items: center; margin: .32rem 0; font-size: .86rem; color: var(--ink-2); }
.tml-factor .bar { height: 8px; border-radius: 4px; background: rgba(255,255,255,.06); position: relative; overflow: hidden; }
.tml-factor .bar i { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.tml-factor .val { font-family: var(--font-mono); text-align: right; color: var(--ink); }

@media (max-width: 760px) {
  .tml-pass, .tml-pass .face { grid-template-columns: 1fr; }
  .tml-pass .stub { border-left: 0; border-top: 2px dashed rgba(227,184,102,.38); }
  .tml-pass .stub::before, .tml-pass .stub::after { display: none; }
  .tml-pass .fields { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .tml-factor { grid-template-columns: 110px 1fr 56px; }
  .tml-pass .stamp { position: static; display: inline-block; margin: 0 0 .7rem; }
  .tml-pass .hint { position: static; margin-top: .7rem; }
  .tml-pass .top { flex-wrap: wrap; gap: .3rem; }
}

/* quiz */
.tml-quiz-card { background: linear-gradient(160deg, #13284f, #0b1834); border: 1px solid rgba(227,184,102,.3); border-radius: 20px; padding: 1.3rem 1.5rem;
  box-shadow: 0 24px 50px -24px rgba(0,0,0,.9); animation: tml-fade-up .5s both; }
.tml-score { display: flex; gap: 1rem; justify-content: center; }
.tml-score .s { text-align: center; padding: .6rem 1.2rem; border-radius: 14px; background: var(--glass); border: 1px solid var(--line); min-width: 130px; }
.tml-score .s .n { font-family: var(--font-display); font-size: 2rem; font-weight: 700; color: var(--ink); }
.tml-score .s .t { font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }

/* timeline of steps (pipeline diagram) */
.tml-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 10px; align-items: stretch; margin: .5rem 0 1rem; }
.tml-flow .step { background: var(--glass); border: 1px solid var(--line); border-radius: 14px; padding: .75rem .85rem; position: relative; }
.tml-flow .step .n { font-family: var(--font-mono); font-size: .66rem; color: var(--brass); letter-spacing: .12em; }
.tml-flow .step .t { font-family: var(--font-display); color: var(--ink); font-weight: 600; margin: .15rem 0 .2rem; }
.tml-flow .step .d { font-size: .78rem; color: var(--ink-2); line-height: 1.45; }
.tml-flow .step .msr { position: absolute; right: .7rem; top: .7rem; color: var(--brass); opacity: .8; }

/* hide default "Deploy"/hamburger clutter but keep the sidebar toggle */
[data-testid="stAppDeployButton"], #MainMenu { visibility: hidden; }
footer { visibility: hidden; }
"""


def inject_theme() -> None:
    # st.markdown (not st.html) so the rules apply to the whole page, not an isolated container.
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
