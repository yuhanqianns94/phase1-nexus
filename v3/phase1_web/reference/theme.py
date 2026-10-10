"""Visual theme for the v3 Streamlit application.

The app always presents a light research-workspace palette. Streamlit's native
theme is also fixed in v3/.streamlit/config.toml; these rules add layout polish
and keep browser dark-mode preferences from creating dark widgets on a light
page.
"""

APP_CSS = r"""
:root {
  color-scheme: light;
  --tox-ink: #17333b;
  --tox-muted: #5b7077;
  --tox-accent: #0f766e;
  --tox-accent-hover: #0b625c;
  --tox-accent-soft: #e5f2f0;
  --tox-page: #f7faf9;
  --tox-sidebar: #eef5f4;
  --tox-panel: #ffffff;
  --tox-panel-muted: #f3f7f6;
  --tox-line: #d6e1e0;
  --tox-line-strong: #bdcecc;
  --tox-info: #edf5f7;
  --tox-info-ink: #285766;
  --tox-success: #eaf6f0;
  --tox-success-ink: #225f49;
  --tox-shadow: 0 12px 30px rgba(23, 51, 59, 0.07);
}

html,
body,
[class*="css"],
[data-testid="stAppViewContainer"] {
  color: var(--tox-ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background:
    radial-gradient(900px 420px at 0% 0%, rgba(218, 238, 234, 0.62), transparent 60%),
    linear-gradient(180deg, #fbfcfb 0%, var(--tox-page) 55%, #f2f7f6 100%) !important;
}

header[data-testid="stHeader"] {
  background: rgba(247, 250, 249, 0.94) !important;
  border-bottom: 1px solid rgba(214, 225, 224, 0.82);
  backdrop-filter: blur(12px);
}

header[data-testid="stHeader"] button,
header[data-testid="stHeader"] [data-testid="stToolbar"] {
  color: var(--tox-ink) !important;
}

header[data-testid="stHeader"] svg {
  fill: currentColor !important;
}

.block-container {
  max-width: 1240px;
  padding-top: 1.15rem;
  padding-bottom: 4rem;
}

h1,
h2,
h3,
h4,
.tox-brand {
  color: var(--tox-ink) !important;
  font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif !important;
  letter-spacing: -0.02em;
}

h2 { margin-bottom: 0.25rem !important; }
h3, h4 { margin-top: 0.25rem !important; }

p,
li,
label,
.stMarkdown,
[data-testid="stMarkdownContainer"] {
  color: var(--tox-ink);
}

.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--tox-muted) !important;
}

.tox-hero {
  margin: 0.15rem 0 1.65rem;
  padding: 1.3rem 1.55rem 1.2rem;
  border: 1px solid var(--tox-line);
  border-radius: 22px;
  background: linear-gradient(135deg, #ffffff 0%, #edf7f5 100%);
  box-shadow: var(--tox-shadow);
}

.tox-brand {
  margin: 0;
  font-size: 2.05rem;
  font-weight: 700;
  line-height: 1.15;
}

.tox-subtitle {
  max-width: 46rem;
  margin: 0.45rem 0 0;
  color: var(--tox-muted);
  font-size: 1.02rem;
  line-height: 1.5;
}

.tox-hero-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 0.9rem;
}

.tox-pill {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 0.2rem 0.65rem;
  border: 1px solid #c9dfdc;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.86);
  color: #31565d;
  font-size: 0.78rem;
  font-weight: 650;
}

.tox-lead {
  max-width: 48rem;
  margin: 0 0 1.1rem;
  color: var(--tox-muted);
  font-size: 1.02rem;
  line-height: 1.55;
}

.tox-kicker {
  margin: 0 0 0.3rem;
  color: var(--tox-accent);
  font-size: 0.76rem;
  font-weight: 750;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.tox-card,
.tox-help {
  color: var(--tox-ink) !important;
  border: 1px solid var(--tox-line);
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.94);
}

.tox-card {
  margin: 0.85rem 0 1.25rem;
  padding: 1rem 1.15rem;
  box-shadow: 0 8px 22px rgba(23, 51, 59, 0.04);
}

.tox-help {
  margin: 0.4rem 0 1rem;
  padding: 0.85rem 1rem;
  line-height: 1.45;
}

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  background: var(--tox-sidebar) !important;
  color: var(--tox-ink) !important;
}

section[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.15rem;
  border-right: 1px solid var(--tox-line);
}

.tox-sidebar-brand {
  margin: 0.1rem 0 0.15rem;
  color: var(--tox-ink);
  font-family: "Iowan Old Style", Georgia, serif;
  font-size: 1.34rem;
  font-weight: 700;
}

.tox-sidebar-note {
  margin: 0 0 1rem;
  color: var(--tox-muted);
  font-size: 0.82rem;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
  color: var(--tox-ink) !important;
}

section[data-testid="stSidebar"] .stButton > button {
  min-height: 42px;
  justify-content: flex-start;
  padding-left: 0.9rem;
  border: 1px solid var(--tox-line-strong) !important;
  border-radius: 12px;
  background: #ffffff !important;
  color: var(--tox-ink) !important;
  font-weight: 650;
}

section[data-testid="stSidebar"] .stButton > button:hover {
  border-color: #9fc9c4 !important;
  background: var(--tox-accent-soft) !important;
  color: var(--tox-accent) !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  border-color: var(--tox-accent) !important;
  background: var(--tox-accent) !important;
  color: #ffffff !important;
  box-shadow: 0 7px 16px rgba(15, 118, 110, 0.18);
}

section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] * {
  color: #ffffff !important;
}

/* Cards, expanders and metrics */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--tox-line) !important;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.72);
}

div[data-testid="stExpander"] {
  overflow: hidden;
  border-color: var(--tox-line) !important;
  border-radius: 14px !important;
  background: var(--tox-panel) !important;
}

div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary:hover {
  background: var(--tox-panel-muted) !important;
  color: var(--tox-ink) !important;
}

div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span,
div[data-testid="stExpander"] summary svg {
  color: var(--tox-ink) !important;
  fill: currentColor !important;
}

div[data-testid="stMetric"] {
  padding: 0.95rem 1.05rem;
  border: 1px solid var(--tox-line);
  border-radius: 16px;
  background: var(--tox-panel);
  box-shadow: 0 8px 20px rgba(23, 51, 59, 0.035);
}

div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {
  color: var(--tox-muted) !important;
  font-weight: 600 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
  color: var(--tox-ink) !important;
  font-size: 1.15rem !important;
  font-weight: 700 !important;
  line-height: 1.35 !important;
  white-space: normal !important;
  overflow-wrap: anywhere;
}

/* Inputs and selectors */
input,
textarea,
button,
select {
  color-scheme: light;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div,
.stTextArea textarea {
  border-color: var(--tox-line-strong) !important;
  border-radius: 11px !important;
  background: #ffffff !important;
  color: var(--tox-ink) !important;
}

div[data-baseweb="input"] input,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
  min-height: 42px;
  background: #ffffff !important;
  color: var(--tox-ink) !important;
  caret-color: var(--tox-accent);
  -webkit-text-fill-color: var(--tox-ink) !important;
}

input::placeholder,
textarea::placeholder {
  color: #7b8d92 !important;
  opacity: 1 !important;
}

.stTextArea textarea {
  min-height: 92px;
  line-height: 1.5;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within,
.stTextArea textarea:focus {
  border-color: var(--tox-accent) !important;
  box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.11) !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stTextArea label,
.stSegmentedControl label,
.stRadio label,
.stCheckbox label {
  color: #294a52 !important;
  font-weight: 650 !important;
}

.stRadio label p,
.stRadio label span,
.stCheckbox label p,
.stCheckbox label span {
  color: var(--tox-ink) !important;
}

.stNumberInput button {
  border-color: var(--tox-line) !important;
  background: #eef4f3 !important;
  color: #31565d !important;
}

.stNumberInput button:hover {
  background: var(--tox-accent-soft) !important;
  color: var(--tox-accent) !important;
}

.stNumberInput button svg,
.stNumberInput button span {
  color: currentColor !important;
  fill: currentColor !important;
}

button[data-testid="stBaseButton-segmented_control"] {
  border-color: var(--tox-line-strong) !important;
  background: #ffffff !important;
  color: var(--tox-ink) !important;
}

button[data-testid="stBaseButton-segmented_controlActive"] {
  border-color: var(--tox-accent) !important;
  background: var(--tox-accent-soft) !important;
  color: var(--tox-accent) !important;
}

/* Buttons */
.stButton > button {
  min-height: 42px;
  border-radius: 12px;
  font-weight: 650;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  border-color: var(--tox-accent) !important;
  background: var(--tox-accent) !important;
  color: #ffffff !important;
  box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18);
}

.stButton > button[kind="primary"] *,
.stButton > button[data-testid="stBaseButton-primary"] * {
  color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  border-color: var(--tox-accent-hover) !important;
  background: var(--tox-accent-hover) !important;
  transform: translateY(-1px);
}

.stButton > button:disabled {
  border-color: #d5dfde !important;
  background: #e9efee !important;
  color: #687b80 !important;
  opacity: 1 !important;
  box-shadow: none !important;
}

/* Alerts use quiet tints instead of fluorescent framework defaults. */
div[data-testid="stAlert"] {
  border: 1px solid var(--tox-line) !important;
  border-radius: 12px !important;
  color: var(--tox-ink) !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
  color: inherit !important;
}

div[data-testid="stAlert"]:has([data-testid="stNotificationContentSuccess"]) {
  border-color: #c8e3d6 !important;
  background: var(--tox-success) !important;
  color: var(--tox-success-ink) !important;
}

div[data-testid="stAlert"]:has([data-testid="stNotificationContentInfo"]) {
  border-color: #cbdfe5 !important;
  background: var(--tox-info) !important;
  color: var(--tox-info-ink) !important;
}

/* Tables and tabs */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  overflow: hidden;
  border: 1px solid var(--tox-line);
  border-radius: 14px;
  background: #ffffff;
}

button[data-baseweb="tab"] {
  color: var(--tox-muted) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--tox-accent) !important;
}

@media (max-width: 760px) {
  .block-container { padding: 0.8rem 0.85rem 3rem; }
  .tox-hero { padding: 1rem; border-radius: 18px; }
  .tox-brand { font-size: 1.7rem; }
  .tox-subtitle { font-size: 0.94rem; }
  div[data-testid="stHorizontalBlock"] { gap: 0.55rem; }
}
"""
