from pathlib import Path
import random
import re
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "movies.csv"

st.set_page_config(page_title="CineQuest | Movie Universe", page_icon="C", layout="wide", initial_sidebar_state="expanded")

LANGUAGES = {"hi": "Hindi", "ml": "Malayalam", "ta": "Tamil", "en": "English", "ko": "Korean"}
LANG_EMOJI = {"hi": "IN", "ml": "ML", "ta": "TA", "en": "EN", "ko": "KR"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-deep: #030408;
    --bg-surface: #0a0d16;
    --panel: #0e1322;
    --panel-hover: #141c30;
    --line: rgba(139, 92, 246, 0.18);
    --line-glow: rgba(34, 211, 238, 0.35);
    
    --text-primary: #ffffff;
    --text-secondary: #c4b5fd;
    --text-muted: #cbd5e1;
    
    --accent-violet: #8b5cf6;
    --accent-cyan: #22d3ee;
    --accent-lavender: #a78bfa;
    --accent-gold: #f5c66b;
    
    --gradient-hero: radial-gradient(circle at 82% 25%, rgba(34, 211, 238, 0.15), transparent 30%),
                     radial-gradient(circle at 10% 80%, rgba(139, 92, 246, 0.22), transparent 40%),
                     linear-gradient(135deg, #090d18 0%, #05070d 100%);
    --gradient-card: linear-gradient(145deg, rgba(14, 19, 34, 0.95), rgba(7, 10, 18, 0.98));
}

* {
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(ellipse at 15% -10%, rgba(139, 92, 246, 0.25), transparent 40%),
                radial-gradient(ellipse at 90% 10%, rgba(34, 211, 238, 0.15), transparent 35%),
                linear-gradient(145deg, #030408 0%, #060913 50%, #030408 100%);
    color: var(--text-primary);
}

[data-testid="stHeader"] {
    background: rgba(3, 4, 8, 0.75);
    backdrop-filter: blur(18px);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080b14 0%, #030408 100%);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"]>div:first-child {
    padding-top: 1rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary);
}

[data-testid="stSidebar"] div.stButton>button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    border: 1px solid transparent;
    background: #ffffff !important;
    color: #000000 !important;
    border-radius: 11px;
    padding: .58rem .72rem;
    font-weight: 600;
    box-shadow: none;
    transition: all .25s ease;
}

[data-testid="stSidebar"] div.stButton>button p,
[data-testid="stSidebar"] div.stButton>button span,
[data-testid="stSidebar"] div.stButton>button div,
[data-testid="stSidebar"] div.stButton>button * {
    color: #000000 !important;
}

[data-testid="stSidebar"] div.stButton>button:hover {
    background: #f0f2f6 !important;
    border-color: rgba(139, 92, 246, 0.3);
    color: #000000 !important;
    transform: translateX(3px);
    box-shadow: 0 0 22px rgba(139, 92, 246, 0.12);
}

[data-testid="stSidebar"] div.stButton>button[kind="primary"] {
    background: #ffffff !important;
    border-left: 3px solid var(--black);
    border-top: 1px solid rgba(167, 139, 250, 0.4);
    border-right: 1px solid rgba(34, 211, 238, 0.2);
    border-bottom: 1px solid rgba(167, 139, 250, 0.4);
    color: #000000 !important;
    box-shadow: inset 0 0 22px rgba(139, 92, 246, 0.15), 0 5px 18px rgba(0, 0, 0, 0.25);
}

[data-testid="stSidebar"] div.stButton>button[kind="primary"] p,
[data-testid="stSidebar"] div.stButton>button[kind="primary"] span,
[data-testid="stSidebar"] div.stButton>button[kind="primary"] div,
[data-testid="stSidebar"] div.stButton>button[kind="primary"] * {
    color: #000000 !important;
}
.nav-group {
    font: 600 .65rem 'DM Mono';
    letter-spacing: .2em;
    color: #000000 !important;
    margin: 20px 0 6px 8px;
    text-transform: uppercase;
}

.assessment-panel {
    padding: 18px 20px;
    border: 1px solid rgba(167, 139, 250, 0.25);
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(14, 19, 34, 0.95), rgba(7, 10, 18, 0.98));
    margin: 10px 0 18px;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
}

.assessment-progress {
    height: 7px;
    border-radius: 99px;
    background: #0f172a;
    overflow: hidden;
    margin: 9px 0 5px;
    border: 1px solid rgba(139, 92, 246, 0.2);
}

.assessment-progress>span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    box-shadow: 0 0 15px rgba(34, 211, 238, 0.5);
}

.result-card {
    padding: 26px;
    border: 1px solid rgba(139, 92, 246, 0.45);
    border-radius: 24px;
    background: radial-gradient(circle at 95% 0, rgba(34, 211, 238, 0.15), transparent 30%),
                linear-gradient(130deg, rgba(43, 31, 75, 0.92), rgba(10, 15, 27, 0.98));
    box-shadow: 0 20px 55px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.result-title {
    font: 700 clamp(1.8rem, 3vw, 2.5rem) 'Space Grotesk';
    color: var(--text-primary);
    margin: 8px 0;
}

div.stButton>button[kind="primary"] {
    background: #ffffff !important;
    border: 1px solid rgba(167, 139, 250, 0.5);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
    color: #000000 !important;
}

div.stButton>button[kind="primary"] p {
    color: #000000 !important;
}

div.stButton>button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(34, 211, 238, 0.4), 0 8px 25px rgba(139, 92, 246, 0.4);
    border-color: var(--accent-cyan);
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1540px;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.035em;
    color: var(--text-primary);
}

p, li, label {
    color: var(--text-muted);
}

small, .muted {
    color: var(--text-muted);
}

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 4px 16px;
}

.brand-mark {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    background: linear-gradient(145deg, var(--accent-violet), var(--accent-cyan));
    box-shadow: 0 8px 28px rgba(139, 92, 246, 0.4);
    font-size: 21px;
    color: #ffffff;
    font-weight: 700;
}

.brand-name {
    font: 700 1.12rem 'Space Grotesk';
    letter-spacing: .08em;
    color: var(--text-primary);
}

.brand-sub {
    font: .62rem 'DM Mono';
    letter-spacing: .13em;
    color: var(--accent-lavender);
    margin-top: 3px;
}

.sidebar-foot {
    padding: 15px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(14, 19, 34, 0.85), rgba(6, 9, 16, 0.85));
    margin-top: 15px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 18px;
    margin: 0 0 22px;
    border: 1px solid var(--line);
    border-radius: 16px;
    background: rgba(10, 13, 22, 0.85);
    backdrop-filter: blur(18px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
}

.topbar-kicker {
    font: .65rem 'DM Mono';
    letter-spacing: .16em;
    color: var(--accent-cyan);
    text-transform: uppercase;
}

.topbar-title {
    font: 600 1.05rem 'Space Grotesk';
    color: var(--text-primary);
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 28px;
    padding: clamp(25px, 4vw, 48px);
    background: var(--gradient-hero);
    box-shadow: 0 30px 90px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transform: perspective(1400px) rotateX(1deg);
}

.hero:before {
    content: "";
    position: absolute;
    width: 430px;
    height: 430px;
    right: -90px;
    top: -250px;
    border-radius: 50%;
    border: 1px solid rgba(34, 211, 238, 0.25);
    box-shadow: 0 0 70px rgba(34, 211, 238, 0.15), inset 0 0 65px rgba(139, 92, 246, 0.18);
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: 20%;
    bottom: -230px;
    border-radius: 50%;
    background: rgba(139, 92, 246, 0.22);
    filter: blur(34px);
}

.eyebrow, .section-kicker {
    font: 500 .68rem 'DM Mono';
    text-transform: uppercase;
    letter-spacing: .2em;
    color: var(--accent-cyan);
}

.hero-title {
    font: 700 clamp(2.5rem, 5vw, 4.7rem)/.98 'Space Grotesk';
    letter-spacing: -.065em;
    margin: 16px 0;
    background: linear-gradient(100deg, #ffffff 10%, #c7f7ff 52%, #c4b5fd 90%);
    -webkit-background-clip: text;
    color: transparent;
}

.hero-sub {
    max-width: 720px;
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.8;
}

.glass {
    position: relative;
    overflow: hidden;
    background: var(--gradient-card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 21px;
    box-shadow: 0 18px 48px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.07);
    transition: transform .28s ease, border-color .28s ease, box-shadow .28s ease;
}

.glass:hover {
    transform: translateY(-4px);
    border-color: var(--line-glow);
    box-shadow: 0 25px 58px rgba(0, 0, 0, 0.45), 0 0 28px rgba(34, 211, 238, 0.1);
}

.metric {
    font: 700 2rem 'Space Grotesk';
    color: var(--text-primary);
    margin-top: 7px;
}

.metric-label {
    font: 500 .7rem 'DM Mono';
    color: var(--accent-lavender);
    text-transform: uppercase;
    letter-spacing: .13em;
}

.metric-foot {
    font-size: .76rem;
    color: var(--accent-cyan);
    margin-top: 6px;
}

.page-title {
    font: 700 clamp(1.9rem, 3vw, 2.45rem) 'Space Grotesk';
    margin: 0 0 4px;
    color: var(--text-primary);
}

.page-desc {
    color: var(--text-secondary);
    margin-bottom: 24px;
}

.movie-card {
    height: 100%;
    padding: 14px;
    border: 1px solid rgba(167, 139, 250, 0.2);
    border-radius: 20px;
    background: var(--gradient-card);
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.35);
    transition: transform .3s ease, border-color .3s ease, box-shadow .3s ease;
    transform: perspective(1100px) rotateY(-1deg);
    position: relative;
    overflow: hidden;
}

.movie-card:hover {
    transform: perspective(1100px) rotateY(0deg) translateY(-7px) scale(1.012);
    border-color: var(--accent-cyan);
    box-shadow: 0 27px 55px rgba(0, 0, 0, 0.5), 0 0 30px rgba(139, 92, 246, 0.2);
}

.poster {
    height: 185px;
    border-radius: 14px;
    margin-bottom: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 14px;
    font: 700 2.4rem 'Space Grotesk';
    letter-spacing: -.05em;
    color: rgba(255, 255, 255, 0.95);
    background: radial-gradient(circle at 25% 15%, rgba(34, 211, 238, 0.4), transparent 35%),
                linear-gradient(135deg, #1b253f, #281a4c 58%, #080c16);
    border: 1px solid rgba(255, 255, 255, 0.12);
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
    background-size: cover;
    background-position: center;
}

.poster.has-image {
    background-color: #080c16;
}

.movie-title {
    font: 700 1.02rem 'Space Grotesk';
    color: var(--text-primary);
    min-height: 2.4em;
    line-height: 1.2;
}

.movie-meta {
    font: .68rem 'DM Mono';
    color: var(--accent-lavender);
    margin: 8px 0;
}

.pill {
    display: inline-block;
    padding: 4px 9px;
    border: 1px solid rgba(34, 211, 238, 0.3);
    background: rgba(34, 211, 238, 0.1);
    border-radius: 99px;
    color: var(--accent-cyan);
    font-size: .66rem;
    margin: 3px 4px 0 0;
    font-family: 'DM Mono', monospace;
}

.rank {
    font: .78rem 'DM Mono';
    color: var(--accent-gold);
}

div.stButton>button, div.stDownloadButton>button {
    border-radius: 11px;
    border: 1px solid rgba(139, 92, 246, 0.4);
    background: #ffffff !important;
    color: #000000 !important;
    font-weight: 700;
    transition: all .25s ease;
}

div.stButton>button p, 
div.stButton>button span, 
div.stButton>button div, 
div.stButton>button * {
    color: #000000 !important;
}

div.stButton>button:hover, 
div.stDownloadButton>button:hover {
    border-color: var(--accent-cyan);
    color: #000000 !important;
    box-shadow: 0 0 25px rgba(34, 211, 238, 0.25);
    transform: translateY(-2px);
}

div.stButton>button p, 
div.stButton>button span, 
div.stButton>button div,
div.stButton>button * {
    color: #000000 !important;
}

div.stButton>button:hover, 
div.stDownloadButton>button:hover {
    border-color: var(--accent-cyan);
    color: #000000 !important;
    box-shadow: 0 0 25px rgba(34, 211, 238, 0.25);
    transform: translateY(-2px);
}

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
    background: #080b14 !important;
    color: #ffffff !important;
    border-color: rgba(139, 92, 246, 0.3) !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: rgba(255, 255, 255, 0.5) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(10, 13, 22, 0.85);
    padding: 7px;
    border-radius: 14px;
    border: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #000000 !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
}

.stTabs [aria-selected="true"] p {
    color: #000000 !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    background: var(--panel);
}

hr {
    border-color: rgba(139, 92, 246, 0.2);
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(14, 19, 34, 0.9), rgba(6, 9, 16, 0.95));
    border: 1px solid var(--line);
    padding: 16px;
    border-radius: 17px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
}

div[data-testid="stMetricLabel"] {
    color: var(--accent-lavender);
}

div[data-testid="stMetricValue"] {
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(34, 211, 238, 0.3);
    background: rgba(10, 13, 22, 0.9);
}

[data-testid="stForm"] {
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(10, 13, 22, 0.75);
    padding: 20px;
}

@media(max-width:900px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .topbar { flex-wrap: wrap; }
    .poster { height: 165px; }
    .assessment-panel { padding: 15px; }
}

@media(max-width:600px) {
    [data-testid="stSidebar"] { min-width: 0; }
    .result-card { padding: 20px; }
    .hero { padding: 24px 20px; border-radius: 20px; }
    .hero-title { font-size: 2.55rem; }
    .movie-card { padding: 9px; }
    .poster { height: 145px; }
    .metric { font-size: 1.65rem; }
}

@media(prefers-reduced-motion:reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: .01ms !important;
    }
}

footer {
    visibility: hidden;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

def init_state():
    defaults = {
        "xp": 120, "watchlist": [], "watched": [], "favourites": [],
        "personal_ratings": {}, "mystery_score": 0, "arena_wins": 0,
        "quiz_score": 0, "mystery_current": None, "mystery_revealed": False,
        "personality": None, "daily_claimed": False, "history": [],
        "active_page": "Command Centre", "sidebar_compact": False,
        "selected_movie": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

def normalise(df):
    df = df.copy()
    aliases = {
        "original_language": "language", "language_code": "language",
        "vote_average": "rating", "imdb_rating": "rating",
        "release_year": "year", "release_date": "date",
        "movie_name": "title", "original_title": "title",
        "vote_count": "votes", "overview": "overview",
        "runtime": "runtime", "genres": "genres", "budget": "budget",
        "revenue": "revenue", "director": "director"
    }
    for src, dst in aliases.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = df[src]
    if "title" not in df.columns:
        df["title"] = "Untitled Movie"
    if "language" not in df.columns:
        df["language"] = "en"
    if "year" not in df.columns:
        if "date" in df.columns:
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        else:
            df["year"] = 2000
    if "rating" not in df.columns:
        df["rating"] = 0
    for col, default in [("genres", "Unknown"), ("overview", "No synopsis available."), ("budget", 0), ("revenue", 0), ("runtime", 0), ("votes", 0), ("popularity", 0), ("director", "Not listed")]:
        if col not in df.columns:
            df[col] = default
    for c in ["year", "rating", "budget", "revenue", "runtime", "votes", "popularity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["language"] = df["language"].astype(str).str.lower().str[:2]
    df["genres"] = df["genres"].fillna("Unknown").astype(str)
    df["title"] = df["title"].fillna("Untitled Movie").astype(str)
    df = df.drop_duplicates(subset=["title", "year", "language"])
    return df.reset_index(drop=True)

@st.cache_data
def load_data():
    if DATA_FILE.exists():
        df_raw = pd.read_csv(DATA_FILE)
    else:
        np.random.seed(42)
        n = 150
        titles = [f"Movie Title {i}" for i in range(1, n + 1)]
        years = np.random.randint(2010, 2026, size=n)
        ratings = np.round(np.random.uniform(5.0, 9.5, size=n), 1)
        votes = np.random.randint(1000, 500000, size=n)
        languages = np.random.choice(list(LANGUAGES.keys()), size=n)
        genres = np.random.choice(["Action, Thriller", "Drama, Romance", "Comedy, Drama", "Sci-Fi, Adventure", "Crime, Mystery"], size=n)
        df_raw = pd.DataFrame({
            "title": titles,
            "year": years,
            "rating": ratings,
            "votes": votes,
            "language": languages,
            "genres": genres
        })
    return normalise(df_raw)

df = load_data()
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if st.session_state.active_df is not None:
    df = st.session_state.active_df

def money(v):
    try:
        v = float(v)
        if v >= 1e9: return f"₹{v/1e9:.2f}B"
        if v >= 1e7: return f"₹{v/1e7:.1f}Cr"
        if v >= 1e5: return f"₹{v/1e5:.1f}L"
        return f"₹{v:,.0f}"
    except:
        return "N/A"

def genre_list(data):
    vals = set()
    for item in data["genres"].dropna():
        vals.update([x.strip() for x in re.split(r"[|,;/]", str(item)) if x.strip()])
    return sorted(vals)

def page_header(kicker, title, desc):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="page-title">{title}</div><div class="page-desc">{desc}</div>', unsafe_allow_html=True)

def movie_card(row, key):
    title = str(row["title"])
    lang = LANGUAGES.get(str(row["language"]), str(row["language"]).upper())
    genres = str(row["genres"]).replace("|", " · ")
    poster = ""
    for field in ("poster_url", "poster_path", "poster"):
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            candidate = str(row[field]).strip()
            if candidate.startswith("/"):
                poster = "https://image.tmdb.org/t/p/w500" + candidate
            elif candidate.startswith(("https://", "http://")):
                poster = candidate
            break
    safe_title = html.escape(title)
    poster_style = f' style="background-image:linear-gradient(180deg,rgba(5,8,18,.04),rgba(5,8,18,.58)),url(&quot;{html.escape(poster, quote=True)}&quot;)"' if poster else ""
    poster_class = "poster has-image" if poster else "poster"
    st.markdown(f"""
    <div class="movie-card">
      <div class="{poster_class}"{poster_style}>{"" if poster else safe_title[:1].upper()}</div>
      <div class="movie-title">{safe_title}</div>
      <div class="movie-meta">{int(row['year']) if row['year'] else 'Year N/A'} · {html.escape(lang)} · <span class="rank">★ {row['rating']:.1f}</span></div>
      <div>{''.join(f'<span class="pill">{html.escape(g.strip())}</span>' for g in genres.split('·')[:3])}</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Open movie", key=f"open_{key}", use_container_width=True):
        st.session_state.selected_movie = title
        st.session_state.history.append(title)
        st.session_state.xp += 2

def movie_grid(data, prefix="grid", n=12):
    if data.empty:
        st.info("No movies match these filters. Try widening your search.")
        return
    shown = data.head(n)
    cols = st.columns(4)
    for i, (_, row) in enumerate(shown.iterrows()):
        with cols[i % 4]:
            movie_card(row, f"{prefix}_{i}_{str(row['title'])[:12]}")

def metric_card(label, value, foot, icon="—"):
    st.markdown(f'<div class="glass"><div class="metric-label">{icon} &nbsp; {label}</div><div class="metric">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)

def selected_detail(data):
    selected = st.session_state.get("selected_movie")
    if not selected:
        return
    match = data[data.title == selected]
    if match.empty:
        return
    r = match.iloc[0]
    with st.expander(f"Selected movie · {selected}", expanded=True):
        a, b = st.columns([2, 3])
        with a:
            st.markdown(f'<div class="poster" style="height:230px;font-size:4rem">{str(r.title)[:1]}</div>', unsafe_allow_html=True)
        with b:
            st.subheader(r.title)
            st.caption(f"{int(r.year)} · {LANGUAGES.get(r.language, r.language.upper())} · {r.genres}")
            st.write(r.overview)
            st.write(f"**Director:** {r.director}")
            st.write(f"**Rating:** {r.rating:.1f}/10  |  **Runtime:** {int(r.runtime)} min")
            st.write(f"**Budget:** {money(r.budget)}  |  **Revenue:** {money(r.revenue)}")
            c1, c2, c3 = st.columns(3)
            if c1.button("＋ Watchlist", key="detail_watch"):
                if selected not in st.session_state.watchlist:
                    st.session_state.watchlist.append(selected)
                st.session_state.xp += 5
                st.success("Added to CineVault.")
            if c2.button("✓ Watched", key="detail_watched"):
                if selected not in st.session_state.watched:
                    st.session_state.watched.append(selected)
                st.session_state.xp += 5
                st.success("Marked as watched.")
            if c3.button("♥ Favourite", key="detail_fav"):
                if selected not in st.session_state.favourites:
                    st.session_state.favourites.append(selected)
                st.session_state.xp += 5
                st.success("Added to favourites.")

def selection_cards(prompt, options, key, icons=None, descriptions=None, columns=2):
    icons = icons or {}
    descriptions = descriptions or {}
    if key not in st.session_state:
        st.session_state[key] = None
    st.markdown(f'<div class="selection-heading">{html.escape(prompt)}</div>', unsafe_allow_html=True)
    cols = st.columns(columns)
    for i, option in enumerate(options):
        chosen = st.session_state.get(key) == option
        icon = icons.get(option, "")
        with cols[i % columns]:
            label = f"{'✓ ' if chosen else ''}{icon}  {option}"
            if descriptions.get(option):
                st.caption(descriptions[option])
            if st.button(label, key=f"{key}_choice_{i}", type="primary" if chosen else "secondary",
                         use_container_width=True, help=f"Selected: {option}" if chosen else f"Choose {option}"):
                st.session_state[key] = option
                if key.startswith("personality_"):
                    st.session_state.personality_revealed = False
    return st.session_state.get(key)

NAV_GROUPS = [
    ("OVERVIEW", ["Command Centre"]),
    ("DISCOVER", ["Movie Explorer", "CineMystery"]),
    ("ENTERTAINMENT", ["Movie Arena", "Quiz Arena", "CinePersonality"]),
    ("INTELLIGENCE", ["CineGraph", "CinePredict"]),
    ("PERSONAL SPACE", ["CineVault"]),
    ("MANAGEMENT", ["Data Studio"]),
]
NAV_ICONS = {"Command Centre": "", "Movie Explorer": "", "CineMystery": "", "Movie Arena": "", "Quiz Arena": "", "CinePersonality": "", "CineGraph": "", "CinePredict": "", "CineVault": "", "Data Studio": ""}

def select_nav(page_name):
    st.session_state.active_page = page_name

with st.sidebar:
    compact = st.session_state.sidebar_compact
    if compact:
        st.markdown('<div class="brand-lockup" style="justify-content:center"><div class="brand-mark">C</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand-lockup"><div class="brand-mark">C</div><div><div class="brand-name">CINEQUEST</div><div class="brand-sub">THE ULTIMATE MOVIE UNIVERSE</div></div></div>', unsafe_allow_html=True)
    if st.button("››" if compact else "‹‹  Collapse navigation", key="toggle_sidebar", use_container_width=True,
                 help="Expand navigation" if compact else "Collapse navigation"):
        st.session_state.sidebar_compact = not compact
        st.rerun()
    st.markdown("---")
    for group_name, group_pages in NAV_GROUPS:
        if not compact:
            st.markdown(f'<div class="nav-group">{group_name}</div>', unsafe_allow_html=True)
        for nav_page in group_pages:
            icon = NAV_ICONS.get(nav_page, "")
            label = icon if compact else f"{icon}{nav_page}"
            if st.button(label, key=f"nav_{nav_page.replace(' ','_')}",
                         type="primary" if st.session_state.active_page == nav_page else "secondary",
                         use_container_width=True, help=nav_page):
                select_nav(nav_page)
                st.rerun()
    st.markdown("---")
    level = 1 + st.session_state.xp // 100
    progress = (st.session_state.xp % 100) / 100
    if not compact:
        st.markdown(f'<div class="sidebar-foot"><div class="eyebrow">PLAYER PROFILE</div><h3 style="margin:8px 0 2px">LEVEL {level}</h3><div style="font: .7rem DM Mono;color:#9caac2">{st.session_state.xp:,} XP EARNED</div></div>', unsafe_allow_html=True)
        st.progress(progress)
        st.caption(f"{100-(st.session_state.xp%100)} XP to next level")
        st.caption(f"CATALOGUE  ·  {len(df):,} TITLES")
        st.caption(f"{df.language.nunique()} LANGUAGE CODES INDEXED")
    else:
        st.caption(f"LV {level}")

page = st.session_state.active_page
level = 1 + st.session_state.xp // 100

head_left, head_mid, head_right = st.columns([1.35, 1.8, 1.0])
with head_left:
    st.markdown(f'<div style="padding:8px 0"><div class="topbar-kicker">CINEQUEST / WORKSPACE</div><div class="topbar-title">{html.escape(page)}</div></div>', unsafe_allow_html=True)
with head_mid:
    with st.form("global_movie_search", clear_on_submit=False):
        search_cols = st.columns([5, 1])
        global_query = search_cols[0].text_input("Global movie search", placeholder="Search the movie universe…", label_visibility="collapsed", key="global_search")
        submitted_search = search_cols[1].form_submit_button("⌕", use_container_width=True)
        if submitted_search and global_query.strip():
            st.session_state.explorer_query = global_query.strip()
            st.session_state.active_page = "Movie Explorer"
            st.rerun()
with head_right:
    st.markdown(f'<div style="text-align:right;padding:7px 0"><div class="topbar-kicker">{len(df):,} TITLES</div><div class="topbar-title">LV {level} · {st.session_state.xp:,} XP</div></div>', unsafe_allow_html=True)


if page == "Command Centre":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">CINEMATIC DISCOVERY SYSTEM · ONLINE</div>
      <div class="hero-title">ENTER THE<br> MOVIE UNIVERSE.</div>
      <div class="hero-sub">Discover stories, challenge your movie knowledge and explore cinema through an interactive intelligence dashboard. Your next favourite film is one click away.</div>
    </div>""", unsafe_allow_html=True)
    st.write("")
    total = len(df)
    avg = df.rating[df.rating > 0].mean() if (df.rating > 0).any() else 0
    genres = genre_list(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("MOVIES", f"{total:,}", "Across your current catalogue", "▣")
    with c2: metric_card("AVG RATING", f"{avg:.1f}/10", "Mean community score", "★")
    with c3: metric_card("GENRES", str(len(genres)), "Genres indexed", "◈")
    with c4: metric_card("YOUR XP", str(st.session_state.xp), f"Level {level} explorer", "⚡")
    st.write("")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown("### Featured discovery")
        featured = df.sort_values(["popularity", "rating"], ascending=False).iloc[0]
        st.markdown(f'<div class="glass"><div class="eyebrow">EDITORIAL PICK · {LANGUAGES.get(featured.language,"CINEMA").upper()}</div><h2>{featured.title}</h2><p>{featured.overview}</p><span class="pill">{featured.genres}</span><span class="pill">{int(featured.year)}</span><span class="pill">★ {featured.rating:.1f}</span></div>', unsafe_allow_html=True)
        if st.button("Explore featured title", key="featured"):
            st.session_state.selected_movie = featured.title
            st.session_state.xp += 3
    with right:
        st.markdown("### Daily mission")
        st.markdown('<div class="glass"><div class="eyebrow">QUEST 001</div><h3>Curiosity streak</h3><p>Explore three titles and add one movie to your watchlist.</p><div class="metric">+25 XP</div></div>', unsafe_allow_html=True)
        if st.button("Claim daily mission", disabled=st.session_state.daily_claimed):
            st.session_state.xp += 25
            st.session_state.daily_claimed = True
            st.success("Mission claimed. +25 XP")
    st.write("")
    st.markdown("### Trending in your catalogue")
    movie_grid(df.sort_values("popularity", ascending=False), "trend", 8)
    selected_detail(df)

elif page == "Movie Explorer":
    page_header("DISCOVERY ENGINE", "Movie Explorer", "Search the catalogue, tune your filters and open a title to inspect its details.")
    if "explorer_query" not in st.session_state:
        st.session_state.explorer_query = ""
    q = st.text_input("Search titles, plot keywords or directors", value=st.session_state.explorer_query, placeholder="Try: thriller, Nolan, Drishyam...", key="explorer_query_input")
    if q != st.session_state.explorer_query:
        st.session_state.explorer_query = q
    f1, f2, f3, f4 = st.columns(4)
    lang_options = ["All"] + list(LANGUAGES.values())
    language = f1.selectbox("Original language", lang_options)
    all_genres = genre_list(df)
    genre = f2.selectbox("Genre", "All genres" if not all_genres else ["All genres"] + all_genres)
    years = sorted([int(y) for y in df.year.unique() if y > 0])
    yr = f3.selectbox("Release year", ["All years"] + years[::-1])
    sort = f4.selectbox("Sort by", ["Popularity", "Rating", "Newest", "Revenue"])
    result = df.copy()
    if st.session_state.explorer_query:
        query_val = st.session_state.explorer_query
        mask = (result.title.str.contains(query_val, case=False, na=False) | result.overview.str.contains(query_val, case=False, na=False) | result.director.str.contains(query_val, case=False, na=False))
        result = result[mask]
    if language != "All":
        lang_rev_map = {v: k for k, v in LANGUAGES.items()}
        result = result[result.language == lang_rev_map.get(language, language)]
    if genre != "All genres":
        result = result[result.genres.str.contains(genre, case=False, na=False)]
    if yr != "All years":
        result = result[result.year == yr]
    sortcol = {"Popularity": "popularity", "Rating": "rating", "Newest": "year", "Revenue": "revenue"}[sort]
    result = result.sort_values(sortcol, ascending=False)
    st.caption(f"{len(result):,} matching titles")
    movie_grid(result, "explore", 24)
    selected_detail(df)

elif page == "CineMystery":
    page_header("INTERACTIVE DETECTIVE MODE", "CineMystery", "Decode the film using progressive clues. Fewer clues used means a higher score.")
    if st.session_state.mystery_current is None:
        if st.button("Start a new mystery", type="primary"):
            candidates = df[df.overview.notna() & (df.title.str.len() > 2)]
            st.session_state.mystery_current = random.choice(candidates.title.tolist())
            st.session_state.mystery_revealed = False
            st.rerun()
    else:
        target = df[df.title == st.session_state.mystery_current].iloc[0]
        clues = [f"This film was released in {int(target.year)}.", f"Its main genres include {target.genres}.", f"The original language is {LANGUAGES.get(target.language, target.language)}.", f"Director: {target.director}.", f"Plot: {target.overview}"]
        st.markdown('<div class="glass"><div class="eyebrow">CASE FILE CQ-07</div><h2>Identify the movie</h2><p>Each clue reveals more information. Submit your guess when ready.</p></div>', unsafe_allow_html=True)
        reveal = st.slider("Reveal clues", 1, 5, 1)
        for i in range(reveal):
            st.info(f"CLUE {i+1:02d}  ·  {clues[i]}")
        guess = st.text_input("Your movie guess", placeholder="Enter the movie title")
        if st.button("Submit answer", type="primary"):
            if guess.strip().lower() == target.title.lower():
                points = max(5, 30 - (reveal - 1) * 5)
                st.session_state.mystery_score += points
                st.session_state.xp += points
                st.success(f"Correct. You earned {points} XP.")
                st.session_state.mystery_revealed = True
            else:
                st.error("Not quite. Recheck the clues and try again.")
        if st.session_state.mystery_revealed:
            st.markdown(f"### Case solved: {target.title}")
            if st.button("Next mystery"):
                st.session_state.mystery_current = None
                st.session_state.mystery_revealed = False
                st.rerun()
        elif st.button("Reveal answer"):
            st.session_state.mystery_revealed = True
            st.warning(f"The answer is {target.title}")
    st.write("")
    st.metric("Mystery XP earned", st.session_state.mystery_score)

elif page == "Movie Arena":
    page_header("HEAD TO HEAD", "Movie Arena", "Pick a category, compare two films and predict which one leads.")
    if len(df) < 2:
        st.warning("Add at least two movies to use the arena.")
    else:
        titles = df.title.tolist()
        a = st.selectbox("Challenger A", titles, index=0, key="arena_a")
        b = st.selectbox("Challenger B", titles, index=min(1, len(titles) - 1), key="arena_b")
        metric = st.selectbox("Battle metric", ["Rating", "Revenue", "Popularity", "Runtime", "Release year"], key="arena_metric")
        colmap = {"Rating": "rating", "Revenue": "revenue", "Popularity": "popularity", "Runtime": "runtime", "Release year": "year"}
        ra = df[df.title == a].iloc[0]
        rb = df[df.title == b].iloc[0]
        x, y = st.columns(2)
        with x:
            st.markdown(f'<div class="glass"><div class="eyebrow">CHALLENGER A</div><h2>{a}</h2><div class="metric">{ra[colmap[metric]]:,.1f}</div><p>{ra.genres}</p><p>{LANGUAGES.get(ra.language, ra.language)}</p></div>', unsafe_allow_html=True)
        with y:
            st.markdown(f'<div class="glass"><div class="eyebrow">CHALLENGER B</div><h2>{b}</h2><div class="metric">{rb[colmap[metric]]:,.1f}</div><p>{rb.genres}</p><p>{LANGUAGES.get(rb.language, rb.language)}</p></div>', unsafe_allow_html=True)
        if a == b:
            st.info("Choose two different movies for a fair battle.")
        else:
            winner = a if ra[colmap[metric]] > rb[colmap[metric]] else b
            st.markdown(f"### Arena result: **{winner}** leads in {metric.lower()}.")
            pick = selection_cards("Make your prediction", [a, b], "arena_prediction", icons={a: "◈", b: "◈"}, columns=2)
            if st.button("Lock prediction", type="primary"):
                if pick == winner:
                    st.session_state.arena_wins += 1
                    st.session_state.xp += 15
                    st.success("Correct prediction. +15 XP")
                else:
                    st.info(f"The data leader is {winner}. Better luck in the next round.")
    st.metric("Arena wins", st.session_state.arena_wins)

elif page == "CineGraph":
    page_header("MOVIE INTELLIGENCE LAB", "CineGraph", "Explore catalogue patterns with interactive data visualisations.")
    chart_tabs = st.tabs(["Genre signals", "Rating map", "Release timeline", "Revenue lab", "Language mix"])
    with chart_tabs[0]:
        counts = {}
        for gs in df.genres:
            for g in re.split(r"[|,;/]", str(gs)):
                g = g.strip()
                if g:
                    counts[g] = counts.get(g, 0) + 1
        gd = pd.DataFrame({"Genre": list(counts), "Movie count": list(counts.values())}).sort_values("Movie count", ascending=False)
        fig = px.bar(gd, x="Movie count", y="Genre", orientation="h", color="Movie count", color_continuous_scale=["#27365c", "#5ce4f5", "#a17cff"], template="plotly_dark")
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with chart_tabs[1]:
        fig = px.scatter(df, x="votes", y="rating", size="popularity", color="language", hover_name="title", template="plotly_dark", labels={"votes": "Vote count", "rating": "Average rating", "language": "Original language"})
        fig.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with chart_tabs[2]:
        timeline = df[df.year > 0].groupby("year", as_index=False).size()
        fig = px.area(timeline, x="year", y="size", template="plotly_dark", color_discrete_sequence=["#54e6ff"], labels={"size": "Movie count", "year": "Release year"})
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with chart_tabs[3]:
        rev = df[(df.revenue > 0) & (df.budget > 0)].copy()
        if rev.empty:
            st.info("Revenue analysis needs budget and revenue fields in your dataset.")
        else:
            rev["ROI multiple"] = rev.revenue / rev.budget
            fig = px.scatter(rev, x="budget", y="revenue", color="rating", size="popularity", hover_name="title", template="plotly_dark", color_continuous_scale="Viridis")
            fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(rev[["title", "budget", "revenue", "ROI multiple"]].sort_values("ROI multiple", ascending=False), use_container_width=True, hide_index=True)
    with chart_tabs[4]:
        langs = df.language.map(lambda x: LANGUAGES.get(x, x.upper())).value_counts().reset_index()
        langs.columns = ["Language", "Movies"]
        fig = px.pie(langs, names="Language", values="Movies", hole=.62, template="plotly_dark", color_discrete_sequence=["#54e6ff", "#9d7bff", "#ff72c6", "#f4c878", "#62d7a5"])
        fig.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "CinePredict":
    page_header("PREDICTIVE INTELLIGENCE", "CinePredict", "Train a small regression model on your current catalogue and test a hypothetical movie profile.")
    train = df[(df.rating > 0) & (df.year > 0)].copy()
    if len(train) < 8:
        st.warning("CinePredict needs at least 8 movies with valid ratings and release years. Upload a larger dataset in Data Studio.")
    else:
        features = ["year", "runtime", "budget", "popularity", "votes"]
        X = train[features].fillna(0)
        y = train["rating"]
        model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=5)
        model.fit(X, y)
        pred = model.predict(X)
        c1, c2 = st.columns(2)
        with c1: st.metric("Training titles", len(train))
        with c2: st.metric("Training MAE", f"{mean_absolute_error(y, pred):.2f} rating points")
        st.caption("This is a demonstration model trained on the active catalogue. Small or biased datasets produce unstable estimates.")
        with st.form("predict_form"):
            p1, p2 = st.columns(2)
            year = p1.number_input("Release year", min_value=1900, max_value=2035, value=2026)
            runtime = p2.number_input("Runtime (minutes)", min_value=40, max_value=300, value=130)
            budget = p1.number_input("Budget (dataset currency)", min_value=0, value=20000000, step=1000000)
            popularity = p2.slider("Expected popularity index", 0.0, 150.0, 45.0)
            votes = st.number_input("Expected vote count", min_value=0, value=250, step=25)
            submitted = st.form_submit_button("Estimate rating", type="primary")
        if submitted:
            estimate = float(model.predict(pd.DataFrame([[year, runtime, budget, popularity, votes]], columns=features))[0])
            st.markdown(f'<div class="glass"><div class="eyebrow">MODEL ESTIMATE</div><div class="metric">{estimate:.2f} / 10</div><p>Estimated community rating based on the current catalogue.</p></div>', unsafe_allow_html=True)
        imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", template="plotly_dark", color="Importance", color_continuous_scale=["#27365c", "#54e6ff"])
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "CineVault":
    page_header("YOUR PERSONAL COLLECTION", "CineVault", "Keep track of what you want to watch, what you have watched and the films you love.")
    t1, t2, t3, t4 = st.tabs(["Watchlist", "Watched", "Favourites", "My ratings"])
    with t1:
        st.metric("Saved titles", len(st.session_state.watchlist))
        for title in list(st.session_state.watchlist):
            st.write("▸", title)
            if st.button("Remove", key=f"rmw_{title}"):
                st.session_state.watchlist.remove(title)
                st.rerun()
        if not st.session_state.watchlist:
            st.info("Your watchlist is empty. Explore movies and add a title.")
    with t2:
        st.metric("Completed titles", len(st.session_state.watched))
        for title in st.session_state.watched:
            st.write("✓", title)
        if not st.session_state.watched:
            st.info("Your watched collection will appear here.")
    with t3:
        for title in st.session_state.favourites:
            st.write("♥", title)
        if not st.session_state.favourites:
            st.info("Save favourite films from a movie detail panel.")
    with t4:
        title = st.selectbox("Choose a movie", df.title.tolist(), key="vault_rating_title")
        rating = st.slider("Your personal rating", 1, 10, 8, key="vault_rating_slider")
        if st.button("Save personal rating"):
            st.session_state.personal_ratings[title] = rating
            st.success("Your rating was saved for this session.")
        if st.session_state.personal_ratings:
            st.dataframe(pd.DataFrame([{"Movie": k, "Your rating": v} for k, v in st.session_state.personal_ratings.items()]), use_container_width=True, hide_index=True)

elif page == "CinePersonality":
    page_header("YOUR CINEMA DNA", "Find your cinematic identity", "Three quick choices. One personal movie-watching archetype.")
    questions = [
        ("Pick a movie mood", ["Mind-bending and mysterious", "Emotional and meaningful", "Fast, loud and thrilling", "Funny, light and feel-good"]),
        ("Choose a setting", ["Space, science and the unknown", "Real life and human relationships", "Crime, secrets and investigations", "Fantasy worlds and adventure"]),
        ("What keeps you watching?", ["Unexpected twists", "Strong characters", "Action and spectacle", "Clever humour"])
    ]
    choice_icons = {
        "Mind-bending and mysterious": "◎", "Emotional and meaningful": "♥",
        "Fast, loud and thrilling": "⚡", "Funny, light and feel-good": "☺",
        "Space, science and the unknown": "◎", "Real life and human relationships": "♥",
        "Crime, secrets and investigations": "⌕", "Fantasy worlds and adventure": "◈",
        "Unexpected twists": "⤨", "Strong characters": "★",
        "Action and spectacle": "⚡", "Clever humour": "☺"
    }
    answers = []
    for i, (q, opts) in enumerate(questions):
        done = st.session_state.get(f"personality_{i}") in opts
        st.markdown(f'<div class="assessment-panel"><div class="eyebrow">QUESTION {i+1:02d} / 03</div><h3 style="margin:7px 0 2px">{html.escape(q)}</h3><div style="color:#94a3b8;font-size:.85rem">{"Completed" if done else "Choose one option to continue your profile"}</div></div>', unsafe_allow_html=True)
        answer = selection_cards(q, opts, f"personality_{i}", icons=choice_icons, columns=2)
        answers.append(answer)
    completed = sum(1 for i, (_, opts) in enumerate(questions) if st.session_state.get(f"personality_{i}") in opts)
    pct = int(completed / len(questions) * 100)
    st.markdown(f'<div class="assessment-panel"><div style="display:flex;justify-content:space-between;gap:12px"><span class="eyebrow">ASSESSMENT PROGRESS</span><b>{completed} / 3 complete · {pct}%</b></div><div class="assessment-progress"><span style="width:{pct}%"></span></div></div>', unsafe_allow_html=True)
    if st.button("  REVEAL MY CINEMA DNA", type="primary", use_container_width=True, disabled=completed != 3):
        mapping = {
            "Mind-bending and mysterious": "The Puzzle Architect",
            "Emotional and meaningful": "The Heart Collector",
            "Fast, loud and thrilling": "The Adrenaline Seeker",
            "Funny, light and feel-good": "The Comfort Watcher"
        }
        result = mapping[answers[0]]
        st.session_state.personality = result
        st.session_state.xp += 20
        st.session_state.personality_revealed = True
    if st.session_state.get("personality_revealed") and st.session_state.personality:
        result = st.session_state.personality
        result_icons = {"The Puzzle Architect": "◎", "The Heart Collector": "♥", "The Adrenaline Seeker": "⚡", "The Comfort Watcher": "☺"}
        st.markdown(f'<div class="result-card"><div class="eyebrow">YOUR CINEMA DNA · PROFILE UNLOCKED</div><div style="font-size:2.5rem;margin:8px 0">{result_icons.get(result,"")}</div><div class="result-title">{html.escape(result)}</div><p style="color:#c4cee0">Your recommendations are weighted toward the mood you selected.</p><div class="pill">+20 XP EARNED</div></div>', unsafe_allow_html=True)
        genre_map = {"The Puzzle Architect": ["Thriller", "Sci-Fi", "Mystery"], "The Heart Collector": ["Drama", "Romance"], "The Adrenaline Seeker": ["Action", "Adventure"], "The Comfort Watcher": ["Comedy", "Family"]}
        matches = df[df.genres.apply(lambda x: any(g.lower() in str(x).lower() for g in genre_map[result]))].sort_values("rating", ascending=False)
        st.markdown("### Your recommended watchlist")
        movie_grid(matches, "personality", 8)
    elif st.session_state.personality:
        st.info(f"Your latest result: {st.session_state.personality}")

elif page == "Quiz Arena":
    page_header("KNOW YOUR CINEMA", "Quiz Arena", "Answer movie trivia and collect XP. Questions are generated from the active catalogue.")
    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = None
    if st.session_state.quiz_question is None:
        if st.button("Generate quiz question", type="primary"):
            r = df.sample(1).iloc[0]
            st.session_state.quiz_question = {"title": r.title, "answer": LANGUAGES.get(r.language, r.language), "prompt": f"What is the original language of **{r.title}**?", "options": random.sample(list(LANGUAGES.values()), k=min(5, len(LANGUAGES)))}
            st.session_state.quiz_selected = st.session_state.quiz_question["options"][0]
            st.rerun()
    else:
        q = st.session_state.quiz_question
        st.markdown(f'<div class="glass"><div class="eyebrow">QUESTION 01 · LANGUAGE DETECTIVE</div><h2>{q["prompt"]}</h2></div>', unsafe_allow_html=True)
        ans = selection_cards("Select your answer", q["options"], "quiz_selected", icons={option: "◈" for option in q["options"]}, columns=2)
        if st.button("Submit answer", type="primary"):
            if ans == q["answer"]:
                st.session_state.quiz_score += 1
                st.session_state.xp += 10
                st.success("Correct. +10 XP")
            else:
                st.error(f"Correct answer: {q['answer']}")
            st.session_state.quiz_question = None
            st.rerun()
    st.metric("Correct answers", st.session_state.quiz_score)

elif page == "Data Studio":
    page_header("DATA CONTROL ROOM", "Data Studio", "Upload your own CSV, inspect the active dataset and export a clean catalogue.")
    st.markdown('<div class="glass"><div class="eyebrow">DATA INGESTION</div><h3>Bring your movie universe</h3><p>Upload a CSV containing movie metadata. The app recognises common TMDB and Kaggle-style column names.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload movie CSV", type=["csv"])
    if uploaded is not None:
        try:
            newdf = normalise(pd.read_csv(uploaded))
            if len(newdf) == 0:
                st.error("The uploaded file contains no usable rows.")
            else:
                st.session_state.active_df = newdf
                st.success(f"Loaded {len(newdf):,} movie records into this session.")
                st.rerun()
        except Exception as e:
            st.error(f"Could not read this CSV: {e}")
    st.write("")
    a, b, c = st.columns(3)
    a.metric("Active records", f"{len(df):,}")
    b.metric("Language codes", df.language.nunique())
    c.metric("Missing overview", int(df.overview.isna().sum()))
    st.markdown("### Language distribution")
    langdist = df.language.map(lambda x: LANGUAGES.get(x, x.upper())).value_counts().rename_axis("Language").reset_index(name="Movies")
    st.dataframe(langdist, use_container_width=True, hide_index=True)
    st.markdown("### Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download active catalogue CSV", csv, "cinequest_active_movies.csv", "text/csv", type="primary")
    if st.button("Reset to included sample dataset"):
        st.session_state.active_df = None
        st.rerun()

st.markdown("---")
st.caption("CINEQUEST // MOVIE INTELLIGENCE SYSTEM · Built with Python, Streamlit, Pandas, Plotly and Scikit-learn")