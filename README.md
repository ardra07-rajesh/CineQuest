# CineQuest | The Ultimate Movie Universe

CineQuest is a gamified movie discovery, analytics and entertainment platform built with Python and Streamlit. The interface uses a cinematic dark-glass design system, grouped navigation, responsive poster cards and session-persistent navigation.

## Features

- Cinematic 3D-inspired interface with layered glass panels, perspective cards and animated gradients
- Movie Explorer with search, language, genre and year filters
- CineMystery clue-based movie guessing game
- Movie Arena head-to-head comparisons
- CineGraph analytics with interactive Plotly charts
- CinePredict machine-learning estimates for rating and revenue
- CineVault watchlist, favourites, watched list and personal ratings
- CinePersonality quiz with tailored recommendations
- Quiz Arena, XP, levels, badges and missions
- CSV upload support for your own movie dataset
- Optional TMDB API configuration for future live-data integration

## Run locally

### 1. Extract the ZIP and open the folder in VS Code

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install packages

```powershell
pip install -r requirements.txt
```

### 4. Start CineQuest

```powershell
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## Add your own dataset

Open **Data Studio** in the sidebar and upload a CSV. CineQuest recognises common column names such as:

- `title` or `original_title`
- `original_language` or `language`
- `release_date` or `release_year`
- `vote_average` or `rating`
- `revenue`, `budget`, `runtime`, `genres`, `overview`

The app normalises common column names and keeps the uploaded data in the current session. To make your dataset permanent, replace `data/movies.csv` with your prepared CSV.

## Included sample dataset

`data/movies.csv` contains a small, curated demonstration dataset across Hindi, Malayalam, Tamil, English and Korean cinema. It is included so the project runs immediately. It is not an exhaustive movie catalogue. The included CSV currently contains 35 curated records; use the TMDB collector or Data Studio upload to expand it.

## Optional TMDB data

Create a `.env` file based on `.env.example` and add your own TMDB Read Access Token. Keep `.env` private and never commit it to GitHub. Review TMDB's current API terms and attribution requirements before distributing a build that uses TMDB data.

TMDB developer documentation: https://developer.themoviedb.org/docs/getting-started

## Project structure

```text
CineQuest/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── movies.csv
├── assets/
└── modules/
```

## Notes

- CinePredict is a classroom demonstration model. Its estimates depend on the available dataset and should not be treated as reliable commercial forecasts.
- User progress is stored in Streamlit session state. It resets when the session is cleared or the app restarts. Add SQLite persistence as a future extension if permanent multi-user accounts are required.
- The 3D aesthetic is implemented with CSS transforms and layered styling. It does not require a separate 3D rendering engine.


## Collect 1,000+ movies

The included demo CSV is small. To build a larger multilingual catalogue, use the TMDB collector.

1. Create a TMDB account and obtain a Read Access Token from the TMDB API settings.
2. In the project root, copy `.env.example` to `.env`.
3. Put your token after `TMDB_READ_ACCESS_TOKEN=`.
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python scripts/collect_movies.py`

The collector targets up to 250 movies per original language for Malayalam, Hindi, Tamil, English and Korean. It creates one master CSV and five language-specific CSV files in `data/`. The final total depends on TMDB's available results for each language.

The collector uses TMDB's Discover endpoint. It provides titles, original language, release date, overview, genres, popularity, vote average, vote count and poster paths. Budget, revenue, runtime and director are not returned by Discover and are set to blank/zero in this first collection pass. They require additional movie-details and credits requests.

After collection, open CineQuest, go to **Data Studio**, upload `data/cinequest_master_movies.csv`, and the active catalogue will load for the current app session.

Keep `.env` private. Never upload your API token to GitHub. TMDB attribution and API terms apply.


## Premium navigation and selection redesign

The interface now uses grouped custom sidebar button navigation with an internal collapse/expand control, active-page styling, and a global movie-search header. CinePersonality presents the original three questions as stateful two-column choice cards, shows completion progress, and reveals the existing personality mapping and recommendation logic in a dedicated result panel. Movie Arena predictions and Quiz Arena answers use the reusable stateful choice-card helper. Existing catalogue data, model logic, and session-backed features are retained.
