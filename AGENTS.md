# AGENTS.md

## Project

YouTube Comment Language Lab — a multilingual (English/Hindi/Bengali/Code-Mixed)
application for collecting, preprocessing and language-classifying YouTube
comments. The scope covers the first workflow phases only: data collection / CSV
upload, cleaning & normalisation, NLP preprocessing, and language
detection/classification (plus statistics, logs, basic charts and cleaned-dataset
export). Core package: `youtube_sentiment/`; dashboard: `app.py`; CLI:
`youtube_sentiment/cli.py`. See `README.md` for architecture and features.

## Development environment and run notes

- Python deps are installed into a project virtualenv at `.venv` (system pip is
  PEP-668 "externally managed", so always use the venv). Activate with
  `. .venv/bin/activate` (or call binaries directly as `.venv/bin/python` /
  `.venv/bin/pytest`) before running anything.
- The startup update script creates `.venv`, installs `requirements.txt`, and
  downloads required NLTK corpora. Creating the venv relies on the system
  `python3.12-venv` apt package (already present in the VM snapshot); if venv
  creation ever fails with an `ensurepip` error, reinstall it with
  `sudo apt-get install -y python3.12-venv`.
- NLTK data (`stopwords`, `wordnet`, `omw-1.4`) is fetched on first use via
  `youtube_sentiment.preprocessing.ensure_nltk_data()`. It needs network access
  the first time; afterwards it is cached under `~/nltk_data`.
- Run commands (all from repo root, inside the venv):
  - Tests: `python -m pytest`
  - Lint: `flake8 youtube_sentiment app.py data/generate_sample_dataset.py tests`
  - CLI pipeline: `python -m youtube_sentiment.cli process --csv data/sample_comments.csv --output-dir outputs --export outputs/cleaned.csv`
  - Dashboard: `streamlit run app.py` (use `--server.headless true --server.port 8501` on the VM).
- `YOUTUBE_API_KEY` is only required for the live data-collection path (CLI
  `collect` / the dashboard's "Live YouTube" mode). Cleaning, NLP preprocessing,
  language detection/classification, statistics, charts and export are fully
  runnable and testable offline using `data/sample_comments.csv`.
- The sample dataset is regenerated with `python data/generate_sample_dataset.py`.
- Matplotlib uses the non-interactive `Agg` backend (set in
  `youtube_sentiment/visualization.py`) so charts render headless. Charts accept a
  `theme` argument (`"light"`/`"dark"`) to match the dashboard appearance.
