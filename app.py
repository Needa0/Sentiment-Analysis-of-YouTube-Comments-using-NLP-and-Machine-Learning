"""Streamlit dashboard: multilingual YouTube comment preprocessing & language classification.

Features:
- Data sources: bundled sample dataset, CSV upload, or live YouTube Data API v3.
- Cleaning & normalisation, NLP preprocessing, language detection/classification.
- Language statistics, theme-aware charts, filterable comment table, process logs.
- Export of the cleaned, language-annotated dataset.
- Modern, responsive UI with light and dark modes.
"""

from __future__ import annotations

import html
import os
import tempfile

import pandas as pd
import streamlit as st

from youtube_sentiment import visualization as viz
from youtube_sentiment.config import SUPPORTED_LANGUAGES
from youtube_sentiment.data_collection import YouTubeCommentCollector
from youtube_sentiment.pipeline import PreprocessingPipeline

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_comments.csv")

st.set_page_config(
    page_title="Sentiment Analysis of YouTube Comments",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Theming
# --------------------------------------------------------------------------- #
THEMES = {
    "Light": {
        "bg": "#F7F9FC",
        "panel": "#FFFFFF",
        "text": "#111827",
        "muted": "#6B7280",
        "border": "#D6DEE8",
        "accent": "#2563EB",
        "accent_soft": "#E8F1FF",
        "shadow": "0 1px 3px rgba(16,24,40,0.08), 0 1px 2px rgba(16,24,40,0.06)",
        "input_bg": "#FFFFFF",
        "input_text": "#111827",
        "input_border": "#CBD5E1",
        "input_placeholder": "#94A3B8",
        "button_text": "#FFFFFF",
        "radio_bg": "#FFFFFF",
        "toggle_track": "#E5E7EB",
        "toggle_thumb": "#FFFFFF",
        "scrollbar": "#CBD5E1",
        "color_scheme": "light",
    },
    "Dark": {
        "bg": "#0F172A",
        "panel": "#111827",
        "text": "#F9FAFB",
        "muted": "#94A3B8",
        "border": "#263244",
        "accent": "#60A5FA",
        "accent_soft": "#1E293B",
        "shadow": "0 1px 3px rgba(0,0,0,0.5)",
        "input_bg": "#0B1220",
        "input_text": "#F9FAFB",
        "input_border": "#334155",
        "input_placeholder": "#94A3B8",
        "button_text": "#FFFFFF",
        "radio_bg": "#0B1220",
        "toggle_track": "#334155",
        "toggle_thumb": "#E2E8F0",
        "scrollbar": "#334155",
        "color_scheme": "dark",
    },
}


def inject_css(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: {t['color_scheme']};
        }}

        html, body, [data-testid="stAppViewContainer"], .stApp {{
            background: {t['bg']} !important;
            color: {t['text']} !important;
            color-scheme: {t['color_scheme']};
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background: {t['bg']} !important;
        }}

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        footer {{
            background: transparent !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {t['panel']} !important;
            border-right: 1px solid {t['border']} !important;
            color: {t['text']} !important;
        }}

        section[data-testid="stSidebar"] *,
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown label {{
            color: {t['text']};
        }}

        .app-title {{
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 0.15rem;
            color: {t['text']} !important;
        }}

        .app-subtitle {{
            color: {t['muted']} !important;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }}

        .metric-card {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: {t['shadow']};
        }}

        .metric-label {{
            color: {t['muted']};
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .metric-value {{
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.25rem;
            color: {t['text']} !important;
        }}

        .pill {{
            display: inline-block;
            padding: 0.16rem 0.6rem;
            border-radius: 999px;
            background: {t['accent_soft']};
            color: {t['accent']};
            font-weight: 700;
            font-size: 0.8rem;
            border: 1px solid {t['border']};
        }}

        .stButton > button {{
            border-radius: 12px !important;
            font-weight: 700 !important;
            border: 1px solid {t['accent']} !important;
            background: {t['accent']} !important;
            color: {t['button_text']} !important;
        }}

        .stButton > button:hover {{
            opacity: 0.96;
        }}

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {{
            background: {t['input_bg']} !important;
            color: {t['input_text']} !important;
            border: 1px solid {t['input_border']} !important;
        }}

        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {{
            color: {t['input_placeholder']} !important;
            opacity: 1 !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div * {{
            background: {t['input_bg']} !important;
            color: {t['input_text']} !important;
        }}

        div[data-baseweb="select"] > div {{
            border: 1px solid {t['input_border']} !important;
            box-shadow: none !important;
        }}

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {{
            background: {t['panel']} !important;
            color: {t['text']} !important;
            border-color: {t['border']} !important;
        }}

        div[data-testid="stRadio"] div[role="radiogroup"],
        div[data-testid="stToggle"] {{
            background: transparent !important;
        }}

        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label,
        div[data-testid="stToggle"] label,
        div[data-testid="stSlider"] label {{
            color: {t['text']} !important;
        }}

        div[data-testid="stRadio"] [role="radio"] {{
            border-color: {t['border']} !important;
            background: {t['radio_bg']} !important;
        }}

        div[data-testid="stToggle"] [data-testid="stTickBar"],
        div[data-testid="stToggle"] [role="switch"] {{
            background: {t['toggle_track']} !important;
        }}

        div[data-testid="stToggle"] [role="switch"] div {{
            background: {t['toggle_thumb']} !important;
        }}

        .stDataFrame, .stTable {{
            background: {t['panel']} !important;
        }}

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            border: 1px solid {t['border']} !important;
            border-radius: 14px !important;
            overflow: hidden;
        }}

        table {{
            color: {t['text']} !important;
        }}

        .stDataFrame [role="table"],
        .stDataFrame [role="grid"],
        .stDataFrame [data-testid="stArrowVegaLiteChart"],
        .stTable [role="table"] {{
            background: {t['panel']} !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: transparent !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {t['text']} !important;
            background: {t['panel']} !important;
            border: 1px solid {t['border']} !important;
            border-bottom: none !important;
            border-radius: 12px 12px 0 0 !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: {t['accent_soft']} !important;
            color: {t['accent']} !important;
        }}

        .log-box {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 0.9rem 1rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.84rem;
            color: {t['muted']};
            max-height: 320px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}

        .section-card {{
            background: {t['panel']};
            border: 1px solid {t['border']};
            border-radius: 16px;
            padding: 1rem 1rem 0.8rem;
            box-shadow: {t['shadow']};
        }}

        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}

        ::-webkit-scrollbar-thumb {{
            background: {t['scrollbar']};
            border-radius: 999px;
        }}

        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(col, label: str, value: str) -> None:
    col.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div></div>""",
        unsafe_allow_html=True,
    )


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "—"


def _now_stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%H:%M:%S")


def _add_log(message: str) -> None:
    logs = st.session_state.setdefault("activity_logs", [])
    logs.append(f"[{_now_stamp()}] {message}")


def _render_logs(logs: list[str]) -> None:
    if logs:
        lines = "<br>".join(html.escape(line) for line in logs)
    else:
        lines = "No activity yet."
    st.markdown(f'<div class="log-box">{lines}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("activity_logs", [])
st.session_state.setdefault("result", None)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 💬 Sentiment Analysis of YouTube Comments")
    st.caption("Comment preprocessing, language detection and reporting")

    theme_name = st.radio("Appearance", ["Light", "Dark"], horizontal=True)
    st.divider()

    source = st.radio(
        "Data source", ["Sample dataset", "Upload CSV", "Live YouTube (API key)"]
    )

    uploaded = None
    text_column = "comment"

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload a comments CSV", type=["csv"])
        text_column = st.text_input("Text column name", value="comment")
    elif source == "Live YouTube (API key)":
        api_key = st.text_input(
            "YouTube API key", value=os.environ.get("YOUTUBE_API_KEY", ""), type="password"
        )
        mode = st.radio("Fetch by", ["Video ID", "Channel ID"], horizontal=True)
        identifier = st.text_input(mode, placeholder=f"Enter {mode.lower()}")
        max_comments = st.slider("Max comments", 20, 500, 200, step=20)

        if st.button("Collect comments"):
            try:
                if not api_key.strip():
                    raise ValueError("YouTube API key is required.")
                if not identifier.strip():
                    raise ValueError(f"{mode} is required.")
                collector = YouTubeCommentCollector(api_key=api_key)
                tmp = os.path.join(tempfile.gettempdir(), "yt_live_comments.csv")
                collector.collect_to_csv(
                    output_path=tmp,
                    video_id=identifier.strip() if mode == "Video ID" else None,
                    channel_id=identifier.strip() if mode == "Channel ID" else None,
                    max_comments=max_comments,
                )
                st.session_state["live_csv"] = tmp
                _add_log(
                    f"Collected comments using {mode.lower()} {identifier.strip()} and saved to temporary CSV."
                )
                st.success("Comments collected.")
            except Exception as exc:  # noqa: BLE001 - surface API errors to the user
                _add_log(f"Collection failed: {exc}")
                st.error(f"Collection failed: {exc}")

    st.divider()
    use_stemming = st.toggle("Apply stemming", value=False)
    run = st.button("Run pipeline", type="primary", use_container_width=True)


inject_css(theme_name)
chart_theme = "dark" if theme_name == "Dark" else "light"

st.markdown('<div class="app-title">Sentiment Analysis of YouTube Comments</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Multilingual collection · cleaning · NLP preprocessing · '
    "language classification</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve_source_df() -> tuple[pd.DataFrame, str]:
    if source == "Sample dataset":
        return pd.read_csv(SAMPLE_CSV, encoding="utf-8"), "comment"
    if source == "Upload CSV":
        if uploaded is None:
            raise ValueError("Upload a CSV file to continue.")
        if not text_column.strip():
            raise ValueError("Text column name is required.")
        return pd.read_csv(uploaded, encoding="utf-8"), text_column.strip()

    csv = st.session_state.get("live_csv")
    if not csv:
        raise ValueError("Collect comments first using the sidebar.")
    return pd.read_csv(csv, encoding="utf-8"), "comment"


def _columns_for_compare(df: pd.DataFrame, stemming_enabled: bool) -> list[str]:
    base = ["comment", "cleaned_comment"]
    if stemming_enabled:
        base.extend(["processed_text_no_stemming", "processed_text_with_stemming"])
    base.append("processed_text")
    base.append("language")
    return [c for c in base if c in df.columns]


# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
if run:
    try:
        df_in, col = _resolve_source_df()
        if col not in df_in.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df_in.columns)}")

        pipeline = PreprocessingPipeline(use_stemming=use_stemming)
        result = pipeline.run(dataframe=df_in, text_column=col)
        st.session_state["result"] = {
            "df": result.dataframe,
            "stats": result.language_stats,
            "distribution": result.language_distribution,
            "total": result.total_comments,
            "logs": result.logs,
            "report": result.preprocessing_report,
        }
        _add_log(
            f"Pipeline completed on {result.total_comments} comments with stemming "
            f"{'enabled' if use_stemming else 'disabled'}."
        )
        st.success("Pipeline completed.")
    except Exception as exc:  # noqa: BLE001 - show pipeline errors in the UI
        _add_log(f"Pipeline failed: {exc}")
        st.error(str(exc))

result = st.session_state.get("result")

st.write("")


# --------------------------------------------------------------------------- #
# Summary cards
# --------------------------------------------------------------------------- #
if result:
    distribution = result["distribution"]
    stats = result["stats"]
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total comments", f"{result['total']:,}")
    metric_card(c2, "Languages", str(len(distribution)))
    top_lang = stats[0]["language"] if stats else "—"
    metric_card(c3, "Top language", top_lang)
    top_pct = f"{stats[0]['percentage']}%" if stats else "—"
    metric_card(c4, "Top share", top_pct)
else:
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Total comments", "—")
    metric_card(c2, "Languages", "—")
    metric_card(c3, "Top language", "—")
    metric_card(c4, "Top share", "—")

st.write("")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_comments, tab_logs = st.tabs(["📊 Overview", "💬 Comments", "🪵 Logs"])

with tab_overview:
    if not result:
        st.info("Run the pipeline to see the language distribution, report and comparison table.")
    else:
        df = result["df"]
        distribution = result["distribution"]
        stats = result["stats"]

        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("Language distribution")
            st.pyplot(viz.bar_chart(distribution, theme=chart_theme))
        with right:
            st.subheader("Language share")
            st.pyplot(viz.pie_chart(distribution, theme=chart_theme))

        st.subheader("Language statistics")
        stats_df = pd.DataFrame(stats)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "language": st.column_config.TextColumn("Language"),
                "count": st.column_config.NumberColumn("Comments"),
                "percentage": st.column_config.ProgressColumn(
                    "Share", format="%.2f%%", min_value=0, max_value=100
                ),
            },
        )

        report = result.get("report", {})
        st.subheader("Preprocessing report")
        a, b, c, d = st.columns(4)
        changed_rows = int(report.get("changed_rows", 0) or 0)
        total_rows = int(report.get("total_rows", len(df)) or len(df))
        a.metric("Rows changed by stemming", f"{changed_rows:,}")
        b.metric("Rows unchanged", f"{max(total_rows - changed_rows, 0):,}")
        c.metric("Stemming used", "Yes" if report.get("stemming_enabled") else "No")
        d.metric("Change rate", _fmt_pct(100.0 * changed_rows / total_rows if total_rows else 0))

        compare_cols = _columns_for_compare(df, bool(report.get("stemming_enabled")))
        if compare_cols:
            st.caption("Comparison of cleaned and processed text")
            st.dataframe(
                df[compare_cols].head(12),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "comment": st.column_config.TextColumn("Original comment", width="large"),
                    "cleaned_comment": st.column_config.TextColumn("Cleaned"),
                    "processed_text_no_stemming": st.column_config.TextColumn("Without stemming"),
                    "processed_text_with_stemming": st.column_config.TextColumn("With stemming"),
                    "processed_text": st.column_config.TextColumn("Active output"),
                    "language": st.column_config.TextColumn("Language"),
                },
            )

with tab_comments:
    if not result:
        st.info("Run the pipeline to browse cleaned comments.")
    else:
        df = result["df"]
        distribution = result["distribution"]

        f1, f2 = st.columns([1, 2])
        with f1:
            langs = st.multiselect(
                "Filter by language",
                options=[lang for lang in SUPPORTED_LANGUAGES if lang in distribution]
                + [lng for lng in distribution if lng not in SUPPORTED_LANGUAGES],
                default=list(distribution.keys()),
            )
        with f2:
            query = st.text_input("Search comments", placeholder="Type to filter…")

        view = df[df["language"].isin(langs)] if langs else df
        if query:
            view = view[view["comment"].astype(str).str.contains(query, case=False, na=False)]

        report = result.get("report", {})
        display_cols = ["comment", "cleaned_comment"]
        if report.get("stemming_enabled"):
            display_cols.extend(["processed_text_no_stemming", "processed_text_with_stemming"])
        display_cols.extend(["processed_text", "language"])
        display_cols = [c for c in display_cols if c in view.columns]

        st.caption(f"Showing {len(view):,} of {len(df):,} comments")
        st.dataframe(
            view[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "comment": st.column_config.TextColumn("Original comment", width="large"),
                "cleaned_comment": st.column_config.TextColumn("Cleaned"),
                "processed_text_no_stemming": st.column_config.TextColumn("Without stemming"),
                "processed_text_with_stemming": st.column_config.TextColumn("With stemming"),
                "processed_text": st.column_config.TextColumn("Active output"),
                "language": st.column_config.TextColumn("Language"),
            },
        )

        export_cols = [
            c
            for c in [
                "comment_id",
                "video_id",
                "author",
                "comment",
                "cleaned_comment",
                "processed_text_no_stemming",
                "processed_text_with_stemming",
                "processed_text",
                "language",
            ]
            if c in view.columns
        ]
        csv_bytes = view[export_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export cleaned dataset (CSV)",
            data=csv_bytes,
            file_name="cleaned_comments.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_logs:
    st.subheader("Process logs")
    _render_logs(st.session_state.get("activity_logs", []))
