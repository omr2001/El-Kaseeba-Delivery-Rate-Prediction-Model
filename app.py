from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from model import (
    Settings,
    build_all_models,
    make_excel_download,
    make_forecast_input_template_download,
    read_forecast_input_file,
    read_orders_excel,
    run_forecast,
)

st.set_page_config(page_title="Delivery Rate Forecast", layout="wide")


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
def apply_theme(mode: str) -> None:
    """Visual theme override with dark dashboard styling."""
    if mode == "Dark":
        bg = (
            "radial-gradient(circle at top right, rgba(20,109,128,0.35), transparent 28%), "
            "linear-gradient(135deg, #04111B 0%, #031A22 45%, #062B34 100%)"
        )
        panel = "rgba(8, 30, 38, 0.86)"
        panel_2 = "rgba(5, 20, 28, 0.88)"
        text = "#FFFFFF"
        muted = "#FFFFFF"
        accent = "#73ADBB"
        accent_2 = "#73ADBB"
        border = "rgba(115, 173, 187, 0.24)"
        shadow = (
            "0 0 0 1px rgba(115,173,187,0.10), "
            "0 12px 30px rgba(0,0,0,0.28), "
            "inset 0 1px 0 rgba(255,255,255,0.03)"
        )
        title_shadow = "0 0 18px rgba(115,173,187,0.20)"
        signature_shadow = "0 0 12px rgba(115,173,187,0.28)"
    else:
        bg = "#FFFFFF"
        panel = "#F8FAFC"
        panel_2 = "#FFFFFF"
        text = "#0F172A"
        muted = "#475569"
        accent = "#0EA5B7"
        accent_2 = "#0F172A"
        border = "#E2E8F0"
        shadow = "0 1px 2px rgba(15,23,42,0.05)"
        title_shadow = "none"
        signature_shadow = "none"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text};
        }}

        .main .block-container {{
            padding-top: 1.4rem;
        }}

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {accent_2} !important;
            letter-spacing: 0.2px;
            font-weight: 650 !important;
            text-shadow: {title_shadow};
        }}

        [data-testid="stSidebar"] {{
            background: {panel_2};
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(135deg, {panel} 0%, {panel_2} 100%);
            border: 1px solid {border};
            box-shadow: {shadow};
            padding: 12px 14px;
            border-radius: 18px;
            min-height: 96px;
        }}

        div[data-testid="stMetricLabel"] p {{
            font-size: 0.84rem !important;
            color: {accent} !important;
            text-transform: none;
            letter-spacing: 0.15px;
            white-space: normal !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stMetricValue"] {{
            font-size: 1.26rem !important;
            color: {text} !important;
            word-break: break-word !important;
            font-weight: 650 !important;
        }}

        div[data-testid="stMetricDelta"] {{
            font-size: 0.88rem !important;
        }}

        .section-note {{
            background: linear-gradient(135deg, {panel} 0%, {panel_2} 100%);
            border: 1px solid {border};
            box-shadow: {shadow};
            padding: 12px 14px;
            border-radius: 14px;
            color: {muted};
        }}

        div[data-testid="stExpander"] {{
            background: linear-gradient(135deg, {panel} 0%, {panel_2} 100%);
            border: 1px solid {border};
            border-radius: 14px;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stNumberInput div[data-baseweb="input"] > div {{
            background: {panel_2};
            border-color: {border} !important;
        }}

        .stDataFrame,
        div[data-testid="stDataEditor"] {{
            border: 1px solid {border};
            border-radius: 14px;
            overflow: hidden;
        }}

        .stButton > button {{
            border-radius: 12px;
            border: 1px solid {border};
        }}

        .oa-signature {{
            margin-top: 28px;
            text-align: center;
            font-size: 0.78rem;
            letter-spacing: 0.18em;
            opacity: 0.92;
            color: {accent};
            font-style: italic;
            font-weight: 600;
            text-shadow: {signature_shadow};
        }}

        .oa-signature span {{
            display: inline-block;
            padding: 6px 12px;
            border: 1px solid {border};
            border-radius: 999px;
            background: linear-gradient(135deg, {panel} 0%, {panel_2} 100%);
            box-shadow: {shadow};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_dark_overrides() -> None:
    """Extra dark-mode overrides requested for white UI text."""
    st.markdown(
        """
        <style>
        /* Main page title: Delivery Rate Forecast System */
        h1 {
            color: #FFFFFF !important;
        }

        /* Subtitle under main title */
        div[data-testid="stCaptionContainer"] p {
            color: #FFFFFF !important;
        }

        /* File uploader label and text */
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploader"] section * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        /* Sidebar section titles and regular sidebar markdown */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown p {
            color: #FFFFFF !important;
        }

        /* Sidebar widget labels */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
            color: #FFFFFF !important;
        }

        /* Number inputs */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] div[data-baseweb="input"] input {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        /* Slider values and helper text */
        section[data-testid="stSidebar"] div[data-testid="stSlider"],
        section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
        section[data-testid="stSidebar"] div[data-testid="stSlider"] span {
            color: #FFFFFF !important;
        }

        /* Browse Files button */
        div[data-testid="stFileUploader"] button {
            background-color: #73ADBB !important;
            color: #0B1720 !important;
            border: 1px solid #73ADBB !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stFileUploader"] button:hover {
            background-color: #8FC3CD !important;
            color: #06151B !important;
            border: 1px solid #8FC3CD !important;
        }
        /* File uploader drop area text: Drag and drop file here */
        div[data-testid="stFileUploaderDropzone"] {
        background-color: #F8FAFC !important;
        border: 1px dashed #73ADBB !important;
        }

        div[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] small {
            color: #0B1720 !important;
            -webkit-text-fill-color: #0B1720 !important;
        }

        /* Upload icon color */
        div[data-testid="stFileUploaderDropzone"] svg {
        color: #73ADBB !important;
        fill: #73ADBB !important;
        }

        /* Make all metric/card text white in dark mode */
        div[data-testid="stMetric"],
        div[data-testid="stMetric"] *,
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] *,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] * {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        div[data-testid="stMetricLabel"] p {
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            font-weight: 700 !important;
        }
        /* Dark mode secondary action buttons:
   Clear scenario + Load sample simulation */
        div[data-testid="stButton"] button {
            background-color: #73ADBB !important;
            color: #0B1720 !important;
            border: 1px solid #73ADBB !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }
        
        div[data-testid="stButton"] button p {
            color: #0B1720 !important;
            -webkit-text-fill-color: #0B1720 !important;
            font-weight: 700 !important;
        }
        
        div[data-testid="stButton"] button:hover {
            background-color: #8FC3CD !important;
            color: #06151B !important;
            border: 1px solid #8FC3CD !important;
        }
        
        div[data-testid="stButton"] button:hover p {
            color: #06151B !important;
            -webkit-text-fill-color: #06151B !important;
        }
        /* File uploader dropzone text fix in dark mode */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #F8FAFC !important;
            border: 1px dashed #73ADBB !important;
        }
        
        section[data-testid="stFileUploaderDropzone"] * {
            color: #0B1720 !important;
            -webkit-text-fill-color: #0B1720 !important;
        }
        
        /* Specifically target "Drag and drop file here" */
        section[data-testid="stFileUploaderDropzone"] div,
        section[data-testid="stFileUploaderDropzone"] span,
        section[data-testid="stFileUploaderDropzone"] p,
        section[data-testid="stFileUploaderDropzone"] small {
            color: #0B1720 !important;
            -webkit-text-fill-color: #0B1720 !important;
        }
        
        /* Upload icon */
        section[data-testid="stFileUploaderDropzone"] svg {
            color: #73ADBB !important;
            fill: #73ADBB !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_plot_style(fig, mode: str):
    if mode == "Dark":
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#73ADBB", family="Inter, Segoe UI, sans-serif"),
            title_font=dict(color="#73ADBB", size=22),
            legend_title_font=dict(color="#73ADBB"),
            legend_font=dict(color="#73ADBB"),
            xaxis=dict(
                title_font=dict(color="#73ADBB"),
                tickfont=dict(color="#73ADBB"),
                gridcolor="rgba(115,173,187,0.16)",
                zerolinecolor="rgba(115,173,187,0.20)",
            ),
            yaxis=dict(
                title_font=dict(color="#73ADBB"),
                tickfont=dict(color="#73ADBB"),
                gridcolor="rgba(115,173,187,0.16)",
                zerolinecolor="rgba(115,173,187,0.20)",
            ),
            colorway=["#73ADBB", "#9AC6CF", "#5F96A3", "#B7D6DC", "#4C7F8A"],
        )
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A", family="Inter, Segoe UI, sans-serif"),
        )
    return fig


def pct(x: float) -> str:
    return f"{x:.2%}" if pd.notna(x) else "-"


def num(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}" if pd.notna(x) else "-"


def most_common_value(df: pd.DataFrame, sku: str, column: str, fallback: str = "") -> str:
    if df.empty or column not in df.columns:
        return fallback
    s = df.loc[df["sku_code"].astype(str).eq(str(sku)), column].dropna().astype(str)
    if s.empty:
        return fallback
    return s.value_counts().index[0]


def render_metric_grid(items: list[tuple], columns_per_row: int) -> None:
    """Render metric cards in multiple rows so mobile mode is readable."""
    for i in range(0, len(items), columns_per_row):
        cols = st.columns(columns_per_row)
        for col, item in zip(cols, items[i : i + columns_per_row]):
            if len(item) == 2:
                label, value = item
                col.metric(label, value)
            else:
                label, value, delta = item
                col.metric(label, value, delta=delta)


@st.cache_data(show_spinner="Building forecast model...")
def build_models_cached(
    file_bytes: bytes,
    sku_smoothing: float,
    recency_window_days: int,
    recency_weight: float,
    volume_elasticity: float,
    min_predicted_dr: float,
    max_predicted_dr: float,
    forecast_days: int,
    telesales_weight: float,
    whatsapp_weight: float,
    validation_weight: float,
    time_to_fulfill_weight: float,
    first_attempt_weight: float,
):
    """Cache model building so UI reruns do not repeatedly parse/rebuild the workbook."""
    cached_settings = Settings(
        sku_smoothing=float(sku_smoothing),
        recency_window_days=int(recency_window_days),
        recency_weight=float(recency_weight),
        volume_elasticity=float(volume_elasticity),
        min_predicted_dr=float(min_predicted_dr),
        max_predicted_dr=float(max_predicted_dr),
        forecast_days=int(forecast_days),
        telesales_weight=float(telesales_weight),
        whatsapp_weight=float(whatsapp_weight),
        validation_weight=float(validation_weight),
        time_to_fulfill_weight=float(time_to_fulfill_weight),
        first_attempt_weight=float(first_attempt_weight),
    )
    raw = read_orders_excel(io.BytesIO(file_bytes))
    return build_all_models(raw, cached_settings)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("App View")
    theme_mode = st.radio("Theme", ["Light", "Dark"], horizontal=True)

    input_mode = st.radio(
        "Input mode",
        ["Table mode", "Mobile form mode"],
        help="Use Table mode on laptop. Use Mobile form mode on phones/tablets.",
    )

    forecast_counting_mode = st.radio(
        "Forecast counting mode",
        ["One SKU = One Order mode", "Order Group mode"],
        help=(
            "One SKU = One Order mode treats every SKU row as separate orders. "
            "Order Group mode counts rows with the same scenario_order_group and Forecast Date as the same forecasted orders."
        ),
    )

    courier_input_mode = st.radio(
        "Courier input mode",
        ["Manual courier", "Auto courier from history"],
        help=(
            "Manual courier requires courier_name in the forecast input. "
            "Auto courier from history derives the courier from historical SKU/seller/date behavior."
        ),
    )

    auto_context_method = st.radio(
        "Auto-context method",
        ["Most common", "Historical distribution"],
        help=(
            "Most common uses the single most frequent historical city/WhatsApp/validation/timing context. "
            "Historical distribution splits the prediction across the historical mix of city, WhatsApp, validation, and timing buckets."
        ),
    )

    st.header("Model Settings")
    forecast_days = st.number_input("Forecast days", min_value=1, max_value=14, value=7, step=1)
    recency_window = st.number_input("Recency window days", min_value=1, max_value=30, value=7, step=1)
    recency_weight = st.slider("Recency weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    sku_smoothing = st.number_input("SKU smoothing strength", min_value=0.0, max_value=500.0, value=50.0, step=10.0)
    volume_elasticity = st.slider("Volume elasticity", min_value=0.0, max_value=0.20, value=0.03, step=0.01)
    telesales_weight = st.slider("Telesales weight", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    whatsapp_weight = st.slider("WhatsApp reply weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    validation_weight = st.slider("Validation tag weight", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
    time_to_fulfill_weight = st.slider("Time to fulfill weight", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
    first_attempt_weight = st.slider("Time to first attempt weight", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    calibration_factor = st.slider("Backtest calibration factor", min_value=0.50, max_value=2.00, value=0.83, step=0.01)
    min_dr = st.slider("Minimum predicted DR", min_value=0.0, max_value=0.50, value=0.05, step=0.01)
    max_dr = st.slider("Maximum predicted DR", min_value=0.50, max_value=1.0, value=0.95, step=0.01)

apply_theme(theme_mode)
if theme_mode == "Dark":
    apply_dark_overrides()

st.title("Delivery Rate Forecast System")
st.caption("Upload historical orders, enter SKU forecast scenarios, click Run Forecast, and measure selected-SKU and full-business DR impact.")

settings = Settings(
    sku_smoothing=float(sku_smoothing),
    recency_window_days=int(recency_window),
    recency_weight=float(recency_weight),
    volume_elasticity=float(volume_elasticity),
    min_predicted_dr=float(min_dr),
    max_predicted_dr=float(max_dr),
    forecast_days=int(forecast_days),
    telesales_weight=float(telesales_weight),
    whatsapp_weight=float(whatsapp_weight),
    validation_weight=float(validation_weight),
    time_to_fulfill_weight=float(time_to_fulfill_weight),
    first_attempt_weight=float(first_attempt_weight),
    calibration_factor=float(calibration_factor),
    forecast_counting_mode=forecast_counting_mode,
    courier_input_mode=courier_input_mode,
    auto_context_method=auto_context_method,
)

uploaded_file = st.file_uploader("Upload historical orders Excel file", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload the 45-day orders file to start.")
    st.stop()

try:
    file_bytes = uploaded_file.getvalue()
    models = build_models_cached(
        file_bytes,
        float(sku_smoothing),
        int(recency_window),
        float(recency_weight),
        float(volume_elasticity),
        float(min_dr),
        float(max_dr),
        int(forecast_days),
        float(telesales_weight),
        float(whatsapp_weight),
        float(validation_weight),
        float(time_to_fulfill_weight),
        float(first_attempt_weight),
    )
except Exception as exc:
    st.error("Could not process the uploaded file. Please check the column names and file format.")
    st.exception(exc)
    st.stop()

# -----------------------------------------------------------------------------
# Model outputs / top metrics
# -----------------------------------------------------------------------------
global_dr = float(models["global_dr"])
order_level = models["order_level_data"]
sku_model = models["sku_model"]
factor_model = models["factor_model"]
daily_sku = models["daily_sku_history"]
order_sku = models["order_sku_map"]

st.subheader("Historical Data Summary")
summary_items = [
    ("Historical Start", pd.to_datetime(models["historical_start"]).date().isoformat()),
    ("Historical End", pd.to_datetime(models["historical_end"]).date().isoformat()),
    ("Fulfilled Orders", f"{int(order_level['Fulfilled Order'].sum()):,}"),
    ("Delivered Orders", f"{int(order_level['Delivered Order'].sum()):,}"),
    ("Global DR", f"{global_dr:.2%}"),
    ("Unique SKUs", f"{sku_model['sku_code'].nunique():,}"),
]
render_metric_grid(summary_items, columns_per_row=2 if input_mode == "Mobile form mode" else 6)

with st.expander("Preview model tables"):
    tab1, tab2, tab3 = st.tabs(["SKU Model", "Factor Model", "Daily Orders"])
    with tab1:
        st.dataframe(sku_model.head(50), use_container_width=True)
    with tab2:
        st.dataframe(factor_model.head(100), use_container_width=True)
    with tab3:
        st.dataframe(models["daily_order_history"], use_container_width=True)

st.divider()
st.subheader("Forecast Scenario Input")
st.markdown(
    '<div class="section-note">Use manual entry or upload an Excel/CSV forecast input file. Required fields are Forecast Date, SKU, seller, planned orders, and courier only when Manual courier mode is selected. City, WhatsApp reply, validation tag, timing buckets, and telesales are auto-derived from historical data using the selected auto-context method. After preparing the scenario, click <b>Run Forecast</b>.</div>',
    unsafe_allow_html=True,
)

if forecast_counting_mode == "Order Group mode":
    st.info(
        "Order Group mode: rows with the same scenario_order_group and Forecast Date are counted as the same forecasted orders. "
        "Use this when one order can contain multiple SKUs."
    )
else:
    st.info(
        "One SKU = One Order mode: every SKU row is counted as separate forecasted orders. "
        "Use this for normal SKU planning where each SKU row represents its own orders."
    )

if courier_input_mode == "Auto courier from history":
    st.info(
        "Auto courier mode: courier_name is derived from historical data using SKU, seller, and forecast weekday. "
        "The output table will show which courier was used."
    )
else:
    st.info("Manual courier mode: courier_name is entered by the user and used directly in the forecast.")

if auto_context_method == "Historical distribution":
    st.info(
        "Historical distribution mode: city, WhatsApp reply, validation tag, and timing buckets are applied as a historical mix from the best matching context, not as one most-common value."
    )
else:
    st.info(
        "Most common auto-context mode: city, WhatsApp reply, validation tag, and timing buckets are filled using the single most common value from the best matching historical context."
    )

forecast_start = pd.to_datetime(models["historical_end"]) + pd.Timedelta(days=1)
forecast_dates = [(forecast_start + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(settings.forecast_days)]

# Options for dropdowns.
sku_options = sorted(sku_model["sku_code"].astype(str).unique().tolist())
seller_options = sorted(models["order_level_data"]["seller_name"].astype(str).dropna().unique().tolist())
courier_options = sorted(models["order_level_data"]["courier_name"].astype(str).dropna().unique().tolist())
city_options = sorted(models["order_level_data"]["mapped_city"].astype(str).dropna().unique().tolist())
whatsapp_options = sorted(models["order_level_data"].get("wap_recentresponse", pd.Series(dtype=str)).astype(str).dropna().unique().tolist())
validation_options = sorted(models["order_level_data"].get("validation_tag", pd.Series(dtype=str)).astype(str).dropna().unique().tolist())

input_method = st.radio(
    "Forecast Input Method",
    ["Manual entry", "Bulk upload"],
    horizontal=True,
    help="Manual entry lets you type rows in the app. Bulk upload lets you upload an Excel/CSV scenario file.",
)


def blank_input() -> pd.DataFrame:
    data = {
        "Forecast Date": [forecast_dates[0]],
        "sku_code": [""],
        "seller_name": [""],
        "Planned Fulfilled Orders": [0.0],
        "Selected?": ["Yes"],
    }

    if courier_input_mode == "Manual courier":
        data = {**data, "courier_name": [""]}

    if forecast_counting_mode == "Order Group mode":
        data = {"scenario_order_group": [""], **data}

    return pd.DataFrame(data)


if "forecast_input" not in st.session_state:
    st.session_state.forecast_input = blank_input()

# Keep forecast input columns aligned with the selected counting mode.
# This matters when the user switches between:
# - One SKU = One Order mode
# - Order Group mode
# Without this sync, the manual table may not show scenario_order_group after switching modes.
if forecast_counting_mode == "Order Group mode":
    if "scenario_order_group" not in st.session_state.forecast_input.columns:
        st.session_state.forecast_input.insert(0, "scenario_order_group", "")
else:
    # In One SKU = One Order mode, every row is automatically treated as its own order group,
    # so scenario_order_group is hidden from manual input.
    if "scenario_order_group" in st.session_state.forecast_input.columns:
        st.session_state.forecast_input = st.session_state.forecast_input.drop(columns=["scenario_order_group"])

# Keep courier_name aligned with selected courier input mode.
if courier_input_mode == "Manual courier":
    if "courier_name" not in st.session_state.forecast_input.columns:
        insert_at = min(len(st.session_state.forecast_input.columns), 4 if forecast_counting_mode == "Order Group mode" else 3)
        st.session_state.forecast_input.insert(insert_at, "courier_name", "")
else:
    if "courier_name" in st.session_state.forecast_input.columns:
        st.session_state.forecast_input = st.session_state.forecast_input.drop(columns=["courier_name"])

# Keep Forecast Date as text so Streamlit data_editor can use a selectbox.
if "Forecast Date" in st.session_state.forecast_input.columns:
    st.session_state.forecast_input["Forecast Date"] = pd.to_datetime(
        st.session_state.forecast_input["Forecast Date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d").fillna(forecast_dates[0])

sample_sku = "BKSA-KS-HO-S19-BL1165"
button_cols = st.columns([1.1, 1.4, 4])

with button_cols[0]:
    if st.button("Clear scenario", use_container_width=True):
        st.session_state.forecast_input = blank_input()
        st.session_state.pop("forecast_result", None)
        st.rerun()

with button_cols[1]:
    if st.button("Load sample simulation", use_container_width=True):
        sku = sample_sku if sample_sku in sku_options else (sku_options[0] if sku_options else "")

        default_seller = most_common_value(order_sku, sku, "seller_name", seller_options[0] if seller_options else "")
        default_courier = most_common_value(order_sku, sku, "courier_name", courier_options[0] if courier_options else "")

        sample_dates = forecast_dates[: min(3, len(forecast_dates))]
        sample_orders = [300.0, 250.0, 300.0][: len(sample_dates)]

        sample_data = {
            "Forecast Date": sample_dates,
            "sku_code": [sku] * len(sample_dates),
            "seller_name": [default_seller] * len(sample_dates),
            "Planned Fulfilled Orders": sample_orders,
            "Selected?": ["Yes"] * len(sample_dates),
        }

        if courier_input_mode == "Manual courier":
            sample_data = {**sample_data, "courier_name": [default_courier] * len(sample_dates)}

        if forecast_counting_mode == "Order Group mode":
            sample_data = {"scenario_order_group": [f"SAMPLE_{i+1}" for i in range(len(sample_dates))], **sample_data}

        st.session_state.forecast_input = pd.DataFrame(sample_data)

        st.session_state.pop("forecast_result", None)
        st.rerun()

with button_cols[2]:
    st.caption(
        "Manual entry and bulk upload both use the same model. "
        "The model will not run while you are editing. Click Run Forecast when finished."
    )

run_clicked = False
edited_input = st.session_state.forecast_input.copy()

# -----------------------------------------------------------------------------
# MANUAL INPUT METHOD
# -----------------------------------------------------------------------------
if input_method == "Manual entry":
    # -------------------------------------------------------------------------
    # TABLE MODE
    # -------------------------------------------------------------------------
    if input_mode == "Table mode":
        with st.form("forecast_input_form", clear_on_submit=False):
            edited_input = st.data_editor(
                st.session_state.forecast_input,
                num_rows="dynamic",
                use_container_width=True,
                column_config=(
                    {
                        "scenario_order_group": st.column_config.TextColumn(
                            "scenario_order_group",
                            help="Optional. Use the same group ID for SKUs in the same forecasted orders. Blank = one group per row.",
                        )
                    }
                    if forecast_counting_mode == "Order Group mode"
                    else {}
                )
                | {
                    "Forecast Date": st.column_config.SelectboxColumn("Forecast Date", options=forecast_dates, required=True),
                    "sku_code": st.column_config.SelectboxColumn("sku_code", options=[""] + sku_options, required=True),
                    "seller_name": st.column_config.SelectboxColumn("seller_name", options=[""] + seller_options, required=True),
                    **({
                        "courier_name": st.column_config.SelectboxColumn("courier_name", options=[""] + courier_options, required=True),
                    } if courier_input_mode == "Manual courier" else {}),
                    "Planned Fulfilled Orders": st.column_config.NumberColumn(
                        "Planned Fulfilled Orders",
                        min_value=0.0,
                        step=1.0,
                        format="%.2f",
                    ),
                    "Selected?": st.column_config.SelectboxColumn("Selected?", options=["Yes", "No"], required=True),
                },
                key="forecast_editor_form",
            )

            run_clicked = st.form_submit_button("Run Forecast", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # MOBILE FORM MODE
    # -------------------------------------------------------------------------
    else:
        st.markdown("### Add Forecast Row")

        with st.form("mobile_add_row_form", clear_on_submit=True):
            if forecast_counting_mode == "Order Group mode":
                m_group = st.text_input(
                    "Scenario order group (optional)",
                    help="Use the same group ID for multiple SKUs that belong to the same forecasted orders.",
                )
            else:
                m_group = ""
            m_date = st.selectbox("Forecast Date", forecast_dates)
            m_sku = st.selectbox("SKU", [""] + sku_options)
            m_seller = st.selectbox("Seller", [""] + seller_options)
            if courier_input_mode == "Manual courier":
                m_courier = st.selectbox("Courier", [""] + courier_options)
            else:
                m_courier = ""
                st.caption("Courier will be auto-derived from historical data.")
            m_orders = st.number_input("Planned Fulfilled Orders", min_value=0.0, step=1.0, format="%.2f")
            m_selected = st.selectbox("Selected?", ["Yes", "No"])

            add_row_clicked = st.form_submit_button("Add Row", use_container_width=True)

        if add_row_clicked:
            missing_courier = courier_input_mode == "Manual courier" and not m_courier
            if not m_sku or not m_seller or missing_courier or m_orders <= 0:
                st.warning("Please complete SKU, seller, courier if manual mode is selected, and planned orders before adding the row.")
            else:
                new_row_data = {
                    "Forecast Date": m_date,
                    "sku_code": m_sku,
                    "seller_name": m_seller,
                    "Planned Fulfilled Orders": float(m_orders),
                    "Selected?": m_selected,
                }

                if courier_input_mode == "Manual courier":
                    new_row_data = {**new_row_data, "courier_name": m_courier}

                if forecast_counting_mode == "Order Group mode":
                    new_row_data = {"scenario_order_group": m_group.strip(), **new_row_data}

                new_row = pd.DataFrame([new_row_data])

                current_input = st.session_state.forecast_input.copy()
                is_initial_blank = (
                    len(current_input) == 1
                    and current_input["sku_code"].astype(str).iloc[0].strip() == ""
                    and float(current_input["Planned Fulfilled Orders"].iloc[0]) == 0
                )

                if is_initial_blank:
                    st.session_state.forecast_input = new_row
                else:
                    st.session_state.forecast_input = pd.concat([current_input, new_row], ignore_index=True)

                st.session_state.pop("forecast_result", None)
                st.success("Forecast row added.")
                st.rerun()

        st.markdown("### Current Scenario Rows")
        st.dataframe(st.session_state.forecast_input, use_container_width=True)

        with st.form("mobile_run_forecast_form", clear_on_submit=False):
            run_clicked = st.form_submit_button("Run Forecast", type="primary", use_container_width=True)

        edited_input = st.session_state.forecast_input.copy()

# -----------------------------------------------------------------------------
# BULK UPLOAD INPUT METHOD
# -----------------------------------------------------------------------------
else:
    st.markdown("### Bulk Upload Forecast Input")
    courier_note = "<b>courier_name</b>, " if courier_input_mode == "Manual courier" else ""
    group_note = "optional <b>scenario_order_group</b>, " if forecast_counting_mode == "Order Group mode" else ""
    st.markdown(
        f'<div class="section-note">Upload an Excel or CSV file with columns: {group_note}<b>Forecast Date</b>, <b>sku_code</b>, <b>seller_name</b>, {courier_note}<b>Planned Fulfilled Orders</b>, and optional <b>Selected?</b>. Use the same scenario_order_group for multiple SKUs inside the same forecasted orders when Order Group mode is selected. City, WhatsApp, validation tag, timing buckets, telesales, and courier when Auto mode is selected will be auto-derived.</div>',
        unsafe_allow_html=True,
    )

    template_bytes = make_forecast_input_template_download(models, settings)
    st.download_button(
        "Download Forecast Input Template",
        data=template_bytes,
        file_name="forecast_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    uploaded_forecast_file = st.file_uploader(
        "Upload forecast input Excel/CSV file",
        type=["xlsx", "xls", "csv"],
        key="forecast_input_bulk_upload",
    )

    if uploaded_forecast_file is not None:
        try:
            uploaded_forecast_input = read_forecast_input_file(
                uploaded_forecast_file,
                default_forecast_date=forecast_dates[0],
                forecast_counting_mode=forecast_counting_mode,
                courier_input_mode=courier_input_mode,
            )
        except Exception as exc:
            st.error("Could not read the forecast input file. Please use the template columns and try again.")
            st.exception(exc)
            uploaded_forecast_input = pd.DataFrame()

        if not uploaded_forecast_input.empty:
            st.markdown("### Uploaded Forecast Rows")
            st.dataframe(uploaded_forecast_input, use_container_width=True)

            bulk_cols = st.columns([1.3, 1.3, 4])
            with bulk_cols[0]:
                load_uploaded_clicked = st.button(
                    "Load Uploaded Rows",
                    use_container_width=True,
                )
            with bulk_cols[1]:
                run_uploaded_clicked = st.button(
                    "Run Forecast From Upload",
                    type="primary",
                    use_container_width=True,
                )
            with bulk_cols[2]:
                st.caption("Load rows if you want to review/edit later, or run directly from the uploaded file.")

            if load_uploaded_clicked:
                st.session_state.forecast_input = uploaded_forecast_input.copy()
                st.session_state.pop("forecast_result", None)
                st.success("Uploaded rows loaded into Forecast Input. Switch to Manual entry to review/edit, or click Run Forecast From Upload.")

            if run_uploaded_clicked:
                edited_input = uploaded_forecast_input.copy()
                run_clicked = True
        else:
            st.warning("The uploaded forecast input file has no usable rows.")
    else:
        st.info("Download the template, fill it in Excel, then upload it here.")
# -----------------------------------------------------------------------------
# Run forecast only after Run Forecast is clicked
# -----------------------------------------------------------------------------
if run_clicked:
    st.session_state.forecast_input = edited_input.copy()

    valid_input = edited_input.copy()
    valid_mask = (
        (valid_input["Selected?"].astype(str).str.lower() == "yes")
        & (valid_input["sku_code"].astype(str).str.strip() != "")
        & (valid_input["seller_name"].astype(str).str.strip() != "")
        & (pd.to_numeric(valid_input["Planned Fulfilled Orders"], errors="coerce") > 0)
    )
    if courier_input_mode == "Manual courier":
        if "courier_name" not in valid_input.columns:
            valid_input["courier_name"] = ""
        valid_mask = valid_mask & (valid_input["courier_name"].astype(str).str.strip() != "")
    valid_input = valid_input[valid_mask]

    if valid_input.empty:
        st.warning("Please enter at least one complete selected forecast row before running the forecast.")
        st.session_state.pop("forecast_result", None)
    else:
        forecast, metrics, daily_output = run_forecast(edited_input, models, settings)

        st.session_state.forecast_result = {
            "forecast": forecast,
            "metrics": metrics,
            "daily_output": daily_output,
        }

if "forecast_result" not in st.session_state:
    st.info("After entering or editing scenario rows, click **Run Forecast** to generate outputs.")
    st.markdown('<div class="oa-signature"><span>OA</span></div>', unsafe_allow_html=True)
    st.stop()

forecast = st.session_state.forecast_result["forecast"]
metrics = st.session_state.forecast_result["metrics"]
daily_output = st.session_state.forecast_result["daily_output"]

# -----------------------------------------------------------------------------
# Forecast output
# -----------------------------------------------------------------------------
st.subheader("Forecast Output")
selected_items = [
    ("Selected Planned Orders", f"{metrics['selected_planned_orders']:,.0f}"),
    ("Selected Current DR", f"{metrics['selected_current_dr']:.2%}"),
    ("Selected Forecast DR", f"{metrics['selected_forecast_dr']:.2%}", f"{metrics['selected_impact_vs_current']:.2%}"),
    ("Selected Predicted Delivered", f"{metrics['selected_predicted_delivered']:,.1f}"),
]
render_metric_grid(selected_items, columns_per_row=2 if input_mode == "Mobile form mode" else 4)

business_items = [
    ("Full Business Current DR", f"{metrics['full_business_current_dr']:.2%}"),
    ("Full Business Forecast DR", f"{metrics['full_business_forecast_dr']:.2%}", f"{metrics['full_business_impact']:.2%}"),
    ("Full Business Forecast Orders", f"{metrics['full_business_forecast_orders']:,.0f}"),
    ("Full Business Forecast Delivered", f"{metrics['full_business_forecast_delivered']:,.1f}"),
]
render_metric_grid(business_items, columns_per_row=2 if input_mode == "Mobile form mode" else 4)

with st.expander("How full business impact is calculated"):
    st.markdown(
        """
**Full Business Current DR** is the baseline full-business forecast using every SKU's average daily orders and base predicted DR.

**Full Business Forecast DR** replaces the baseline for selected order groups with your scenario input. If multiple SKUs share the same `scenario_order_group`, the group is counted once as one planned-order volume:

```text
Full Business Forecast Orders = Baseline Orders - Selected Group Baseline Orders + Selected Group Planned Orders
Full Business Forecast Delivered = Baseline Delivered - Selected Group Baseline Delivered + Selected Group Predicted Delivered
Full Business Forecast DR = Full Business Forecast Delivered / Full Business Forecast Orders
```

This means you can enter one or two SKUs only. The selected SKUs use your forecast scenario. City, WhatsApp, validation, and telesales are derived automatically from historical data, while all other SKUs remain at normal baseline.
        """
    )

if not forecast.empty:
    st.dataframe(forecast, use_container_width=True)
else:
    st.info("Enter at least one selected SKU scenario row and click Run Forecast to see forecast output.")

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
if input_mode == "Mobile form mode":
    chart_col1 = st.container()
    chart_col2 = st.container()
else:
    chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if not forecast.empty:
        comp = forecast[forecast["Selected?"].str.lower().eq("yes")].groupby("sku_code", as_index=False).agg(
            **{
                "Planned Orders": ("Planned Fulfilled Orders", "sum"),
                "Current DR": ("SKU DR Now", "mean"),
                "Last 3 Days DR": ("SKU Last 3 Days DR", "mean"),
                "Forecast DR": ("Predicted DR", "mean"),
            }
        )
        if not comp.empty:
            comp_long = comp.melt(
                id_vars=["sku_code", "Planned Orders"],
                value_vars=["Current DR", "Last 3 Days DR", "Forecast DR"],
                var_name="Metric",
                value_name="DR",
            )
            fig = px.bar(
                comp_long,
                x="sku_code",
                y="DR",
                color="Metric",
                barmode="group",
                text=comp_long["DR"].map(lambda x: f"{x:.1%}"),
                title="Selected SKU DR: Current / Last 3 Days / Forecast",
            )
            fig.update_layout(xaxis_title="SKU", yaxis_tickformat=".0%")
            fig.update_xaxes(tickangle=45 if input_mode == "Mobile form mode" else 0)
            fig = apply_plot_style(fig, theme_mode)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    total_comp = pd.DataFrame(
        {
            "Metric": ["Full Business Current DR", "Full Business Forecast DR"],
            "DR": [metrics["full_business_current_dr"], metrics["full_business_forecast_dr"]],
        }
    )
    fig2 = px.bar(
        total_comp,
        x="Metric",
        y="DR",
        text=total_comp["DR"].map(lambda x: f"{x:.2%}"),
        title="Total DR Before vs After Forecast",
    )
    fig2.update_layout(yaxis_tickformat=".0%")
    fig2.update_xaxes(tickangle=20 if input_mode == "Mobile form mode" else 0)
    fig2 = apply_plot_style(fig2, theme_mode)
    fig2.update_traces(textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

if not daily_output.empty:
    fig3 = px.line(
        daily_output,
        x="Forecast Date",
        y="Weighted DR",
        markers=True,
        text=daily_output["Weighted DR"].map(lambda x: f"{x:.1%}"),
        title="Daily Selected Forecast DR",
    )
    fig3.update_layout(yaxis_tickformat=".0%")
    fig3 = apply_plot_style(fig3, theme_mode)
    fig3.update_traces(textposition="top center")
    st.plotly_chart(fig3, use_container_width=True)

# SKU trend chart.
st.subheader("SKU Trend Before and After Forecast")
selected_skus = forecast["sku_code"].dropna().astype(str).unique().tolist() if not forecast.empty else sku_options[:10]
if selected_skus:
    chosen_sku = st.selectbox("Choose SKU for trend", options=selected_skus)
    hist = daily_sku[daily_sku["sku_code"] == chosen_sku][["Fulfilled Date", "DR%", "Fulfilled Orders"]].copy()
    hist = hist.rename(columns={"Fulfilled Date": "Date", "DR%": "DR", "Fulfilled Orders": "Orders"})
    hist["Period"] = "Historical"

    fc = forecast[(forecast["sku_code"] == chosen_sku) & (forecast["Selected?"].str.lower().eq("yes"))][
        ["Forecast Date", "Predicted DR", "Planned Fulfilled Orders"]
    ].copy()
    fc = fc.rename(columns={"Forecast Date": "Date", "Predicted DR": "DR", "Planned Fulfilled Orders": "Orders"})
    fc["Period"] = "Forecast"

    trend = pd.concat([hist, fc], ignore_index=True)
    if not trend.empty:
        fig4 = px.line(
            trend,
            x="Date",
            y="DR",
            color="Period",
            markers=True,
            hover_data=["Orders"],
            title=f"{chosen_sku}: Historical DR vs Forecast DR",
        )
        fig4.update_layout(yaxis_tickformat=".0%")
        fig4 = apply_plot_style(fig4, theme_mode)
        st.plotly_chart(fig4, use_container_width=True)

excel_bytes = make_excel_download(models, forecast, metrics, daily_output)
st.download_button(
    "Download forecast output Excel",
    data=excel_bytes,
    file_name="delivery_forecast_output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown('<div class="oa-signature"><span>OA</span></div>', unsafe_allow_html=True)
