import streamlit as st

from stock_forecaster.chart_service import build_price_chart, filter_data_by_timeframe
from stock_forecaster.config import AppConfig, ModelConfig
from stock_forecaster.data_service import DataServiceError, load_stock_bundle
from stock_forecaster.forecast_service import predict_next_day
from stock_forecaster.model_store import get_or_train_artifact


APP_CONFIG = AppConfig()
MODEL_CONFIG = ModelConfig()


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f7f9fc;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --primary: #0f766e;
            --accent: #0ea5e9;
        }

        .stApp {
            background: radial-gradient(circle at top right, #e0f2fe 0%, var(--bg) 45%);
        }

        .main-title {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: var(--muted);
            margin-bottom: 1rem;
        }

        .info-card {
            background: var(--card);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px 16px;
            margin: 8px 0;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .card-title {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 4px;
        }

        .card-value {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 700;
        }

        @media (max-width: 768px) {
            .main-title { font-size: 1.6rem; }
            .card-value { font-size: 1.05rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=APP_CONFIG.cache_ttl)
def load_data_cached(ticker: str):
    return load_stock_bundle(ticker, sequence_length=MODEL_CONFIG.sequence_length)


@st.cache_resource
def train_or_load_model_cached(ticker: str, data):
    return get_or_train_artifact(
        ticker=ticker,
        data=data,
        config=MODEL_CONFIG,
        force_retrain=False,
    )


def render_sidebar():
    with st.sidebar:
        st.header("Forecast Controls")
        ticker_input = st.text_input("Stock Symbol", "AAPL").upper().strip()

        st.subheader("Visualization")
        timeframe = st.radio(
            "Timeframe",
            ("1M", "6M", "1Y", "5Y", "Max"),
            index=2,
            horizontal=True,
        )
        chart_type = st.radio("Chart Type", ("Line", "Candlestick"), horizontal=True)

        st.subheader("Model")
        force_retrain = st.checkbox("Force Retrain Model", value=False)

        predict_button = st.button("Run Forecast", type="primary", use_container_width=True)
        st.markdown("---")
        st.caption(
            "If Yahoo rate-limits requests, cached market data is used automatically when available."
        )
        st.info("Educational demo only. Do not use for financial decisions.")

    return ticker_input, timeframe, chart_type, force_retrain, predict_button


def render_evaluation_block(evaluation, run_config):
    st.subheader("Model Evaluation")

    model_cols = st.columns(3)
    model_cols[0].metric("MAE (Model)", f"{evaluation.model_metrics.mae:,.4f}")
    model_cols[1].metric("RMSE (Model)", f"{evaluation.model_metrics.rmse:,.4f}")
    model_cols[2].metric("MAPE (Model)", f"{evaluation.model_metrics.mape:,.2f}%")

    base_cols = st.columns(3)
    base_cols[0].metric("MAE (Baseline)", f"{evaluation.baseline_metrics.mae:,.4f}")
    base_cols[1].metric("RMSE (Baseline)", f"{evaluation.baseline_metrics.rmse:,.4f}")
    base_cols[2].metric("MAPE (Baseline)", f"{evaluation.baseline_metrics.mape:,.2f}%")

    st.caption(
        f"Directional Accuracy: {evaluation.direction_accuracy:,.2f}% | "
        f"Split -> Train {evaluation.train_size}, Validation {evaluation.val_size}, Test {evaluation.test_size}"
    )

    with st.expander("Training Run Details"):
        st.json(run_config)


def render_forecast(ticker: str, timeframe: str, chart_type: str, force_retrain: bool) -> None:
    with st.spinner(f"Preparing forecast for {ticker}..."):
        bundle = load_data_cached(ticker)
        if force_retrain:
            artifact = get_or_train_artifact(
                ticker=ticker,
                data=bundle.data,
                config=MODEL_CONFIG,
                force_retrain=True,
            )
        else:
            artifact = train_or_load_model_cached(ticker, bundle.data)

        last_close = float(bundle.data["Close"].iloc[-1])
        forecast = predict_next_day(
            model=artifact.model,
            scaler=artifact.scaler,
            last_window=bundle.last_window,
            last_close=last_close,
        )

        trend_icon = "🔼" if forecast.trend == "Up" else "🔽"
        model_source = "Freshly Trained" if artifact.source == "trained" else "Loaded from Cache"

        st.markdown(f"### Forecast for {bundle.profile.name} ({ticker})")
        st.caption(f"Model source: {model_source} | Artifact: {artifact.artifact_path}")

        hero_cols = st.columns([1.3, 1, 1, 1])
        hero_cols[0].metric(
            label="Estimated Closing Price",
            value=f"${forecast.close_price:,.2f}",
            delta=f"{forecast.delta_close:,.2f} ({forecast.trend}) {trend_icon}",
        )
        hero_cols[1].metric("Open", f"${forecast.open_price:,.2f}")
        hero_cols[2].metric("High", f"${forecast.high_price:,.2f}")
        hero_cols[3].metric("Low", f"${forecast.low_price:,.2f}")

        st.markdown(
            f"<div class='info-card'><div class='card-title'>Volume Forecast</div>"
            f"<div class='card-value'>{forecast.volume:,.0f}</div></div>",
            unsafe_allow_html=True,
        )

        tab_forecast, tab_eval, tab_chart, tab_data = st.tabs(
            ["Forecast Summary", "Evaluation", "Market Chart", "Raw Data"]
        )

        with tab_forecast:
            st.write(f"**Business/Exchange Info:** {bundle.profile.sector}")
            st.write(
                "The model predicts tomorrow values using the latest historical sequence "
                f"of {MODEL_CONFIG.sequence_length} trading days."
            )

        with tab_eval:
            render_evaluation_block(artifact.evaluation, artifact.run_config)

        with tab_chart:
            chart_data = filter_data_by_timeframe(bundle.data, timeframe)
            fig = build_price_chart(chart_data=chart_data, ticker=ticker, chart_type=chart_type)
            fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with tab_data:
            st.dataframe(bundle.data.tail(20).style.format("{:,.2f}"), use_container_width=True)


def run_app() -> None:
    st.set_page_config(
        page_title=APP_CONFIG.page_title,
        page_icon=APP_CONFIG.page_icon,
        layout=APP_CONFIG.layout,
    )

    inject_custom_styles()

    st.markdown("<div class='main-title'>AI Stock Price Forecaster</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Interview-ready LSTM forecasting dashboard with evaluation and baseline comparison.</div>",
        unsafe_allow_html=True,
    )

    ticker_input, timeframe, chart_type, force_retrain, predict_button = render_sidebar()

    if not predict_button:
        st.info("Set your options in the sidebar and click 'Run Forecast' to start.")
        return

    if not ticker_input:
        st.warning("Please enter a stock symbol.")
        return

    try:
        render_forecast(
            ticker=ticker_input,
            timeframe=timeframe,
            chart_type=chart_type,
            force_retrain=force_retrain,
        )
    except DataServiceError as exc:
        st.error(str(exc))
        st.info(
            "Tip: wait 1-2 minutes if Yahoo temporarily rate-limits requests, then retry."
        )
    except Exception:
        st.error("An unexpected error occurred. Please try again.")
