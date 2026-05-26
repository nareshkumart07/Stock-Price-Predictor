import pandas as pd
import plotly.graph_objects as go


def filter_data_by_timeframe(stock_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    end_date = stock_data.index.max()

    if timeframe == "1M":
        start_date = end_date - pd.DateOffset(months=1)
    elif timeframe == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif timeframe == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif timeframe == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = stock_data.index.min()

    return stock_data[start_date:end_date]


def build_price_chart(chart_data: pd.DataFrame, ticker: str, chart_type: str) -> go.Figure:
    fig = go.Figure()

    if chart_type == "Line":
        fig.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"],
                mode="lines",
                name="Close Price",
            )
        )
        title_text = f"How {ticker}'s Price Trended Over Time"
    else:
        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["Open"],
                high=chart_data["High"],
                low=chart_data["Low"],
                close=chart_data["Close"],
                name="Price",
            )
        )
        title_text = f"{ticker}'s Price Chart"

    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
    )

    return fig
