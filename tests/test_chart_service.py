import pandas as pd

from stock_forecaster.chart_service import filter_data_by_timeframe


def test_filter_data_by_timeframe_keeps_max_range():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=800, freq="D")
    data = pd.DataFrame(
        {
            "Open": range(800),
            "High": range(800),
            "Low": range(800),
            "Close": range(800),
            "Volume": range(800),
        },
        index=dates,
    )

    max_data = filter_data_by_timeframe(data, "Max")
    one_year_data = filter_data_by_timeframe(data, "1Y")
    one_month_data = filter_data_by_timeframe(data, "1M")

    assert len(max_data) == len(data)
    assert len(one_year_data) < len(max_data)
    assert len(one_month_data) < len(one_year_data)
    assert one_month_data.index.max() == data.index.max()
