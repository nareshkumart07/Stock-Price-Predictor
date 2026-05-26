from stock_forecaster.data_service import _is_yahoo_non_retryable_message, _stooq_symbol


def test_stooq_symbol_for_us_ticker():
    assert _stooq_symbol("AAPL") == "aapl.us"
    assert _stooq_symbol("msft") == "msft.us"


def test_stooq_symbol_passthrough_for_index_and_suffix():
    assert _stooq_symbol("^GSPC") == "^gspc"
    assert _stooq_symbol("AAPL.US") == "aapl.us"


def test_non_retryable_yahoo_error_detection():
    msg = "YFTzMissingError('$%ticker%: possibly delisted; No timezone found')"
    assert _is_yahoo_non_retryable_message(msg)
