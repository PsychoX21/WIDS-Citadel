import numpy as np

def compute_vwap(trades_df):
    return (trades_df.price * trades_df.qty).sum() / trades_df.qty.sum()

def validate_pipeline(logger):
    trades = logger.trades_df()
    l1 = logger.l1_df()

    # 1. VWAP bounds
    vwap = compute_vwap(trades)
    low, high = trades.price.min(), trades.price.max()
    assert low <= vwap <= high, "VWAP outside trade range"

    # 2. Spread non-negative
    assert (l1.spread >= 0).all(), "Negative spread detected"

    # 3. Volatility check
    trade_returns = np.log(trades.price).diff().dropna()
    mid_returns = np.log(l1.mid).diff().dropna()

    assert mid_returns.std() < trade_returns.std(), \
        "Mid-price volatility >= trade volatility"

    print("ALL VALIDATIONS PASSED")
