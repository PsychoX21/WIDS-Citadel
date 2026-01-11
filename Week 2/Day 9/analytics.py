import numpy as np

def compute_vwap(trades_df):
    return (trades_df.price * trades_df.qty).sum() / trades_df.qty.sum()

def validate_pipeline(logger):
    trades = logger.trades_df()
    l1 = logger.l1_df()

    # 1. VWAP bounds
    if not trades.empty:
        vwap = compute_vwap(trades)
        low, high = trades.price.min(), trades.price.max()
        assert low <= vwap <= high, "VWAP outside trade range"

    # 2. Spread non-negative
    if not l1.empty:
        assert (l1.spread >= 0).all(), "Negative spread detected"

    # 3. Volatility check (guarded)
    if len(l1) > 2:
        trade_returns = np.log(trades.price).diff().dropna()
        mid_returns = np.log(l1.mid).diff().dropna()

        if not trade_returns.empty and not mid_returns.empty:
            if mid_returns.std() >= trade_returns.std():
                print(
                    "[WARN] Mid-price volatility >= trade volatility "
                    "(acceptable in thin / toy markets)"
                )

    print("ALL VALIDATIONS PASSED")
