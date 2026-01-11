def validate_book_snapshot(snapshot):
    for price, qty in snapshot.bids:
        assert price >= 0
        assert qty >= 0

    for price, qty in snapshot.asks:
        assert price >= 0
        assert qty >= 0

    if snapshot.best_bid() is not None and snapshot.best_ask() is not None:
        assert snapshot.best_bid() < snapshot.best_ask()


def validate_trades(trades_df):
    assert (trades_df.price >= 0).all()
    assert (trades_df.qty > 0).all()
