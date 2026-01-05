import pandas as pd

class Logger:
    def __init__(self):
        self.trades = []
        self.l1 = []
        self.l2 = []

    def record_trade(self, trade):
        self.trades.append({
            "price": trade.price,
            "qty": trade.qty,
            "buy": trade.buy_order_id,
            "sell": trade.sell_order_id
        })

    def record_l1(self, time, bid, ask):
        if bid is None or ask is None:
            return
        self.l1.append({
            "time": time,
            "best_bid": bid,
            "best_ask": ask,
            "spread": ask - bid,
            "mid": (ask + bid) / 2
        })

    def record_l2(self, time, bids, asks):
        self.l2.append({
            "time": time,
            "bids": bids,
            "asks": asks
        })

    def trades_df(self):
        return pd.DataFrame(self.trades)

    def l1_df(self):
        return pd.DataFrame(self.l1)
