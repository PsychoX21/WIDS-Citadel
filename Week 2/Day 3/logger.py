class Logger:
    def __init__(self):
        self.trades = []
        self.snapshots = []

    def record_trade(self, trade):
        self.trades.append(trade)

    def record_snapshot(self, time, snapshot):
        self.snapshots.append((time, snapshot))
