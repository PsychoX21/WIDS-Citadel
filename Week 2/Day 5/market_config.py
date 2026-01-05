class MarketConfig:
    def __init__(
        self,
        tick_size=1,
        lot_size=1,
        mean_latency=1.0,
        snapshot_interval=1.0,
    ):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.mean_latency = mean_latency
        self.snapshot_interval = snapshot_interval