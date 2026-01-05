class MarketConfig:
    def __init__(
        self,
        tick_size=1,
        lot_size=1,
        cancel_prob=0.05,
    ):
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.cancel_prob = cancel_prob
