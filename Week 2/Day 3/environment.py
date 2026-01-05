import random
from events import OrderSubmissionEvent

class MarketEnvironment:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def submit_order(self, order):
        if order.price is not None:
            order.price = round(order.price / self.config.tick_size) * self.config.tick_size
        order.qty = max(
            self.config.lot_size,
            (order.qty // self.config.lot_size) * self.config.lot_size
        )
        # Simulating Server latency not Agent latency
        latency = random.expovariate(1.0 / self.config.mean_latency)
        arrival_time = self.engine.time + latency

        self.engine.schedule(OrderSubmissionEvent(arrival_time, order))