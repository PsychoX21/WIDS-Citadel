import random
from events import OrderSubmissionEvent
from order import Order
from actions import PlaceLimit, PlaceMarket, Cancel

class MarketEnvironment:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config

    def get_market_state(self):
        snapshot = self.engine.order_book.current_snapshot()
        return {
            "best_bid": snapshot.best_bid(),
            "best_ask": snapshot.best_ask(),
            "mid": (
                (snapshot.best_bid() + snapshot.best_ask()) / 2
                if snapshot.best_bid() is not None and snapshot.best_ask() is not None
                else None
            ),
            "l2": snapshot
        }

    def apply_action(self, agent, action):
        if action is None:
            return

        if isinstance(action, PlaceLimit):
            import math
            if action.side == "BUY":
                price = math.floor(action.price / self.config.tick_size) * self.config.tick_size
            else:
                price = math.ceil(action.price / self.config.tick_size) * self.config.tick_size

            order = Order(
                order_id=f"{agent.agent_id}-{self.engine.time}",
                side=action.side,
                price=price,
                qty=max(self.config.lot_size, action.qty),
                timestamp=0,
            )

        elif isinstance(action, PlaceMarket):
            order = Order(
                order_id=f"{agent.agent_id}-{self.engine.time}",
                side=action.side,
                price=None,
                qty=max(self.config.lot_size, action.qty),
                timestamp=0,
            )

        elif isinstance(action, Cancel):
            self.engine.order_book.cancel(action.order_id)
            agent.active_orders.pop(action.order_id, None)
            return

        else:
            return

        latency = random.expovariate(1.0 / self.config.mean_latency)
        arrival_time = self.engine.time + latency

        self.engine.schedule(OrderSubmissionEvent(arrival_time, order))

        if isinstance(action, PlaceLimit):
            agent.active_orders[order.order_id] = order.qty
