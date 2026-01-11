import random
from abc import ABC, abstractmethod
from actions import PlaceLimit, PlaceMarket, Cancel
from collections import deque

# Removed arrival probability as large arrival rate also have same simulation effect

class Agent(ABC):
    def __init__(self, agent_id, arrival_rate=1.0):
        self.agent_id = agent_id
        self.arrival_rate = arrival_rate
        self.balance = 0.0
        self.inventory = 0
        self.active_orders = {}

    def next_event_time(self, current_time):
        return current_time + random.expovariate(self.arrival_rate)

    @abstractmethod
    def get_action(self, market_state):
        pass

    def on_trade(self, trade, side):
        pass


class RandomAgent(Agent):
    def get_action(self, market_state):
        side = random.choice(["BUY", "SELL"])

        if random.random() < 0.5:
            qty = random.randint(1, 5)
            return PlaceMarket(side, qty)

        ref = market_state["mid"] if market_state["mid"] is not None else 100
        price = ref + random.choice([-2, -1, 1, 2])
        qty = random.randint(1, 5)

        return PlaceLimit(side, price, qty)


class MarketMakerAgent(Agent):
    def __init__(
        self,
        agent_id,
        arrival_rate=1.0,
        base_spread=1.0,
        inventory_skew=0.1,
        max_inventory=20,
        cash=100_000
    ):
        super().__init__(agent_id, arrival_rate)
        self.base_spread = base_spread
        self.inventory_skew = inventory_skew
        self.max_inventory = max_inventory
        self.balance = cash

    def get_action(self, market_state):
        mid = market_state["mid"]
        if mid is None:
            mid = 100.0

        # Inventory skew
        skew = self.inventory_skew * self.inventory

        best_bid = market_state["best_bid"]
        best_ask = market_state["best_ask"]

        if best_bid is not None and best_ask is not None:
            bid = max(best_bid + 1, mid - self.base_spread / 2 - skew)
            ask = min(best_ask - 1, mid + self.base_spread / 2 + skew)

            if bid >= ask:
                return None
        else:
            bid = mid - self.base_spread / 2 - skew
            ask = mid + self.base_spread / 2 + skew

        actions = []

        # Cancel old quotes
        for oid in list(self.active_orders):
            actions.append(Cancel(oid))
            del self.active_orders[oid]

        # Place new quotes
        qty = 1

        if self.inventory < self.max_inventory:
            actions.append(PlaceLimit("BUY", bid, qty))

        if self.inventory > -self.max_inventory:
            actions.append(PlaceLimit("SELL", ask, qty))

        return actions

    def on_trade(self, trade, side):
        if side == "BUY":
            self.inventory += trade.qty
            self.balance -= trade.price * trade.qty
        else:
            self.inventory -= trade.qty
            self.balance += trade.price * trade.qty


# no inventory updates
class NoiseTraderAgent(Agent):
    # Zero-Intelligence trader with budget and inventory constraints.

    def __init__(self, agent_id, fair_value_process, arrival_rate=1.0, max_qty=5, cash=10_000):
        super().__init__(agent_id, arrival_rate)
        self.fair_value = fair_value_process
        self.balance = cash
        self.inventory = 10
        self.max_qty = max_qty

    def get_action(self, market_state):
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(1, self.max_qty)

        fv = self.fair_value.get()

        if side == "BUY" and self.balance < fv * qty:
            return None

        if side == "SELL" and self.inventory < qty:
            return None

        # 70% market, 30% aggressive limit
        if random.random() < 0.7:
            return PlaceMarket(side, qty)

        # Aggressive limit near fair value
        price = fv + random.randint(-4, 4)
        return PlaceLimit(side, price, qty)
    
    def on_trade(self, trade, side):
        if side == "BUY":
            self.inventory += trade.qty
            self.balance -= trade.price * trade.qty
        else:
            self.inventory -= trade.qty
            self.balance += trade.price * trade.qty

class MomentumAgent(Agent):
    # Trend following momentum trader using SMA crossover

    def __init__(
        self,
        agent_id,
        window=50,
        arrival_rate=1.0,
        max_qty=5,
        cash=10_000
    ):
        super().__init__(agent_id, arrival_rate)
        self.window = window
        self.prices = deque(maxlen=window)
        self.balance = cash
        self.inventory = 0
        self.max_qty = max_qty

    def get_action(self, market_state):
        mid = market_state["mid"]

        if mid is None:
            return None

        self.prices.append(mid)

        if len(self.prices) < self.window:
            return None # Not enough history

        sma = sum(self.prices) / self.window

        side = "BUY" if mid > sma else "SELL"
        qty = random.randint(1, self.max_qty)

        # Budget / inventory constraints
        if side == "BUY" and self.balance < mid * qty:
            return None

        if side == "SELL" and self.inventory < qty:
            return None

        # Momentum traders are aggressive
        return PlaceMarket(side, qty)
    
    def on_trade(self, trade, side):
        if side == "BUY":
            self.inventory += trade.qty
            self.balance -= trade.price * trade.qty
        else:
            self.inventory -= trade.qty
            self.balance += trade.price * trade.qty