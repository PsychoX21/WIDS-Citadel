import random
from abc import ABC, abstractmethod
from actions import PlaceLimit, PlaceMarket

# Removed arrival probability as large arrival rate also have same simulation effect

class Agent(ABC):
    def __init__(self, agent_id, arrival_rate=1.0):
        self.agent_id = agent_id
        self.arrival_rate = arrival_rate
        self.balance = 0.0
        self.inventory = 0

    def next_event_time(self, current_time):
        return current_time + random.expovariate(self.arrival_rate)

    @abstractmethod
    def get_action(self, market_state):
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
    def get_action(self, market_state):
        if market_state["mid"] is None:
            return None

        spread = 2
        side = random.choice(["BUY", "SELL"])
        price = market_state["mid"] - spread // 2 if side == "BUY" else market_state["mid"] + spread // 2

        return PlaceLimit(side, price, 3)
