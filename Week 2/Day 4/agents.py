import random
from order import Order

# Removed arrival probability as large arrival rate also have same simulation effect

class Agent:
    def __init__(self, agent_id, arrival_rate=1.0):
        self.agent_id = agent_id
        self.arrival_rate = arrival_rate

    def next_event_time(self, current_time):
        return current_time + random.expovariate(self.arrival_rate)

    def act(self, env, current_time):
        raise NotImplementedError


class RandomTraderAgent(Agent):
    def act(self, env, current_time):
        side = random.choice(["BUY", "SELL"])
        price = random.choice([98, 99, 100, 101, 102])
        qty = random.randint(1, 5)
        return Order(self.agent_id, side, price, qty, 0)


class MarketTakerAgent(Agent):
    def act(self, env, current_time):
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(1, 5)
        return Order(self.agent_id, side, None, qty, 0)


class MarketMakerAgent(Agent):
    def act(self, env, current_time):
        mid = 100
        spread = 2
        side = random.choice(["BUY", "SELL"])
        price = mid - spread // 2 if side == "BUY" else mid + spread // 2
        return Order(self.agent_id, side, price, 3, 0)
