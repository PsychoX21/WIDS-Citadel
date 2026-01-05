import random
from order import Order

class Agent:
    def __init__(self, agent_id, arrival_prob=0.3):
        self.agent_id = agent_id
        self.arrival_prob = arrival_prob

    def act(self, env):
        raise NotImplementedError


class RandomTraderAgent(Agent):
    def act(self, env):
        if random.random() > self.arrival_prob:
            return None
        side = random.choice(["BUY", "SELL"])
        price = random.choice([98, 99, 100, 101, 102])
        qty = random.randint(1, 5)
        return Order(self.agent_id, side, price, qty, 0)


class MarketTakerAgent(Agent):
    def act(self, env):
        if random.random() > self.arrival_prob:
            return None
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(1, 5)
        return Order(self.agent_id, side, None, qty, 0)


class MarketMakerAgent(Agent):
    def act(self, env):
        if random.random() > self.arrival_prob:
            return None
        mid = 100
        spread = 2
        side = random.choice(["BUY", "SELL"])
        price = mid - spread//2 if side == "BUY" else mid + spread//2
        return Order(self.agent_id, side, price, 3, 0)
