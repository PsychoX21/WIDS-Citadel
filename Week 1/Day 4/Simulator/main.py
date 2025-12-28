from order_book import OrderBook
from market_config import MarketConfig
from environment import MarketEnvironment
from agents import RandomTraderAgent, MarketTakerAgent, MarketMakerAgent
from logger import Logger

book = OrderBook()
config = MarketConfig()
logger = Logger()
env = MarketEnvironment(book, config, logger)

agents = [
    RandomTraderAgent("R1"),
    RandomTraderAgent("R2"),
    MarketTakerAgent("T1"),
    MarketMakerAgent("MM1"),
]

env.reset()

for _ in range(50):
    for agent in agents:
        order = agent.act(env)
        if order:
            env.submit_order(order)
    env.step()

print("TRADES:")
for t in logger.trades:
    print(t)

print("\nFINAL BOOK:")
print(logger.snapshots[-1][1].pretty())
