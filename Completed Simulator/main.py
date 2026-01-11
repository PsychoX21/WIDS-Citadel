import random
from order_book import OrderBook
from logger import Logger
from engine import MarketEngine
from environment import MarketEnvironment
from market_config import MarketConfig
from agents import RandomAgent, MarketMakerAgent
from events import AgentArrivalEvent, MarketCloseEvent, SnapshotEvent

random.seed(42)

book = OrderBook()
config = MarketConfig()
logger = Logger()
engine = MarketEngine(book, logger)
env = MarketEnvironment(engine, config)

agents = [
    RandomAgent("R1", arrival_rate=0.6),
    RandomAgent("R2", arrival_rate=0.6),
    MarketMakerAgent("MM1", arrival_rate=0.8),
]

SIMULATION_TIME = 50.0

for agent in agents:
    t = agent.next_event_time(0)
    engine.schedule(AgentArrivalEvent(t, agent, env))

engine.schedule(SnapshotEvent(0, env))
engine.schedule(MarketCloseEvent(SIMULATION_TIME))
engine.run()

print("TRADES:")
for t in logger.trades:
    print(t)

print("\nFINAL BOOK:")
print(book.current_snapshot().pretty())
