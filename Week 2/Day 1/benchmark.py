import time
import random

from market_config import MarketConfig
from environment import MarketEnvironment
from agents import RandomTraderAgent, MarketTakerAgent, MarketMakerAgent
from logger import Logger

from order_book import OrderBook as HeapOrderBook
from order_book_list import OrderBook as ListOrderBook


def run_simulation(OrderBookClass, num_orders=10000):
    random.seed(42)

    book = OrderBookClass()
    config = MarketConfig(cancel_prob=0.0) # No cancellations for same calculation
    logger = Logger()
    env = MarketEnvironment(book, config, logger)

    agents = [
        RandomTraderAgent("R1", arrival_prob=1.0),
        RandomTraderAgent("R2", arrival_prob=1.0),
        MarketTakerAgent("T1", arrival_prob=1.0),
        MarketMakerAgent("MM1", arrival_prob=1.0),
    ]

    env.reset()

    start = time.perf_counter()

    orders_processed = 0
    while orders_processed < num_orders:
        for agent in agents:
            order = agent.act(env)
            if order:
                env.submit_order(order)
                orders_processed += 1
                if orders_processed >= num_orders:
                    break
        env.step()

    end = time.perf_counter()
    return end - start


if __name__ == "__main__":
    heap_time = run_simulation(HeapOrderBook)
    list_time = run_simulation(ListOrderBook)

    print(f"Heap-based OrderBook time: {heap_time:.4f} seconds")
    print(f"List-based OrderBook time: {list_time:.4f} seconds")
