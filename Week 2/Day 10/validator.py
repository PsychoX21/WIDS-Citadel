# Day 9 Validation System

import numpy as np

from agents import MarketMakerAgent, NoiseTraderAgent, MomentumAgent
from fair_value import FairValueProcess


# Basic metrics

def mean_spread(logger):
    l1 = logger.l1_df()
    if l1.empty:
        raise RuntimeError("No L1 snapshots recorded")
    return l1["spread"].mean()


def final_mid_price(logger):
    l1 = logger.l1_df()
    if l1.empty:
        raise RuntimeError("No L1 snapshots recorded")
    return l1["mid"].iloc[-1]


def compute_pnl(agent, mid_price, initial_cash):
    return agent.balance + agent.inventory * mid_price - initial_cash


# Simulation runner

def run_scenario(agents, seed=42, horizon=500):
    from order_book import OrderBook
    from engine import MarketEngine
    from environment import MarketEnvironment
    from logger import Logger
    from market_config import MarketConfig
    from events import (
        AgentArrivalEvent,
        MarketCloseEvent,
        SnapshotEvent,
        FairValueUpdateEvent,
    )

    import random
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(engine, MarketConfig(snapshot_interval=1.0))

    fair_value = FairValueProcess(100.0, sigma=0.0, seed=seed)

    for agent in agents:
        engine.agents[agent.agent_id] = agent
        engine.schedule(AgentArrivalEvent(agent.next_event_time(0), agent, env))

    engine.schedule(SnapshotEvent(0, env))
    engine.schedule(FairValueUpdateEvent(0, fair_value, dt=1.0))
    engine.schedule(MarketCloseEvent(horizon))
    engine.run()

    return engine.agents, logger


# Validation tests

def test_spread_tightening():
    print("\n[TEST] Spread tightening")

    fv = FairValueProcess()

    agents_no_mm = [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
    ]

    _, logger_no_mm = run_scenario(agents_no_mm)
    spread_no_mm = mean_spread(logger_no_mm)

    agents_with_mm = agents_no_mm + [
        MarketMakerAgent("MM1", arrival_rate=0.6)
    ]

    _, logger_mm = run_scenario(agents_with_mm)
    spread_mm = mean_spread(logger_mm)

    print(f"Mean spread without MM: {spread_no_mm:.4f}")
    print(f"Mean spread with MM   : {spread_mm:.4f}")

    assert spread_mm < spread_no_mm, "Spread did not tighten with market maker"


def test_inventory_mean_reversion():
    print("\n[TEST] Inventory mean reversion")

    fv = FairValueProcess()

    mm = MarketMakerAgent("MM1", arrival_rate=0.6, inventory_skew=0.2)
    agents = [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        mm,
    ]

    agents_dict, _ = run_scenario(agents)
    mm = agents_dict["MM1"]

    print(f"Final inventory: {mm.inventory}")

    assert abs(mm.inventory) < 0.5 * mm.max_inventory, \
        "Inventory is drifting instead of mean-reverting"


def test_inventory_risk_increases_without_skew():
    print("\n[TEST] Inventory risk increases without skew")

    fv = FairValueProcess(100.0, sigma=0.0)

    mm_with_skew = MarketMakerAgent("MM_skew", 0.6, inventory_skew=0.2)
    mm_without_skew = MarketMakerAgent("MM_noskew", 0.6, inventory_skew=0.0)

    agents_skew = [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        mm_with_skew,
    ]

    agents_noskew = [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        mm_without_skew,
    ]

    _, logger_skew = run_scenario(agents_skew)
    _, logger_noskew = run_scenario(agents_noskew)

    inv_skew = logger_skew.inventory_df()
    inv_noskew = logger_noskew.inventory_df()

    std_skew = inv_skew[inv_skew.agent == "MM_skew"]["inventory"].std()
    std_noskew = inv_noskew[inv_noskew.agent == "MM_noskew"]["inventory"].std()

    print(f"Inventory std with skew   : {std_skew:.3f}")
    print(f"Inventory std without skew: {std_noskew:.3f}")

    assert std_noskew > std_skew, \
        "Inventory risk did not increase when skew was removed"


def test_pnl_from_spread():
    print("\n[TEST] PnL source = spread")

    fv = FairValueProcess()

    mm = MarketMakerAgent("MM1", arrival_rate=0.6)
    agents = [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        mm,
    ]

    agents_dict, logger = run_scenario(agents)
    mm = agents_dict["MM1"]

    mid = final_mid_price(logger)
    pnl = compute_pnl(mm, mid, initial_cash=100_000)

    print(f"Final PnL (flat market): {pnl:.2f}")

    assert pnl > 0, "Market maker failed to earn spread in flat market"


def test_momentum_stress():
    print("\n[TEST] Momentum trader stress")

    fv = FairValueProcess()

    mm = MarketMakerAgent("MM1", arrival_rate=0.6)
    agents = [
        MomentumAgent("M1", window=20, arrival_rate=1.0),
        MomentumAgent("M2", window=20, arrival_rate=1.0),
        mm,
    ]

    agents_dict, _ = run_scenario(agents)
    mm = agents_dict["MM1"]

    print(f"Inventory under momentum stress: {mm.inventory}")

    assert abs(mm.inventory) < mm.max_inventory, \
        "Market maker inventory exploded under momentum pressure"


def test_multi_mm_competition():
    print("\n[TEST] Multi-market-maker competition")

    fv = FairValueProcess()

    agents_two_mm = [
        NoiseTraderAgent("N1", fv, 1.2),
        MarketMakerAgent("MM1", 0.8),
        MarketMakerAgent("MM2", 0.8),
    ]

    _, logger_two = run_scenario(agents_two_mm)
    spread_two = mean_spread(logger_two)

    agents_one_mm = [
        NoiseTraderAgent("N1", fv, 1.2),
        MarketMakerAgent("MM1", 0.8),
    ]

    _, logger_one = run_scenario(agents_one_mm)
    spread_one = mean_spread(logger_one)

    print(f"Single MM spread: {spread_one:.4f}")
    print(f"Two MM spread   : {spread_two:.4f}")

    assert spread_two < spread_one, \
        "Competition did not narrow the spread"




if __name__ == "__main__":
    print("\n================ DAY-9 VALIDATION ================")

    test_spread_tightening()
    test_inventory_mean_reversion()
    test_inventory_risk_increases_without_skew()
    test_pnl_from_spread()
    test_momentum_stress()
    test_multi_mm_competition()

    print("\nALL DAY-9 VALIDATIONS PASSED")
