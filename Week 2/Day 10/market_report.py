import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from agents import NoiseTraderAgent, MarketMakerAgent, MomentumAgent
from fair_value import FairValueProcess
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


# Core simulation runner

def run_scenario(agents, seed=42, horizon=500):
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(engine, MarketConfig(snapshot_interval=1.0))

    fair_value = FairValueProcess(initial_value=100.0, sigma=0.5, seed=seed)

    for agent in agents:
        engine.agents[agent.agent_id] = agent
        t0 = agent.next_event_time(0)
        engine.schedule(AgentArrivalEvent(t0, agent, env))

    engine.schedule(SnapshotEvent(0, env))
    engine.schedule(FairValueUpdateEvent(0, fair_value, dt=1.0))
    engine.schedule(MarketCloseEvent(horizon))

    engine.run()
    return logger


# Metric extraction

def extract_metrics(logger):
    l1 = logger.l1_df()
    l1 = l1.set_index("time")

    mid = l1["mid"]
    spread = l1["spread"]
    returns = np.log(mid).diff()

    returns = returns.dropna()
    returns = returns[returns != 0]

    metrics = {
        "mid": mid,
        "spread": spread,
        "volatility": returns.rolling(20).std(),
        "avg_spread": spread.mean(),
        "volatility_mean": returns.rolling(20).std().mean(),
    }

    return metrics


# Plotting helpers

def plot_scenario(pdf, title, metrics, price_ylim):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(metrics["mid"])
    axes[0].set_title(f"{title} — Mid Price")
    axes[0].set_ylim(price_ylim)

    axes[1].plot(metrics["spread"])
    axes[1].set_title("Spread")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# Scenario definitions

def scenario_noise(seed):
    fv = FairValueProcess(seed=seed)
    return [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        NoiseTraderAgent("N3", fv, 1.2),
    ]


def scenario_noise_mm(seed):
    fv = FairValueProcess(seed=seed)
    return [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        NoiseTraderAgent("N3", fv, 1.2),
        MarketMakerAgent("MM1", arrival_rate=0.5),
    ]


def scenario_noise_momentum(seed):
    fv = FairValueProcess(seed=seed)
    return [
        NoiseTraderAgent("N1", fv, 1.2),
        NoiseTraderAgent("N2", fv, 1.2),
        MomentumAgent("M1", window=30, arrival_rate=1.0),
        MomentumAgent("M2", window=30, arrival_rate=1.0),
    ]


# Main report generation

def main():
    SEED = 42
    HORIZON = 500

    print("\nRunning Day-10 ecosystem experiments...\n")

    scenarios = {
        "Scenario A: Noise Only": scenario_noise(SEED),
        "Scenario B: Noise + Market Maker": scenario_noise_mm(SEED),
        "Scenario C: Noise + Momentum": scenario_noise_momentum(SEED),
    }

    results = {}

    for name, agents in scenarios.items():
        logger = run_scenario(agents, seed=SEED, horizon=HORIZON)
        results[name] = extract_metrics(logger)

    all_prices = pd.concat([m["mid"] for m in results.values()])
    price_ylim = (all_prices.min() - 2, all_prices.max() + 2)

    with PdfPages("market_report.pdf") as pdf:
        for name, metrics in results.items():
            plot_scenario(pdf, name, metrics, price_ylim)

    summary = pd.DataFrame({
        "Avg Spread": {k: v["avg_spread"] for k, v in results.items()},
        "Volatility": {k: v["volatility_mean"] for k, v in results.items()},
    })

    print("=== Summary Statistics ===")
    print(summary.round(4))
    print("\nmarket_report.pdf generated successfully")


if __name__ == "__main__":
    main()
