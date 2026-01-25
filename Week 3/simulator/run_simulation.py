import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from matplotlib.backends.backend_pdf import PdfPages

from order_book import OrderBook
from engine import MarketEngine
from environment import MarketEnvironment
from logger import Logger
from market_config import MarketConfig
from fair_value import FairValueProcess

from agents import NoiseTraderAgent, MarketMakerAgent, MomentumAgent
from events import (
    AgentArrivalEvent,
    MarketCloseEvent,
    SnapshotEvent,
    FairValueUpdateEvent,
)

# ============================================================
# HARD CONSTRAINTS (NON-NEGOTIABLE)
# ============================================================

SEED = 42
SIMULATION_TIME = 1800.0        # 30 minutes sim-time
SNAPSHOT_INTERVAL = 1.0

SCENARIOS = {
    "A": {"noise": 100, "mm": 0,  "momentum": 0},
    "B": {"noise": 80,  "mm": 20, "momentum": 0},
    "C": {"noise": 80,  "mm": 0,  "momentum": 20},
}

# ============================================================
# CORE SIMULATION
# ============================================================

def run_single_scenario(cfg, seed):
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(
        engine,
        MarketConfig(snapshot_interval=SNAPSHOT_INTERVAL)
    )

    fv = FairValueProcess(initial_value=100.0, sigma=0.5, seed=SEED)

    agents = []

    for i in range(cfg["noise"]):
        agents.append(
            NoiseTraderAgent(f"N{i}", fv, arrival_rate=1.2)
        )

    for i in range(cfg["mm"]):
        agents.append(
            MarketMakerAgent(
                f"MM{i}",
                fv,
                arrival_rate=0.2,
                base_spread=1.0,
                inventory_skew=0.2
            )
        )

    for i in range(cfg["momentum"]):
        agents.append(
            MomentumAgent(
                f"M{i}",
                window=50,
                arrival_rate=1.0
            )
        )

    for agent in agents:
        engine.agents[agent.agent_id] = agent
        engine.schedule(
            AgentArrivalEvent(agent.next_event_time(0), agent, env)
        )

    engine.schedule(SnapshotEvent(0, env))
    engine.schedule(FairValueUpdateEvent(0, fv, dt=1.0))
    engine.schedule(MarketCloseEvent(SIMULATION_TIME))

    engine.run()
    return logger

# ============================================================
# METRICS & OHLC
# ============================================================

def extract_metrics(logger):
    l1 = logger.l1_df().set_index("time")

    mid = l1["mid"]
    spread = l1["spread"]

    returns = np.log(mid).diff().dropna()

    return {
        "mid": mid,
        "spread": spread,
        "avg_spread": spread.mean(),
        "volatility": returns.rolling(20).std(),
        "volatility_mean": returns.rolling(20).std().mean(),
    }

def generate_ohlc(trades_df):
    # Create fixed 30-minute candle index
    start = pd.Timestamp("2024-01-01 00:00:00")
    end = start + pd.Timedelta(seconds=SIMULATION_TIME)

    full_index = pd.date_range(start, end, freq="15s")

    if trades_df.empty:
        return None

    df = trades_df.copy()
    df["time"] = start + pd.to_timedelta(df.index, unit="s")
    df.set_index("time", inplace=True)

    ohlc = df["price"].resample("15s").ohlc()
    volume = df["qty"].resample("15s").sum()
    # 🔑 Force full horizon
    ohlc = ohlc.reindex(full_index)
    volume = volume.reindex(full_index)

    # Fill logic
    ohlc["close"] = ohlc["close"].ffill()
    for col in ["open", "high", "low"]:
        ohlc[col] = ohlc[col].fillna(ohlc["close"])

    ohlc["Volume"] = volume.fillna(0)

    ohlc.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"},
        inplace=True
    )

    assert (ohlc["Low"] <= ohlc["High"]).all()

    return ohlc


# ============================================================
# PDF PAGES (LATEX → CODED)
# ============================================================

def page_setup(pdf):
    fig = plt.figure(figsize=(10, 6))
    text = f"""
MARKET REPORT — EXPERIMENTAL SETUP

Simulation length : 1800 time units (30 minutes)
Random seed       : {SEED}
Snapshot interval : {SNAPSHOT_INTERVAL} second
Tick size         : 1
Latency model     : Exponential (mean = 1.0)
Matching engine   : Price–Time Priority
Fair value        : Random walk (σ = 0.5)

Agent Types
-Noise Trader: Zero-intelligence trader using market and aggressive
limit orders
-Market Maker: Posts bid and ask quotes with inventory-dependent
skew
-Momentum Trader: Trend-following agent using SMA crossover

Scenarios
-Scenario A: 100 Noise Traders
-Scenario B: 80 Noise + 20 Market Makers
-Scenario C: 80 Noise + 20 Momentum Traders
"""
    plt.text(0.05, 0.95, text, va="top")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

def page_scenario(pdf, label, metrics, ohlc):
    fig = plt.figure(figsize=(8.5, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 2.6])

    ax_price = fig.add_subplot(gs[0])
    ax_spread = fig.add_subplot(gs[1], sharex=ax_price)
    ax_candle = fig.add_subplot(gs[2])

    # ---- Mid Price ----
    ax_price.plot(metrics["mid"], linewidth=1)
    ax_price.set_title(f"Scenario {label} — Mid Price")
    ax_price.set_ylabel("Price")

    # ---- Spread ----
    ax_spread.plot(metrics["spread"], color="tab:red", linewidth=1)
    ax_spread.set_title("Bid–Ask Spread")
    ax_spread.set_ylabel("Spread")

    # ---- Candlesticks ----
    if ohlc is not None and not ohlc.empty:
        mpf.plot(
            ohlc,
            type="candle",
            ax=ax_candle,
            volume=False,                 # REQUIRED for external axes
            show_nontrading=False,
            style="charles",
            warn_too_much_data=len(ohlc) + 1
        )
        ax_candle.set_title("1-Minute Candlesticks (Trade-Derived)")
        ax_candle.set_ylabel("Price")

    pdf.savefig(fig)
    plt.close(fig)

def page_comparison(pdf, results):
    fig = plt.figure(figsize=(10, 6))

    table = pd.DataFrame({
        "Avg Spread": {k: v["avg_spread"] for k, v in results.items()},
        "Volatility": {k: v["volatility_mean"] for k, v in results.items()},
    }).round(4)

    inequalities = f"""
EXPECTED INEQUALITIES

Spread:
Scenario B < Scenario A < Scenario C

Volatility:
Scenario B < Scenario A < Scenario C
"""

    plt.text(0.05, 0.88, "COMPARATIVE ANALYSIS", fontsize=14, weight="bold")
    plt.text(0.05, 0.72, table.to_string())
    plt.text(0.05, 0.40, inequalities)

    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def page_interpretation(pdf):
    fig = plt.figure(figsize=(10, 6))
    text = """
INTERPRETATION
Liquidity Provision
Market makers continuously supply both sides of the book, converting random order flow into predictable
execution prices. This compresses spreads and dampens volatility without predicting price direction.

Feedback Loops
• Momentum traders amplify trends by reinforcing recent price moves
• This creates endogenous volatility, not noise
• When combined with insufficient liquidity, this leads to instability

Inventory Risk & Stability
Market makers absorb order flow at the cost of inventory risk. Inventory-based quote skew prevents
runaway exposure and enables continuous participation.
Without this mechanism:
• Liquidity vanishes
• Spreads explode
• Price becomes discontinuous
"""
    plt.text(0.05, 0.95, text, va="top")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)

# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    print("\nRunning Week-2 FINAL ecosystem simulation...\n")

    results = {}
    ohlcs = {}

    for i, (label, cfg) in enumerate(SCENARIOS.items()):
        logger = run_single_scenario(cfg, seed=SEED + i * 100)
        results[label] = extract_metrics(logger)
        ohlcs[label] = generate_ohlc(logger.trades_df())

    with PdfPages("simulation_report.pdf") as pdf:
        page_setup(pdf)
        page_scenario(pdf, "A", results["A"], ohlcs["A"])
        page_scenario(pdf, "B", results["B"], ohlcs["B"])
        page_scenario(pdf, "C", results["C"], ohlcs["C"])
        page_comparison(pdf, results)
        page_interpretation(pdf)

    print("simulation_report.pdf generated successfully")

if __name__ == "__main__":
    main()
