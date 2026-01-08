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
from agents import RandomTraderAgent, MarketTakerAgent, MarketMakerAgent
from events import AgentArrivalEvent, MarketCloseEvent, SnapshotEvent
from sanity_checks import validate_book_snapshot, validate_trades
from analytics import validate_pipeline


def run_simulation(seed=42, simulation_time=50.0):
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    config = MarketConfig(snapshot_interval=1.0)
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(engine, config)

    agents = [
        RandomTraderAgent("R1", arrival_rate=0.6),
        RandomTraderAgent("R2", arrival_rate=0.6),
        MarketTakerAgent("T1", arrival_rate=0.4),
        MarketMakerAgent("MM1", arrival_rate=0.8),
    ]

    for agent in agents:
        t = agent.next_event_time(0)
        engine.schedule(AgentArrivalEvent(t, agent, env))

    engine.schedule(SnapshotEvent(0, env))
    engine.schedule(MarketCloseEvent(simulation_time))
    engine.run()

    return book, logger


def build_dataframes(logger):
    trades = logger.trades_df()
    l1 = logger.l1_df()

    trades["time"] = pd.to_datetime(trades.index, unit="s")
    l1["time"] = pd.to_datetime(l1["time"], unit="s")

    trades.set_index("time", inplace=True)
    l1.set_index("time", inplace=True)

    return trades, l1


def generate_ohlc(trades):
    ohlc = trades["price"].resample("1min").ohlc()
    volume = trades["qty"].resample("1min").sum()
    return ohlc, volume


def export_pdf(trades, l1, ohlc, volume):
    with PdfPages("simulation_report.pdf") as pdf:

        # --- Candlestick + Volume ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        mpf.plot(
            ohlc,
            type="candle",
            ax=axes[0],
            volume=axes[1],
            show_nontrading=False
        )

        axes[0].set_title("Price (Candlestick)")
        axes[1].set_title("Volume")

        pdf.savefig(fig)
        plt.close(fig)

        # --- Spread ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(l1.index, l1["spread"])
        ax.set_title("Spread Over Time")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Mid-price volatility ---
        mid_returns = np.log(l1["mid"]).diff()
        vol = mid_returns.rolling(10).std()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(vol.index, vol)
        ax.set_title("Rolling Mid-Price Volatility")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    book, logger = run_simulation()

    trades, l1 = build_dataframes(logger)

    validate_trades(trades)

    if not l1.empty:
        last_snapshot = book.current_snapshot()
        validate_book_snapshot(last_snapshot)

    validate_pipeline(logger)
    
    ohlc, volume = generate_ohlc(trades)

    assert (ohlc["low"] <= ohlc["high"]).all()

    export_pdf(trades, l1, ohlc, volume)

    print("simulation_report.pdf generated successfully")
