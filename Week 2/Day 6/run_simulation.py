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
from agents import RandomAgent, MarketMakerAgent
from events import AgentArrivalEvent, MarketCloseEvent, SnapshotEvent
from sanity_checks import validate_book_snapshot, validate_trades
from analytics import validate_pipeline


def run_simulation(seed=42, simulation_time=500.0):
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    config = MarketConfig(snapshot_interval=1.0)
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(engine, config)

    agents = [
        RandomAgent("R1", arrival_rate=0.6),
        RandomAgent("R2", arrival_rate=0.6),
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

    if not trades.empty:
        trades["time"] = pd.to_datetime(trades.index, unit="s")
        trades.set_index("time", inplace=True)

    if not l1.empty:
        l1["time"] = pd.to_datetime(l1["time"], unit="s")
        l1.set_index("time", inplace=True)

    return trades, l1


def generate_ohlc(trades):
    if trades.empty:
        return None

    ohlc = trades["price"].resample("1min").ohlc()
    volume = trades["qty"].resample("1min").sum()

    ohlc["Volume"] = volume
    ohlc.dropna(inplace=True)

    # mplfinance REQUIRES these exact column names
    ohlc.columns = ["Open", "High", "Low", "Close", "Volume"]

    return ohlc


def export_pdf(ohlc):
    with PdfPages("simulation_report.pdf") as pdf:
        fig, axes = mpf.plot(
            ohlc,
            type="candle",
            volume=True,
            show_nontrading=False,
            returnfig=True,
            figsize=(10, 8)
        )

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
    
    ohlc = generate_ohlc(trades)

    if ohlc is not None:
        assert (ohlc["Low"] <= ohlc["High"]).all()
        export_pdf(ohlc)
        print("simulation_report.pdf generated successfully")
    else:
        print("No trades → skipping OHLC and candlestick plots")
