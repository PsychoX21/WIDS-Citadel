import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_book(mid_price, tick, depth_levels, base_qty, decay):
    bids, asks = {}, {}

    for i in range(1, depth_levels + 1):
        qty = max(1, int(base_qty * np.exp(-decay * i)))
        bids[mid_price - i * tick] = qty
        asks[mid_price + i * tick] = qty

    return bids, asks


def plot_depth(bids, asks, title):
    bid_prices = sorted(bids.keys(), reverse=True)
    ask_prices = sorted(asks.keys())

    plt.figure()
    plt.barh(bid_prices, [bids[p] for p in bid_prices])
    plt.barh(ask_prices, [asks[p] for p in ask_prices])
    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title)
    plt.show()


# Commodity (illiquid)
bids_c, asks_c = generate_synthetic_book(
    mid_price=100, tick=1, depth_levels=10,
    base_qty=50, decay=0.4
)
plot_depth(bids_c, asks_c, "Synthetic Depth – Commodity (Low Liquidity)")

# Equity (liquid)
bids_e, asks_e = generate_synthetic_book(
    mid_price=100, tick=1, depth_levels=10,
    base_qty=200, decay=0.15
)
plot_depth(bids_e, asks_e, "Synthetic Depth – Equity (High Liquidity)")
