import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def poisson_agent_market(
    steps=500,
    n_traders=1000,
    lambda_total=20,
    p_limit=0.7,
    cancel_prob=0.02,
    max_depth=10,
    tick=1,
    init_mid=100,
):
    bids, asks = defaultdict(int), defaultdict(int)
    mid = init_mid

    spreads = []
    depth_history = []

    traders = np.arange(n_traders)

    for _ in range(steps):

        arrivals = np.random.poisson(lambda_total)

        for _ in range(arrivals):
            trader = np.random.choice(traders)
            side = np.random.choice(["BUY", "SELL"])

            if np.random.rand() < p_limit:
                offset = np.random.randint(1, max_depth + 1)
                if side == "BUY":
                    bids[mid - offset * tick] += 1
                else:
                    asks[mid + offset * tick] += 1

            else:
                if side == "BUY" and asks:
                    p = min(asks)
                    asks[p] -= 1
                    if asks[p] == 0:
                        del asks[p]

                elif side == "SELL" and bids:
                    p = max(bids)
                    bids[p] -= 1
                    if bids[p] == 0:
                        del bids[p]

        # Random cancellations
        for book in (bids, asks):
            if book and np.random.rand() < cancel_prob:
                p = np.random.choice(list(book.keys()))
                book[p] -= 1
                if book[p] == 0:
                    del book[p]

        # Measure spread and depth
        if bids and asks:
            best_bid = max(bids)
            best_ask = min(asks)

            spread = best_ask - best_bid
            spreads.append(spread)

            mid = (best_bid + best_ask) // 2

            depth_snapshot = defaultdict(int)
            for p, q in bids.items():
                depth_snapshot[p - mid] += q
            for p, q in asks.items():
                depth_snapshot[p - mid] += q

            depth_history.append(depth_snapshot)

    return spreads, depth_history

spreads, depth_history = poisson_agent_market()

plt.figure(figsize=(10, 4))
plt.plot(spreads, alpha=0.7)
plt.xlabel("Event time")
plt.ylabel("Bid–Ask Spread")
plt.title("Spread Dynamics (1000 Poisson Traders)")
plt.show()
