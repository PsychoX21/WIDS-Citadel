from collections import defaultdict

class BookSnapshot:
    def __init__(self, bids, asks):
        self.bids = self._aggregate(bids, reverse=True)
        self.asks = self._aggregate(asks, reverse=False)

    def _aggregate(self, heap, reverse):
        levels = defaultdict(int)
        for _, _, order in heap:
            levels[order.price] += order.qty
        prices = sorted(levels.keys(), reverse=reverse)
        return [(p, levels[p]) for p in prices]

    def pretty(self, depth=5):
        out = ["BIDS:"]
        for p, q in self.bids[:depth]:
            out.append(f"  {p:>6} → {q}")
        out.append("ASKS:")
        for p, q in self.asks[:depth]:
            out.append(f"  {p:>6} → {q}")
        return "\n".join(out)
