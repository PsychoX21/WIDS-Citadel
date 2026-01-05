from trade import Trade
from snapshot import BookSnapshot

class OrderBook:
    def __init__(self):
        self.bids = [] # list of (-price, timestamp, order)
        self.asks = [] # list of ( price, timestamp, order)
        self.trades = []
        self.snapshots = {}
        self.time = 0

    def submit(self, order):
        self.time += 1
        order.timestamp = self.time
        self._match(order)
        if order.price is not None and order.qty > 0:
            self._add(order)
        self._snapshot(order.order_id)

    def _add(self, order):
        if order.side == "BUY":
            self.bids.append((-order.price, order.timestamp, order))
            self.bids.sort()
        else:
            self.asks.append((order.price, order.timestamp, order))
            self.asks.sort()

    def _match(self, incoming):
        opposite = self.asks if incoming.side == "BUY" else self.bids
        while incoming.qty > 0 and opposite:
            price, _, top = opposite[0]
            best_price = price if incoming.side == "BUY" else -price
            if incoming.price is not None:
                if incoming.side == "BUY" and best_price > incoming.price:
                    break
                if incoming.side == "SELL" and best_price < incoming.price:
                    break
            opposite.pop(0)
            traded = min(incoming.qty, top.qty)
            incoming.qty -= traded
            top.qty -= traded
            self.trades.append(
                Trade(
                    price=best_price,
                    qty=traded,
                    buy_order_id=incoming.order_id if incoming.side == "BUY" else top.order_id,
                    sell_order_id=incoming.order_id if incoming.side == "SELL" else top.order_id,
                )
            )
            if top.qty > 0:
                opposite.insert(0, (price, top.timestamp, top))

    def cancel_random(self, prob):
        import random
        for book in (self.bids, self.asks):
            if book and random.random() < prob:
                book.pop(random.randrange(len(book)))

    def _snapshot(self, order_id):
        self.snapshots[order_id] = BookSnapshot(self.bids, self.asks)

    def book_after(self, order_id):
        return self.snapshots[order_id]
