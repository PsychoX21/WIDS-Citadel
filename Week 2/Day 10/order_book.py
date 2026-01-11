import heapq
from trade import Trade
from snapshot import BookSnapshot

class OrderBook:
    def __init__(self):
        self.bids = [] # list of (-price, timestamp, order)
        self.asks = [] # list of ( price, timestamp, order)
        self.trades = []
        self.snapshots = {}

    def submit(self, order):
        self._match(order)
        if order.price is not None and order.qty > 0:
            self._add(order)
        self._snapshot(order.order_id)

    def _add(self, order):
        if order.side == "BUY":
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

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
            heapq.heappop(opposite)
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
                heapq.heappush(opposite, (price, top.timestamp, top))

    def cancel_random(self, prob):
        import random
        for book in (self.bids, self.asks):
            if book and random.random() < prob:
                book.pop(random.randrange(len(book)))
                heapq.heapify(book)

    def _snapshot(self, order_id):
        self.snapshots[order_id] = BookSnapshot(self.bids, self.asks)

    def current_snapshot(self):
        return BookSnapshot(self.bids, self.asks)

    def book_after(self, order_id):
        return self.snapshots[order_id]
    
    def cancel(self, order_id):
        for book in (self.bids, self.asks):
            book[:] = [x for x in book if x[2].order_id != order_id]
            heapq.heapify(book)
