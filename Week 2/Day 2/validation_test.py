from order import Order
from order_book import OrderBook

book = OrderBook()

book.submit(Order("S1", "SELL", 101, 10, 0))
book.submit(Order("S2", "SELL", 102, 20, 0))
book.submit(Order("S3", "SELL", 103, 30, 0))

book.submit(Order("B1", "BUY", None, 60, 0))

assert len(book.asks) == 0

trades = book.trades

assert len(trades) == 3
assert [t.price for t in trades] == [101, 102, 103]
assert [t.qty for t in trades] == [10, 20, 30]
assert sum(t.qty for t in trades) == 60

print("VALIDATION PASSED")
for t in trades:
    print(t)
