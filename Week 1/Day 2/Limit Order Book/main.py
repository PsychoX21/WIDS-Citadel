from order_book import OrderBook
from input_data import ORDERS

book = OrderBook()

for o in ORDERS:
    book.submit(o)
    print(f"After {o.order_id}:")
    print(book.book_after(o.order_id).pretty())
    print("-" * 30)

# print("\nTo see Orderbook after Particular Order")
# print(book.book_after("L4").pretty())

print("\nTRADES")
for t in book.trades:
    print(t)
