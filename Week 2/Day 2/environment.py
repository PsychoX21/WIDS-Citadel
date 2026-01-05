def reset(self):
    self.time = 0

def step(self):
    self.order_book.cancel_random(self.config.cancel_prob)
    self.time += 1

def submit_order(self, order):
    if order.price is not None:
        order.price = round(order.price / self.config.tick_size) * self.config.tick_size
    order.qty = max(self.config.lot_size,
                    (order.qty // self.config.lot_size) * self.config.lot_size)

    prev_trade_count = len(self.order_book.trades)
    self.order_book.submit(order)

    for t in self.order_book.trades[prev_trade_count:]:
        self.logger.record_trade(t)

    self.logger.record_snapshot(self.time,
        self.order_book.book_after(order.order_id))

def observe(self):
    if not self.order_book.snapshots:
        return None
    last = list(self.order_book.snapshots.keys())[-1]
    return self.order_book.book_after(last)
