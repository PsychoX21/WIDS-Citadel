import heapq

class MarketEngine:
    def __init__(self, order_book, logger):
        self.order_book = order_book
        self.logger = logger
        self.time = 0
        self.event_queue = []
        self.seq = 0
        self.running = True
        self.agents = {}

    def schedule(self, event):
        heapq.heappush(
            self.event_queue,
            (event.time, self.seq, event)
        )
        self.seq += 1

    def run(self):
        while self.event_queue and self.running:
            event_time, _, event = heapq.heappop(self.event_queue)
            self.time = event_time
            event.execute(self)
