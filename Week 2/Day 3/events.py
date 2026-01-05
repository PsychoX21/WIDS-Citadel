class Event:
    def __init__(self, time):
        self.time = time

    def execute(self, engine):
        raise NotImplementedError


class AgentArrivalEvent(Event):
    def __init__(self, time, agent, env):
        super().__init__(time)
        self.agent = agent
        self.env = env

    def execute(self, engine):
        order = self.agent.act(self.env, self.time)
        if order:
            self.env.submit_order(order)

        next_time = self.agent.next_event_time(self.time)
        engine.schedule(AgentArrivalEvent(next_time, self.agent, self.env))


class MarketCloseEvent(Event):
    def execute(self, engine):
        engine.running = False

class OrderSubmissionEvent(Event):
    def __init__(self, time, order):
        super().__init__(time)
        self.order = order

    def execute(self, engine):
        prev_trades = len(engine.order_book.trades)
        self.order.timestamp = engine.time
        engine.order_book.submit(self.order)
        
        for t in engine.order_book.trades[prev_trades:]:
            engine.logger.record_trade(t)

        snapshot = engine.order_book.book_after(self.order.order_id)
        engine.logger.record_snapshot(engine.time, snapshot)

