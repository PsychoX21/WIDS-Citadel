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
        market_state = self.env.get_market_state()
        action = self.agent.get_action(market_state)
        self.env.apply_action(self.agent, action)

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
        self.order.timestamp = engine.time #  Execution time of order and not submission time 
        engine.order_book.submit(self.order)
        
        for t in engine.order_book.trades[prev_trades:]:
            engine.logger.record_trade(t)


class SnapshotEvent(Event):
    def __init__(self, time, env, depth=5):
        super().__init__(time)
        self.env = env
        self.depth = depth

    def execute(self, engine):
        snapshot = engine.order_book.current_snapshot()

        if snapshot.best_bid() is not None and snapshot.best_ask() is not None:
            engine.logger.record_l1(
                engine.time,
                snapshot.best_bid(),
                snapshot.best_ask()
            )
            engine.logger.record_l2(
                engine.time,
                snapshot.bids[:self.depth],
                snapshot.asks[:self.depth]
            )

        if engine.running:
            engine.schedule(
                SnapshotEvent(engine.time + self.env.config.snapshot_interval,
                            self.env,
                            self.depth)
            )

class FairValueUpdateEvent(Event):
    def __init__(self, time, fv_process, dt=1.0):
        super().__init__(time)
        self.fv = fv_process
        self.dt = dt

    def execute(self, engine):
        self.fv.step()
        engine.schedule(
            FairValueUpdateEvent(engine.time + self.dt, self.fv, self.dt)
        )
