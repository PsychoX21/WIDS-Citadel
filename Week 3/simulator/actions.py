class Action:
    pass


class PlaceLimit(Action):
    def __init__(self, side, price, qty):
        self.side = side
        self.price = price
        self.qty = qty


class PlaceMarket(Action):
    def __init__(self, side, qty):
        self.side = side
        self.qty = qty


class Cancel(Action):
    def __init__(self, order_id):
        self.order_id = order_id
        # NOTE: Cancels are assumed instantaneous in this model
