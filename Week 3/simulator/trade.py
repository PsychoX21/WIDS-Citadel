from dataclasses import dataclass

@dataclass(frozen=True)
class Trade:
    price: float
    qty: int
    buy_order_id: str
    sell_order_id: str
