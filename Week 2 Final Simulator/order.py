from dataclasses import dataclass

@dataclass
class Order:
    order_id: str
    side: str
    price: float | None
    qty: int
    timestamp: int
