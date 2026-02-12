from storage.memory import Memory
from storage.tracking import Tracking
from storage.data_attributes import TradePosition
from config import (
    TRADE_LOG_FILEPATH
)

tracking = Tracking(
    f"algorithm/storage/{TRADE_LOG_FILEPATH}",
    TradePosition
)

with Memory() as memory:

    @tracking.log_trades
    memory.remove_finished_positions