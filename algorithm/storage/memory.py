import copy
from types import TracebackType
from typing import Optional, Dict, List, Type, Self
from storage.tracking import Tracking
from storage.data_attributes import TradePosition
from config import (
    SYMBOLS_MAP,
    TRADE_LOG_FILEPATH
)

class Memory:
    def __init__(self):
        self.pending_trades:Dict[str,List[TradePosition]] = {symbol: [] for symbol in SYMBOLS_MAP.keys()}
        self._tracking = Tracking(f'algorithm/storage/{TRADE_LOG_FILEPATH}', TradePosition)
        self.remove_finished_positions = self._tracking.log_trades(self.remove_finished_positions)

    def add_new_trade_position(self, entered_trade_positions:Dict[str,TradePosition]) -> None:

        # Appends the New Entered Trade Positions In-Memory
        if not entered_trade_positions or not any(entered_trade_positions.values()):
            return
        for symbol, position in entered_trade_positions.items():
            if position is not None:
                self.pending_trades[symbol].append(position)

        return
    
    def return_current_positions(self) -> Optional[Dict[str,List[TradePosition]]]:

        # Returns all the Current Pending Positions In-Memory
        if not any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys()):
            return
        pending_trades = copy.deepcopy(self.pending_trades)

        return pending_trades
        
    def remove_finished_positions(self, redeemed_trade_positions: Dict[str, List[TradePosition]] | None) -> None:
        
        if redeemed_trade_positions is None:
            return
        for symbol in SYMBOLS_MAP.keys():
            redeemed_list = redeemed_trade_positions.get(symbol, [])
            if not redeemed_list:
                continue
            completed_order_ids = {
                pos.order_id
                for pos in redeemed_list
                if pos.position_status == "COMPLETE"
            }
            if not completed_order_ids:
                continue
            before = len(self.pending_trades[symbol])
            self.pending_trades[symbol] = [
                pos for pos in self.pending_trades[symbol]
                if pos.order_id not in completed_order_ids
            ]

        return
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type:Optional[Type[BaseException]], exc_val:Optional[BaseException],
                 exc_tb:Optional[TracebackType]) -> bool:
        
        # Emergency Logging Procedure in-case of Abrupt Stop to System 
        pending_trades = any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys())
        if exc_type is not None or pending_trades:
            if pending_trades:
                print("INFO: activating emergency logging procedure")
                self._tracking.emergency_log_trades(self.pending_trades)

        return False