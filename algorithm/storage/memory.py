import copy
from types import TracebackType
from typing import Optional, Dict, List, Type, Self
from tracking import Tracking
from data_attributes import TradePosition
from config import (
    SYMBOLS_MAP,
    TRADE_LOG_FILEPATH
)

class Memory:
    def __init__(self):
        self.pending_trades:Dict[str,List[TradePosition]] = {symbol: [] for symbol in SYMBOLS_MAP.keys()}
        self.trade_logs_filepath:str = f'algorithm/storage/{TRADE_LOG_FILEPATH}'

    def add_new_trade_position(self, entered_trade_positions:Dict[str,TradePosition]) -> None:

        # Appends the New Entered Trade Positions In-Memory
        if not entered_trade_positions:
            return
        for symbol in SYMBOLS_MAP.keys():
            self.pending_trades[symbol].append(entered_trade_positions[symbol])

        return
    
    def return_current_positions(self) -> Optional[Dict[str,List[TradePosition]]]:

        # Returns all the Current Pending Positions In-Memory
        if not any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys()):
            return
        pending_trades = copy.deepcopy(self.pending_trades)

        return pending_trades
    
    def remove_finished_positions(self, redeemed_trade_positions:Dict[str,List[TradePosition]]) -> None:
        
        if not redeemed_trade_positions:
            return
        for symbol in SYMBOLS_MAP.keys():
            completed_order_ids = {
                pos.order_id 
                for pos in redeemed_trade_positions[symbol] 
                if pos.outcome == "COMPLETE"
            }
            self.pending_trades[symbol] = [
                pos for pos in self.pending_trades[symbol]
                if pos.order_id not in completed_order_ids
            ]
        
        return
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type:Optional[Type[BaseException]], exc_val:Optional[BaseException],
                 exc_tb:Optional[TracebackType]) -> None:
        
        # Emergency Logging Procedure in-case of Abrupt Stop to System 
        print("INFO: activating emergency logging procedure")
        if not any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys()):
            return
        Tracking.emergency_log_trades(self.trade_logs_filepath, self.pending_trades)

        return