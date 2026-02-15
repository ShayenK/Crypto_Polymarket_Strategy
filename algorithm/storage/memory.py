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

    def add_new_trade_position(self, entered_trade_positions:Optional[Dict[str,TradePosition]]) -> None:
        """
        Add new entry trade position to memory.

        Args:
            entered_trade_positions:Dict[str,TradePosition] -> entry trade position
        Returns:
            None:None -> no output return required
        """

        if not entered_trade_positions or not any(entered_trade_positions.values()):
            return None
        for symbol, position in entered_trade_positions.items():
            if position is not None:
                self.pending_trades[symbol].append(position)

        return
    
    def return_current_positions(self) -> Optional[Dict[str,List[TradePosition]]]:
        """
        Memory operation to retrieve pending trade positions for all symbols.

        Args:
            None:None -> no input arg required
        Returns:
            pending_trade_positions:Dict[str,List[TradePosition]] -> current in-memory positions
        """

        # Returns all the Current Pending Positions In-Memory
        if not any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys()):
            return None
        pending_trade_positions = copy.deepcopy(self.pending_trades)

        return pending_trade_positions
        
    def remove_finished_positions(self, redeemed_trade_positions:Dict[str,List[TradePosition]]) -> None:
        """
        Removes redeemed or unresolved positions from memory.

        Args:
            redeemed_trade_positions:Dict[str,List[TradePosition]] -> modified positions
        Returns:
            None:None -> no output return required
        """
        
        if not redeemed_trade_positions or \
            not any(redeemed_trade_positions[symbol] for symbol in SYMBOLS_MAP.keys()):
            return None
        for symbol in SYMBOLS_MAP.keys():
            redeemed_list = redeemed_trade_positions.get(symbol, [])
            if not redeemed_list:
                continue
            completed_order_ids = {
                pos.order_id for pos in redeemed_list 
                if pos.position_status == "COMPLETE"
            }
            if not completed_order_ids:
                continue
            self.pending_trades[symbol] = [
                pos for pos in self.pending_trades[symbol] 
                if pos.order_id not in completed_order_ids
            ]

        return
    
    def __enter__(self) -> Self:
        return self
    
    def __exit__(self, exc_type:Optional[Type[BaseException]], exc_val:Optional[BaseException],
                 exc_tb:Optional[TracebackType]) -> bool:
        
        # Emergency Logging Procedure In-Case of Abrupt Stop
        has_pending = any(self.pending_trades[symbol] for symbol in SYMBOLS_MAP.keys())
        if exc_type is not None or has_pending:
            self._tracking.emergency_log_trades(self.pending_trades)

        return False