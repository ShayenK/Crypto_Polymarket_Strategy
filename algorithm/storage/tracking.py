import os
import csv
import functools
from typing import Callable, Dict, List, Any, Type
from dataclasses import fields, astuple
from storage.data_attributes import TradePosition
from config import (
    SYMBOLS_MAP
)

class Tracking:
    def __init__(self, filepath:str, dataclass:Type):
        self.filepath:str = filepath
        self.dataclass:Type = dataclass
        self.field_names = [field.name for field in fields(dataclass)]
        self.__write_csv_headers()

    def __write_csv_headers(self) -> None:

        # Check if Header Rows Exist to Write Headers
        write_headers = False
        if not os.path.exists(self.filepath):
            write_headers = True
        elif os.path.getsize(self.filepath) == 0:
            write_headers = True         
        if write_headers:
            with open(self.filepath, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.field_names)
            print(f"INFO: initialized {self.filepath} with headers -> {len(self.field_names)} columns")

        return

    def log_trades(self, function:Callable) -> Callable:

        # Wrapper Function for Logging Trades
        @functools.wraps(function)
        def wrapper(*args:Any, **kwargs:Any) -> Any:
            
            result = function(*args, **kwargs)
            try:
                if len(args) < 2:
                    return result
                redeemed_trade_positions:Dict[str,List[TradePosition]] = args[1]
                if redeemed_trade_positions is None or not isinstance(redeemed_trade_positions, dict):
                    return result
                with open(self.filepath, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for symbol in SYMBOLS_MAP.keys():
                        for pos in redeemed_trade_positions.get(symbol, []):
                            if pos.position_status == "COMPLETE":
                                writer.writerow(astuple(pos))
            except Exception as e:
                print(f"WARNING: failed to log completed trades: {e}")

            return result

        return wrapper

    def emergency_log_trades(self, in_memory_trade_positions:Dict[str,List[TradePosition]]) -> None:
        """
        Emergency trade logging if memory function crashes

        Args:
            in_memory_trade_positions:Dict[str,List[TradePosition]] -> all of the current pending trades stored in-memory
        Returns:
            None
        """

        # Emergency Logging On Sudden Algorithm Crash
        if in_memory_trade_positions:
            with open(file=self.filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                for symbol in SYMBOLS_MAP.keys():
                    for in_memory_trade_position in in_memory_trade_positions[symbol]:
                        values_tuple = astuple(in_memory_trade_position)
                        writer.writerow(values_tuple)

        return