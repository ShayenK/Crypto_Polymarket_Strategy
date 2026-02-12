import os
import csv
import functools
from typing import Callable, Dict, List, Any, Type
from dataclasses import fields, astuple
from data_attributes import TradePosition
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
            print(f"INFO: initialized {self.filepath} csv with headers -> {len(self.field_names)} columns")

        return

    def log_trades(self, function:Callable) -> Callable:

        # Wrapper Function for Logging Trades
        @functools.wraps(function)
        def wrapper(*args:Any, **kwargs:Any) -> Any:

            append = function(*args, **kwargs)
            if args:
                with open(file=self.filepath, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    first_arg:Dict[str,List[TradePosition]] = args[0]
                    for symbol in SYMBOLS_MAP.keys():
                        for redeemed_position in first_arg[symbol]:
                            if not redeemed_position.outcome == "COMPLETE": 
                                continue
                            values_tuple = astuple(redeemed_position)
                            writer.writerow(values_tuple)

            return append

        return wrapper
    
    @classmethod
    def emergency_log_trades(cls, filepath:str, in_memory_trade_positions:Dict[str,List[TradePosition]]) -> None:
        """
        Emergency log trade if memory function crashes

        Args:
            in_memory_trade_positions:Dict[str,List[TradePosition]] -> all of the current pending trades stored in-memory
        Returns:
            None
        """

        # Emergency Logging On Sudden Algorithm Crash
        if in_memory_trade_positions:
            with open(file=filepath, mode='a', newline='') as f:
                writer = csv.writer(f)
                for symbol in SYMBOLS_MAP.keys():
                    for in_memory_trade_position in in_memory_trade_positions[symbol]:
                        values_tuple = astuple(in_memory_trade_position)
                        writer.writerow(values_tuple)

        return