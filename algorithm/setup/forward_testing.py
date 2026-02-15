import os
import csv
import time
from datetime import datetime, timezone
from typing import Dict
from config import (
    SYMBOLS_MAP,
    TRADE_LOG_FILEPATH
)

class ForwardTesting:
    def __init__(self):
        self.filepath = f'algorithm/storage/{TRADE_LOG_FILEPATH}'
        self.__write_csv_headers()

    def __write_csv_headers(self) -> None:

        # Check if Header Rows Exist to Write Headers
        write_headers = False
        if not os.path.exists(self.filepath):
            write_headers = True
        elif os.path.getsize(self.filepath) == 0:
            write_headers = True         
        if write_headers:
            headers = ['datetime'] + [symbol for symbol in SYMBOLS_MAP.keys()]
            with open(self.filepath, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            print(f"INFO: initialized {self.filepath} with headers")

        return

    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market

        return int((time.time() // 3600) * 3600)
    
    def record_probabilities(self, predictions:Dict[str,float]) -> None:

        # Record Probabilities
        if not predictions or not any(predictions[symbol] for symbol in SYMBOLS_MAP.keys()):
            return None
        unix_time = self._get_recent_market_time()
        human_time = datetime.fromtimestamp(unix_time, tz=timezone.utc)
        results = [human_time] + [predictions[symbol] for symbol in SYMBOLS_MAP.keys()]
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results)
        
        return