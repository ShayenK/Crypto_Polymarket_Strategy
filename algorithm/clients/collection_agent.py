import time
import copy
import requests
from datetime import datetime
from typing import Optional, Dict, List
from storage.data_attributes import CandleData
from config import (
    SYMBOLS_MAP,
    KLINES_URL,
    MAX_RETRY_ATTEMPTS,
    MAX_LIST_LEN
)

class CollectionAgent:
    def __init__(self):
        self.recent_candle_data:Dict[str,List[CandleData]] = {symbol: [] for symbol, _ in SYMBOLS_MAP.items()}
        self.last_hour:int = -1

    def _reset_recent_candle_data(self) -> None:

        # Reset recent_candle_data state variable
        self.recent_candle_data = {symbol: [] for symbol in SYMBOLS_MAP}

        return
    
    def _collection_check(self) -> bool:

        # Check Collection Timing
        now = datetime.now()
        if now.hour != self.last_hour:
            self.last_hour = now.hour
            return True

        return False
    
    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market (1 hour)

        return int((time.time() // 3600) * 3600)

    def _get_candle_data(self) -> None:

        # Retrieve recent_candle_data (1 hour)
        try:
            for symbol, _ in SYMBOLS_MAP.items():
                unix_time = self._get_recent_market_time()
                itr = 0
                while True:
                    if itr >= MAX_RETRY_ATTEMPTS:
                        print("ERROR: exceeded max attempts, cannot retrieve candles right now")
                        break
                    params = {
                        'symbol': f'{symbol}T',
                        'interval': '1h',
                        'endTime': str((unix_time-1)*1000),
                        'limit': str(MAX_LIST_LEN)
                    }
                    resp = requests.get(KLINES_URL, params, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        for kline in data:
                            candle_data = CandleData(
                                symbol=symbol,
                                time=int(kline[0])/1000,
                                open=float(kline[1]),
                                high=float(kline[2]),
                                low=float(kline[3]),
                                close=float(kline[4]),
                                volume=float(kline[5]),
                                qa_volume=float(kline[7]),
                                num_trades=int(kline[8]),
                                taker_buy_ba_volume=float(kline[9]),
                                taker_buy_qa_volume=float(kline[10]),
                                taker_sell_ba_volume=float(kline[5]) - float(kline[9]),
                                taker_sell_qa_volume=float(kline[7]) - float(kline[10])
                            )
                            self.recent_candle_data[symbol].append(candle_data)
                        print(f"INFO: retrieved candle data for {symbol}")
                        break
                    else:
                        print(resp.status_code, resp)
                    itr += 1
                    time.sleep(1.333*itr)
            return
        except Exception as e:
            print(f"ERROR: unable to request candle data {e}")
            return
        
    def data_collection(self) -> Optional[Dict[str,List[CandleData]]]:
        """
        Data collection method to pull recent historical 1 hour candle data from binance for all symbols

        Args:
            None
        Returns:
            recent_candle_data:Dict[str,List[CandleData]] -> returns most recent klines for MAX_LIST_LEN
        """

        check_1 = self._collection_check()
        if not check_1: return
        self._reset_recent_candle_data()
        self._get_candle_data()
        dict_recent_candle_data = copy.deepcopy(self.recent_candle_data)   # Deepcopy for safety

        return dict_recent_candle_data