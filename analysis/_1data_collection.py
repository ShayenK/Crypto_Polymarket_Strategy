import copy
import time
import requests
import pandas as pd
from functools import reduce
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

@dataclass(frozen=True)
class CandleData:
    symbol:str
    time:int
    open:float
    high:float
    low:float
    close:float
    volume:float
    qa_volume:float
    num_trades:float
    taker_buy_ba_volume:float
    taker_buy_qa_volume:float
    taker_sell_ba_volume:float
    taker_sell_qa_volume:float

class DataCollection:
    def __init__(self, symbols:List[str], output_filepath:str):
        self.symbols:List[str] = symbols
        self.output_filepath:str = output_filepath
        self.list_candle_data:Optional[List[CandleData]] = []
        self.df:Optional[pd.DataFrame] = None

    def _reset_candle_data(self) -> None:

        # Reset Candle Data
        self.list_candle_data = []

        return None
        
    def _get_unix_time(self, datetime_str:str) -> int:

        # Convert str to Timestamp
        unix_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S").timestamp()

        return unix_time

    def _fetch_batch(self, symbol:str, unix_timestamp:int) -> None:

        # Retrieve batch of candle data
        unix_time_milli = int(unix_timestamp * 1000)
        url = 'https://api.binance.com/api/v3/klines'
        try:
            params = {
                'symbol': f'{symbol}T',
                'interval': '1h',
                'startTime': str(unix_time_milli),
                'limit': str(1000)
            }
            resp = requests.get(url=url, params=params, timeout=10)
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
                        num_trades=float(kline[8]),
                        taker_buy_ba_volume=float(kline[9]),
                        taker_buy_qa_volume=float(kline[10]),
                        taker_sell_ba_volume=float(kline[5]) - float(kline[9]),
                        taker_sell_qa_volume=float(kline[7]) - float(kline[10])
                    )
                    self.list_candle_data.append(candle_data)
            else:
                print(resp.status_code, resp.json())
                return None
        except Exception as e:
            print(f"INFO: unable to collect candle data batch {e}")

        return None
    
    def _output_dataset(self) -> None:

        # Output Data to CSV
        self.df.to_csv(self.output_filepath, index=False)
        print("INFO: outputed data to csv")

        return None
    
    def collect_candle_data(self, start:str, end:str) -> None:

        try:
            all_symbols_list = []
            for symbol in self.symbols:
                print(f"INFO: fetching for {symbol}")
                
                start_unix = self._get_unix_time(start)
                end_unix = self._get_unix_time(end)
                current_unix = start_unix
                while current_unix < end_unix:

                    self._fetch_batch(symbol, current_unix)
                    if not self.list_candle_data: break
                    current_unix = self.list_candle_data[-1].time+3600
                    time.sleep(1)
                    
                symbol_candle_data = copy.copy(self.list_candle_data)
                symbol_df = pd.DataFrame(symbol_candle_data)
                if not symbol_df.empty:
                    print(f"INFO: collected candle data for {symbol}")
                    symbol_df = symbol_df.drop(columns='symbol')
                    symbol_df = symbol_df.add_prefix(f"{symbol}_")
                    symbol_df = symbol_df.rename(columns={f"{symbol}_time": "time"})
                    all_symbols_list.append(symbol_df)
                self._reset_candle_data()
            
            ## Create Wide DF after
            self.df = reduce(lambda left, right: pd.merge(left, right, on='time', how='inner'), all_symbols_list)
            self.df = self.df.sort_values('time').reset_index(drop=True)
            self._output_dataset()

        except Exception as e:
            print(f"INFO: unable to run collection process {e}")

        return None
    
def main() -> None:
    data = DataCollection(
        ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
        'analysis/data/crypto_1h.csv'
    )

    data.collect_candle_data(
        '2022-01-01 00:00:00',
        '2026-02-07 00:00:00'
    )

    return None

if __name__ == "__main__":
    main()