import pandas as pd
import mplfinance as mpf
from typing import List

class DataCheck:
    def __init__(self, symbols:List[str], filepath:str):
        self.symbols:List[str] = symbols
        self.df:pd.DataFrame = pd.read_csv(filepath)
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

    def _check_dataset(self) -> None:
        
        # Check Dataset for Missingness / Inf Values / Error Values
        df = self.df.copy()
        df = df.set_index('time')
        total_cols = len(df.columns)
        print(f"INFO: len of columns: {total_cols}")
        missingness = df.isnull().sum()
        if missingness.any(): 
            print(f"INFO: missing values found\n{missingness[missingness > 0]}")
        inf_mask = df.isin([float('inf'), float('-inf')])
        if inf_mask.any().any(): 
            print(f"INFO: infinite values found in columns: {inf_mask.any()[inf_mask.any()].index.tolist()}")
        time_diffs = df.index.to_series().diff()
        unique_jump_len = time_diffs.dropna().unique()
        print(f"\nINFO: unique time intervals found: {len(unique_jump_len)}")
        for interval in unique_jump_len:
            count = (time_diffs == interval).sum()
            print(f"  - {interval}: {count} occurrences")
        if len(unique_jump_len) > 1:
            print("\nWARNING: Inconsistent time intervals detected!")
            print("\nInterval distribution:")
            print(time_diffs.value_counts().sort_index())
            expected_interval = time_diffs.mode().values[0]
            print(f"\nExpected interval: {expected_interval}")
            anomalies = time_diffs[time_diffs != expected_interval].dropna()
            print(f"\nFound {len(anomalies)} anomalous intervals:")
            print("\nTimestamp | Interval | Previous Timestamp")
            print("-" * 70)
            for idx, interval in anomalies.items():
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                print(f"{idx} | {interval} | {prev_idx}")
                if len(anomalies) > 20 and anomalies.index.get_loc(idx) == 19:
                    print(f"... and {len(anomalies) - 20} more")
                    break

        return None
    
    def _visual_inspection(self, start:str, end:str) -> None:

        # Visual Inspection of Values
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        for symbol in self.symbols:
            df = self.df.copy()
            df = df.set_index('time')
            df = df[(df.index >= start) & (df.index <= end)]
            df = df.rename(columns={
                f'{symbol}_open': 'Open',
                f'{symbol}_high': 'High',
                f'{symbol}_low': 'Low',
                f'{symbol}_close': 'Close',
                f'{symbol}_volume': 'Volume',
            })
            ohlc = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            mpf.plot(
                ohlc, 
                type='candle', 
                style='mike', 
                title=f'{symbol} candlestick chart', 
                volume=True
            )

        return None
    
def main() -> None:
    data_check = DataCheck(
        ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
        'analysis/data/crypto_1h_features.csv'
    )

    # 1. Data Check
    data_check._check_dataset()

    # 2. Visual Inspection
    data_check._visual_inspection(
        '2026-01-01',
        '2026-02-10'
    )

    return None

if __name__ == "__main__":
    main()