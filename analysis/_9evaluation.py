import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from algorithm.config import (
    SYMBOLS_MAP
)

class ForwardTestEvaluation:
    def __init__(self, logs_filepath:str):
        self.logs_filepath:str = logs_filepath
        self.symbols:List[str] = [symbol for symbol in SYMBOLS_MAP.keys()]
        self.symbol_list_data:Dict[str,List] = {symbol:[] for symbol in SYMBOLS_MAP.keys()}
        self.log_df:Optional[pd.DataFrame] = None
        self.candle_data_df:Optional[pd.DataFrame] = None
        self.retreival_unix_time:int = 0
        self.df_len:int = 0
        self.equity_curve:List[int] = [100]
        self.returns:List[int] = []
        self.results_map:Dict = {}

    def _load_prediction_logs(self) -> None:
        
        # Load Prediction Logs as Dataframe
        df = pd.read_csv(self.logs_filepath, index_col=0, parse_dates=['datetime'])
        df['unix_time'] = df.index.map(lambda x: int(x.timestamp()))
        self.log_df = df
        self.retreival_unix_time = int(df['unix_time'].iloc[-1])
        self.df_len = int(len(df)) + 1
        
        return

    def _retrieve_candle_data(self) -> None:

        # Retrieve Data for Records
        for symbol in self.symbols:
            try:
                params = {
                    'symbol': str(symbol) + 'T',
                    'interval': '1h',
                    'endTime': str(self.retreival_unix_time * 1000),
                    'limit': str(self.df_len)
                }
                resp = requests.get('https://api.binance.com/api/v3/klines', params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for kline in data:
                        candle_data = {
                            'unix_time': int(kline[0])//1000,
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4])
                        }
                        self.symbol_list_data[symbol].append(candle_data)
                else:
                    print(resp.url)
                    print(resp.json())
                    print(f"ERROR: unable to retrieve candle data")
                    return None
            except Exception as e:
                print(f"ERROR: cannot request candle data {e}")

        dfs = []
        for symbol, candles in self.symbol_list_data.items():
            symbol_df = pd.DataFrame(candles).set_index('unix_time')
            symbol_df = symbol_df.add_prefix(f'{symbol}_')
            dfs.append(symbol_df)
        self.candle_data_df = pd.concat(dfs, axis=1)

        return
    
    def _compare_to_actualised(self) -> None:

        # Evaluate Results
        val = 100
        for _, row in self.log_df.iterrows():
            if row['unix_time'] not in self.candle_data_df.index:
                continue
            candle_row = self.candle_data_df.loc[row['unix_time']]
            for symbol in self.symbols:
                proba = row[symbol]
                if proba >= 0.505:
                    if candle_row[f'{symbol}_close'] > candle_row[f'{symbol}_open']:
                        self.returns.append(1)
                        val += 1
                    else:
                        self.returns.append(-1)
                        val -= 1
                    self.equity_curve.append(val)
                elif proba <= 0.495:
                    if candle_row[f'{symbol}_close'] < candle_row[f'{symbol}_open']:
                        self.returns.append(1)
                        val += 1
                    else:
                        self.returns.append(-1)
                        val -= 1
                    self.equity_curve.append(val)

        return
    
    def _calculate_statistics(self) -> None:

        total_trade_returns = self.returns
        total_win_returns = [r for r in total_trade_returns if r > 0]
        total_loss_returns = [r for r in total_trade_returns if r < 0]
        total_trades = len(total_trade_returns)
        total_wins = len(total_win_returns)
        winrate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        total_returns = ((self.equity_curve[-1] - self.equity_curve[0]) / abs(self.equity_curve[0])) * 100 if self.equity_curve[0] != 0 else 0

        win_streaks, loss_streaks = [], []
        curr_win, curr_loss = 0, 0
        for ret in total_trade_returns:
            if ret > 0:
                curr_win += 1
                if curr_loss > 0: loss_streaks.append(curr_loss)
                curr_loss = 0
            elif ret < 0:
                curr_loss += 1
                if curr_win > 0: win_streaks.append(curr_win)
                curr_win = 0
        if curr_win > 0: win_streaks.append(curr_win)
        if curr_loss > 0: loss_streaks.append(curr_loss)

        peak = self.equity_curve[0]
        drawdowns = []
        for val in self.equity_curve:
            if val > peak: peak = val
            drawdown = (peak - val) / peak if peak != 0 else 0
            drawdowns.append(drawdown * 100)
        max_drawdown = np.max(drawdowns)
        avg_drawdown = np.mean(drawdowns)

        avg_win = np.mean(total_win_returns) if total_win_returns else 0
        avg_loss = abs(np.mean(total_loss_returns)) if total_loss_returns else 1
        win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        w_decimal = winrate / 100
        kelly = w_decimal - ((1 - w_decimal) / win_loss_ratio) if win_loss_ratio != 0 else 0
        full_kelly = max(0, kelly * 100)

        self.results_map = {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'win_rate': winrate,
            'total_returns': total_returns,
            'max_consec_wins': np.max(win_streaks) if win_streaks else 0,
            'max_consec_losses': np.max(loss_streaks) if loss_streaks else 0,
            'avg_consec_wins': np.mean(win_streaks) if win_streaks else 0,
            'avg_consec_losses': np.mean(loss_streaks) if loss_streaks else 0,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'full_kelly': full_kelly,
            'quarter_kelly': full_kelly / 4
        }

        return

    def _print_results(self) -> None:

        print("FORWARD TEST RESULTS")
        print("--------------------")
        print(f"TOTAL TRADES: {self.results_map['total_trades']}")
        print(f"TOTAL WINS: {self.results_map['total_wins']}")
        print(f"WIN RATE: {self.results_map['win_rate']:.2f}%")
        print(f"TOTAL RETURNS: {self.results_map['total_returns']:.2f}%")
        print(f"MAX CONSEC. WINS: {self.results_map['max_consec_wins']}")
        print(f"MAX CONSEC. LOSSES: {self.results_map['max_consec_losses']}")
        print(f"MAX DRAWDOWN: {self.results_map['max_drawdown']:.2f}%")
        print(f"AVG CONSEC. WINS: {self.results_map['avg_consec_wins']:.2f}")
        print(f"AVG CONSEC. LOSSES: {self.results_map['avg_consec_losses']:.2f}")
        print(f"AVG DRAWDOWN: {self.results_map['avg_drawdown']:.2f}%")
        print(f"FULL KELLY FRACTION: {self.results_map['full_kelly']:.2f}%")
        print(f"1/4 KELLY FRACTION: {self.results_map['quarter_kelly']:.2f}%")
        print("--------------------")

        return

    def _plot_results(self) -> None:

        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        fig.subplots_adjust(hspace=0.4)

        ax1.plot(self.equity_curve, color='royalblue', label='Equity')
        ax1.set_title("Strategy Equity Curve")
        ax1.set_ylabel("Cumulative P&L (units)")
        ax1.set_xlabel("Number of Trades (n)")
        ax1.legend()

        data = np.array(self.returns)
        ax2.hist(data, bins=[-1.5, -0.5, 0.5, 1.5], color='royalblue', alpha=0.7, rwidth=0.6)
        ax2.set_xticks([-1, 1])
        ax2.set_xticklabels(['Loss', 'Win'])
        ax2.set_title("Win / Loss Distribution")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        plt.show()

        return
    
    def evaluate_forward_test(self) -> None:

        # Compile together
        self._load_prediction_logs()
        self._retrieve_candle_data()
        self._compare_to_actualised()
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return 
    
def main() -> None:

    # Evaluate WalkForward Statistics
    evalute = ForwardTestEvaluation(
        'algorithm/storage/trade_logs.csv'
    )
    evalute.evaluate_forward_test()

    return

if __name__ == "__main__":
    main()