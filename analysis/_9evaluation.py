import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

@dataclass
class CandleData:
    datetime:str
    unixtime:int
    open:float
    close:float
    result:str

class ForwardTestEvaluation:
    def __init__(self, filepath:str):
        self.symbols:List[str] = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD']
        self.log_df:pd.DataFrame = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.candle_data:Dict[str,Optional[CandleData]] = {symbol:None for symbol in self.symbols}
        self.returns:List[float] = []
        self.equity_curve:List[float] = [100]
        self.probability_variances:List[float] = []
        self.results_map:Dict[str,Any] = {}

    def _reset_candle_data(self) -> None:
        
        # Reset Candle Data
        self.candle_data = {symbol:None for symbol in self.symbols}

        return 
    
    def _convert_unix_time(self, datetime:str) -> int:

        # Convert to Unix Time
        unix_ms = int(pd.Timestamp(datetime).timestamp() * 1000)

        return unix_ms
 
    def _retrieve_candle_data(self, unixtime:int) -> None:

        # Retrieve Candle Data
        for symbol in self.symbols:
            try:
                params = {
                    'symbol': f'{symbol}T',
                    'interval': '1h',
                    'startTime': str(unixtime),
                    'limit': '1'
                }
                resp = requests.get(url='https://api.binance.com/api/v3/klines', params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for kline in data:
                        symbol_candle_data = CandleData(
                            datetime=int(pd.Timestamp(kline[0]//1000).timestamp()),
                            unixtime=int(kline[0]//1000),
                            open=float(kline[1]),
                            close=float(kline[4]),
                            result='up' if float(kline[4]) >= float(kline[1]) else 'down'
                        )
                        self.candle_data[symbol] = symbol_candle_data
                else:
                    print(resp.status_code)
                    print(resp)
            except Exception as e:
                print(f"INFO: retrieve candle data failed {e}")

        return

    def _calculate_statistics(self) -> None:

        # Calculate Statistics
        total_trade_returns = self.returns
        total_win_returns = [r for r in total_trade_returns if r > 0]
        total_loss_returns = [r for r in total_trade_returns if r < 0]
        total_trades = len(total_trade_returns)
        total_wins = len(total_win_returns)
        winrate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        total_returns = ((self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0]) * 100

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

        # Print Results
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

        # Plot Results
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        fig.subplots_adjust(hspace=0.4)

        ax1.plot(self.equity_curve, color='royalblue', label='Equity')
        ax1.set_title("Strategy Equity Curve")
        ax1.set_ylabel("Cumulative P&L (units)")
        ax1.set_xlabel("Number of Trades (n)")
        ax1.legend()
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("analysis/results/forward_test_equity.png", bbox_inches=extent1.expanded(1.1, 1.2))

        if self.probability_variances:
            data = np.array(self.probability_variances)
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            q1, q3 = np.percentile(data, [25, 75])
            ax2.hist(data, bins=50, color='royalblue', alpha=0.7, density=True)
            ax2.axvline(mean_val, color='salmon', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            ax2.axvline(median_val, color='gold', linestyle='-', linewidth=2, label=f'Median: {median_val:.4f}')
            ax2.axvspan(q1, q3, color='gray', alpha=0.2, label='IQR (Q1-Q3)')
            stats_text = f"Std Dev: {std_val:.4f}"
            ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            ax2.set_title("Probability Conviction Distribution (Variance from 0.5)")
            ax2.set_xlabel("Abs(Proba - 0.5)")
            ax2.set_ylabel("Density")
            ax2.legend()
            extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig("analysis/results/forward_test_probability_variances.png", bbox_inches=extent2.expanded(1.1, 1.2))

        plt.tight_layout()
        plt.show()

        return

    def _results(self) -> None:

        # Compile and Display Results
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return

    def evaluate(self) -> None:

        # Evaluate Testing Evaluation
        itr = 0
        for datetime, row in self.log_df.iterrows():
            unixtime = self._convert_unix_time(datetime)
            self._retrieve_candle_data(unixtime)

            for symbol in self.symbols:
                candle_data = self.candle_data[symbol]
                if candle_data is None:
                    continue
                if candle_data.unixtime != unixtime // 1000:
                    print(f"WARNING: candle mismatch for {symbol} at {datetime}")
                    continue
                proba = row[symbol]
                if 0.495 < proba < 0.505:
                    continue
                self.probability_variances.append(abs(proba - 0.5))
                if candle_data.result == 'up':
                    if proba >= 0.505:
                        self.returns.append(1)
                        self.equity_curve.append(self.equity_curve[-1] + 1)
                    elif proba <= 0.495:
                        self.returns.append(-1)
                        self.equity_curve.append(self.equity_curve[-1] - 1)
                elif candle_data.result == 'down':
                    if proba <= 0.495:
                        self.returns.append(1)
                        self.equity_curve.append(self.equity_curve[-1] + 1)
                    elif proba >= 0.505:
                        self.returns.append(-1)
                        self.equity_curve.append(self.equity_curve[-1] - 1)
                
            itr += 1
            print(f"INFO: completed {itr}/{len(self.log_df)}")

            self._reset_candle_data()
            time.sleep(0.05)

        self._results()

        return
    
def main() -> None:

    forward_test = ForwardTestEvaluation(
        'algorithm/storage/trade_logs.csv'
    )
    forward_test.evaluate()

    return

if __name__ == "__main__":
    main()