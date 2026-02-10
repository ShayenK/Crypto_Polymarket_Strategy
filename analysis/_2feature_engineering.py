import pandas as pd
import numpy as np
from typing import List
from warnings import filterwarnings
filterwarnings('ignore')

class FeatureEngineering:
    def __init__(self, symbols:List[str], filepath:str):
        self.symbols:List[str] = symbols
        self.df:pd.DataFrame = pd.read_csv(filepath)
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')

    def _engineer_features(self, periods:List[int], save_filepath:str) -> None:

        # Feature Engineering into Dataset
        df = self.df.copy()
        ep = 1e-10
        for symbol in self.symbols:
            # ŷ -> Target Feature
            df[f'{symbol}_target'] = (df[f'{symbol}_close'].shift(-1) > df[f'{symbol}_open'].shift(-1)).astype(int)
            # X -> Input Features
            total_range = df[f'{symbol}_high'] - df[f'{symbol}_low']
            df[f'{symbol}_volume_imbalance'] = (df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']) / (df[f'{symbol}_volume'] + ep)
            df[f'{symbol}_quote_volume_imbalance'] = (df[f'{symbol}_taker_buy_qa_volume'] - df[f'{symbol}_taker_sell_qa_volume']) / (df[f'{symbol}_qa_volume'] + ep)
            df[f'{symbol}_buyer_aggression'] = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_volume'] + ep)
            df[f'{symbol}_seller_aggression'] = df[f'{symbol}_taker_sell_ba_volume'] / (df[f'{symbol}_volume'] + ep)
            df[f'{symbol}_trade_intensity'] = df[f'{symbol}_num_trades'] / (df[f'{symbol}_volume'] + ep)
            df[f'{symbol}_buy_price_impact'] = (df[f'{symbol}_high'] - df[f'{symbol}_open']) / (df[f'{symbol}_taker_buy_ba_volume'] + ep)
            df[f'{symbol}_sell_price_impact'] = (df[f'{symbol}_open'] - df[f'{symbol}_low']) / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
            df[f'{symbol}_net_taker_volume'] = df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']
            df[f'{symbol}_taker_ratio'] = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
            df[f'{symbol}_avg_taker_buy_price_diff'] = df[f'{symbol}_close'] - (df[f'{symbol}_taker_buy_qa_volume'] / (df[f'{symbol}_taker_buy_ba_volume'] + ep))
            df[f'{symbol}_avg_taker_sell_price_diff'] = df[f'{symbol}_close'] - (df[f'{symbol}_taker_sell_qa_volume'] / (df[f'{symbol}_taker_sell_ba_volume'] + ep))
            df[f'{symbol}_avg_trade_size'] = df[f'{symbol}_volume'] / (df[f'{symbol}_num_trades'] + ep)
            df[f'{symbol}_buy_efficiency'] = (df[f'{symbol}_close'] - df[f'{symbol}_open']) / (df[f'{symbol}_taker_buy_ba_volume'] + ep)
            df[f'{symbol}_sell_efficiency'] = (df[f'{symbol}_open'] - df[f'{symbol}_close']) / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
            df[f'{symbol}_vwap_proxy'] = df[f'{symbol}_qa_volume'] / (df[f'{symbol}_volume'] + ep)
            df[f'{symbol}_close_to_vwap'] = (df[f'{symbol}_close'] - df[f'{symbol}_vwap_proxy']) / (df[f'{symbol}_vwap_proxy'] + ep)
            df[f'{symbol}_quote_volume_efficiency'] = (df[f'{symbol}_close'] - df[f'{symbol}_open']) / (df[f'{symbol}_qa_volume'] + ep)
            returns = df[f'{symbol}_close'].pct_change()
            volume_returns = df[f'{symbol}_volume'].pct_change()
            effective_spread_proxy = total_range / (df[f'{symbol}_close'] + ep)
            returns_lagged = returns.shift(1)
            buyer_pct = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_volume'] + ep)
            avg_trade = df[f'{symbol}_volume'] / (df[f'{symbol}_num_trades'] + ep)
            price_change = df[f'{symbol}_close'].diff()
            signed_volume = df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']
            for period in periods:
                ema_period = df[f'{symbol}_close'].ewm(span=period, min_periods=1).mean()
                close_mean = df[f'{symbol}_close'].rolling(period, min_periods=1).mean()
                close_std = df[f'{symbol}_close'].rolling(period, min_periods=1).std()
                roll_cov = returns.rolling(period, min_periods=1).cov(returns_lagged)
                returns_std = returns.rolling(period, min_periods=1).std()
                df[f'{symbol}_open_change_period_{period}'] = (df[f'{symbol}_open'] - df[f'{symbol}_open'].shift(period)) / (df[f'{symbol}_open'].shift(period) + ep)
                df[f'{symbol}_high_change_period_{period}'] = (df[f'{symbol}_high'] - df[f'{symbol}_high'].shift(period)) / (df[f'{symbol}_high'].shift(period) + ep)
                df[f'{symbol}_low_change_period_{period}'] = (df[f'{symbol}_low'] - df[f'{symbol}_low'].shift(period)) / (df[f'{symbol}_low'].shift(period) + ep)
                df[f'{symbol}_close_change_period_{period}'] = (df[f'{symbol}_close'] - df[f'{symbol}_close'].shift(period)) / (df[f'{symbol}_close'].shift(period) + ep)
                df[f'{symbol}_volatility_change_period_{period}'] = (total_range - total_range.shift(period)) / (total_range.shift(period) + ep)
                df[f'{symbol}_distance_from_ema_period_{period}'] = (df[f'{symbol}_close'] - ema_period) / (ema_period + ep)
                df[f'{symbol}_price_zscore_period_{period}'] = (df[f'{symbol}_close'] - close_mean) / (close_std + ep)
                volume_mean = df[f'{symbol}_volume'].rolling(period, min_periods=1).mean()
                volume_long_mean = df[f'{symbol}_volume'].rolling(period * 4, min_periods=1).mean()
                volume_sum = df[f'{symbol}_volume'].rolling(period, min_periods=1).sum()
                volume_weights = df[f'{symbol}_volume'] / (volume_sum + ep)
                df[f'{symbol}_volume_change_period_{period}'] = (df[f'{symbol}_volume'] - df[f'{symbol}_volume'].shift(period)) / (df[f'{symbol}_volume'].shift(period) + ep)
                df[f'{symbol}_volume_burst_period_{period}'] = df[f'{symbol}_volume'] / (volume_long_mean + ep)
                df[f'{symbol}_volume_concentration_period_{period}'] = (
                    (df[f'{symbol}_volume'] ** 2).rolling(period, min_periods=1).sum() /
                    ((volume_sum + ep) ** 2)
                )
                df[f'{symbol}_cumulative_volume_imbalance_period_{period}'] = (
                    (df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']).rolling(period, min_periods=1).sum()
                ) / (volume_sum + ep)
                df[f'{symbol}_buyer_pressure_change_period_{period}'] = buyer_pct - buyer_pct.shift(period)
                df[f'{symbol}_vpin_period_{period}'] = (
                    (df[f'{symbol}_taker_buy_ba_volume'].rolling(period, min_periods=1).sum() -
                    df[f'{symbol}_taker_sell_ba_volume'].rolling(period, min_periods=1).sum()).abs() /
                    (volume_sum + ep)
                )
                vwap_roll = df[f'{symbol}_qa_volume'].rolling(period, min_periods=1).sum() / (volume_sum + ep)
                df[f'{symbol}_vwap_distance_period_{period}'] = (df[f'{symbol}_close'] - vwap_roll) / (vwap_roll + ep)
                df[f'{symbol}_quote_volume_momentum_period_{period}'] = (
                    (df[f'{symbol}_qa_volume'] - df[f'{symbol}_qa_volume'].shift(period)) / (df[f'{symbol}_qa_volume'].shift(period) + ep)
                )
                df[f'{symbol}_avg_trade_size_change_period_{period}'] = (
                    (avg_trade - avg_trade.shift(period)) / (avg_trade.shift(period) + ep)
                )
                df[f'{symbol}_trade_size_volatility_period_{period}'] = (
                    avg_trade.rolling(period, min_periods=1).std() / (avg_trade.rolling(period, min_periods=1).mean() + ep)
                )
                roll_cov_clipped = roll_cov.fillna(0).clip(upper=0)
                df[f'{symbol}_roll_measure_period_{period}'] = 2 * np.sqrt(-roll_cov_clipped)
                df[f'{symbol}_volume_price_corr_period_{period}'] = returns.rolling(period, min_periods=1).corr(volume_returns)
                df[f'{symbol}_amihud_period_{period}'] = (abs(returns) / (df[f'{symbol}_volume'] * df[f'{symbol}_close'] + ep)).rolling(period, min_periods=1).mean()
                df[f'{symbol}_vw_spread_period_{period}'] = (effective_spread_proxy * volume_weights).rolling(period, min_periods=1).sum()
                df[f'{symbol}_vol_norm_volatility_period_{period}'] = returns_std / (np.log(volume_mean + 1.0) + ep)
                df[f'{symbol}_kyle_lambda_period_{period}'] = (
                    price_change.rolling(period, min_periods=1).sum() /
                    (signed_volume.rolling(period, min_periods=1).sum().abs() + ep)
                )

        # Pruning Dataset
        df['time'] = df['time'].astype('int64') // 10**9
        df = df.dropna()

        print("INFO: outputing file...")
        df.to_csv(save_filepath, index=False)
        print("INFO: file output complete")
        self.df = df

        return None

def main() -> None:
    engineering = FeatureEngineering(
        ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
        'analysis/data/crypto_1h.csv'
    )

    # 1. Construct Features and Save to File
    engineering._engineer_features(
        [12, 24, 48, 96, 168],
        'analysis/data/crypto_1h_features.csv'
    )

    return None

if __name__ == "__main__":
    main()