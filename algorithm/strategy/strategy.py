import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import reduce
from dataclasses import asdict
from typing import Optional, Dict, List
from storage.data_attributes import CandleData
from config import (
    SYMBOLS_MAP,
    MODEL_FILEPATHS,
    STRATEGY_PERIODS
)

class StrategyEngine:
    def __init__(self):
        self.model_filepath:Dict[str,str] = {symbol: f'algorithm/production/{MODEL_FILEPATHS[symbol]}' for symbol in SYMBOLS_MAP.keys()}
        self.models:Dict[str,xgb.XGBClassifier] = {symbol:None for symbol in SYMBOLS_MAP.keys()}
        self.df:Optional[pd.DataFrame] = None
        self.prediction_row:Optional[np.ndarray] = None
        self.__load_models()

    def __load_models(self) -> None:

        # Load All Pre-Trained Models
        for symbol in SYMBOLS_MAP.keys():
            self.models[symbol] = joblib.load(self.model_filepath[symbol])

        return
    
    def _reset_dataframe(self):

        # Reset DF
        self.df = None

        return
    
    def _reset_prediction_row(self):

        # Reset Prediction Rows
        self.prediction_row = None

        return
    
    def _prepare_dataframe(self, dict_candle_data:Dict[str,List[CandleData]]) -> bool:

        # Prepare the DF
        df_list = []
        for symbol in SYMBOLS_MAP.keys():
            try:
                symbol_data = [asdict(candle_data) for candle_data in dict_candle_data[symbol]]
                df = pd.DataFrame(symbol_data)
                df = df.add_prefix(f'{symbol}_')
                df['time'] = pd.to_datetime(df[f'{symbol}_time'], unit='s')
                df = df.drop(columns=[f'{symbol}_time', f'{symbol}_symbol'])
                df_list.append(df)
            except Exception as e:
                print("ERROR: unable to set up dataframe")
                return False
        combined = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), df_list)
        combined = combined[['time'] + [col for col in combined.columns if col != 'time']]          # reorder column names
        self.df = combined

        return True
    
    def _calculate_features(self) -> bool:
        try:

            df = self.df.copy()
            ep = 1e-14
            new_columns = {}
            for symbol in SYMBOLS_MAP.keys():
                total_range = df[f'{symbol}_high'] - df[f'{symbol}_low']
                returns = df[f'{symbol}_close'].pct_change()
                volume_returns = df[f'{symbol}_volume'].pct_change()
                effective_spread_proxy = total_range / (df[f'{symbol}_close'] + ep)
                returns_lagged = returns.shift(1)
                buyer_pct = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_volume'] + ep)
                avg_trade = df[f'{symbol}_volume'] / (df[f'{symbol}_num_trades'] + ep)
                price_change = df[f'{symbol}_close'].diff()
                signed_volume = df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']
                new_columns[f'{symbol}_target'] = (df[f'{symbol}_close'].shift(-1) > df[f'{symbol}_open'].shift(-1)).astype(int)
                new_columns[f'{symbol}_volume_imbalance'] = (df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']) / (df[f'{symbol}_volume'] + ep)
                new_columns[f'{symbol}_quote_volume_imbalance'] = (df[f'{symbol}_taker_buy_qa_volume'] - df[f'{symbol}_taker_sell_qa_volume']) / (df[f'{symbol}_qa_volume'] + ep)
                new_columns[f'{symbol}_buyer_aggression'] = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_volume'] + ep)
                new_columns[f'{symbol}_seller_aggression'] = df[f'{symbol}_taker_sell_ba_volume'] / (df[f'{symbol}_volume'] + ep)
                new_columns[f'{symbol}_trade_intensity'] = df[f'{symbol}_num_trades'] / (df[f'{symbol}_volume'] + ep)
                new_columns[f'{symbol}_buy_price_impact'] = (df[f'{symbol}_high'] - df[f'{symbol}_open']) / (df[f'{symbol}_taker_buy_ba_volume'] + ep)
                new_columns[f'{symbol}_sell_price_impact'] = (df[f'{symbol}_open'] - df[f'{symbol}_low']) / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
                new_columns[f'{symbol}_net_taker_volume'] = df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']
                new_columns[f'{symbol}_taker_ratio'] = df[f'{symbol}_taker_buy_ba_volume'] / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
                new_columns[f'{symbol}_avg_taker_buy_price_diff'] = df[f'{symbol}_close'] - (df[f'{symbol}_taker_buy_qa_volume'] / (df[f'{symbol}_taker_buy_ba_volume'] + ep))
                new_columns[f'{symbol}_avg_taker_sell_price_diff'] = df[f'{symbol}_close'] - (df[f'{symbol}_taker_sell_qa_volume'] / (df[f'{symbol}_taker_sell_ba_volume'] + ep))
                new_columns[f'{symbol}_avg_trade_size'] = df[f'{symbol}_volume'] / (df[f'{symbol}_num_trades'] + ep)
                new_columns[f'{symbol}_buy_efficiency'] = (df[f'{symbol}_close'] - df[f'{symbol}_open']) / (df[f'{symbol}_taker_buy_ba_volume'] + ep)
                new_columns[f'{symbol}_sell_efficiency'] = (df[f'{symbol}_open'] - df[f'{symbol}_close']) / (df[f'{symbol}_taker_sell_ba_volume'] + ep)
                new_columns[f'{symbol}_vwap_proxy'] = df[f'{symbol}_qa_volume'] / (df[f'{symbol}_volume'] + ep)
                new_columns[f'{symbol}_close_to_vwap'] = (df[f'{symbol}_close'] - new_columns[f'{symbol}_vwap_proxy']) / (new_columns[f'{symbol}_vwap_proxy'] + ep)
                new_columns[f'{symbol}_quote_volume_efficiency'] = (df[f'{symbol}_close'] - df[f'{symbol}_open']) / (df[f'{symbol}_qa_volume'] + ep)
                for period in STRATEGY_PERIODS:
                    ema_period = df[f'{symbol}_close'].ewm(span=period, min_periods=1).mean()
                    close_mean = df[f'{symbol}_close'].rolling(period, min_periods=1).mean()
                    close_std = df[f'{symbol}_close'].rolling(period, min_periods=1).std()
                    roll_cov = returns.rolling(period, min_periods=1).cov(returns_lagged)
                    returns_std = returns.rolling(period, min_periods=1).std()
                    volume_mean = df[f'{symbol}_volume'].rolling(period, min_periods=1).mean()
                    volume_long_mean = df[f'{symbol}_volume'].rolling(period * 4, min_periods=1).mean()
                    volume_sum = df[f'{symbol}_volume'].rolling(period, min_periods=1).sum()
                    volume_weights = df[f'{symbol}_volume'] / (volume_sum + ep)
                    vwap_roll = df[f'{symbol}_qa_volume'].rolling(period, min_periods=1).sum() / (volume_sum + ep)
                    roll_cov_clipped = roll_cov.fillna(0).clip(upper=0)
                    new_columns[f'{symbol}_open_change_period_{period}'] = (df[f'{symbol}_open'] - df[f'{symbol}_open'].shift(period)) / (df[f'{symbol}_open'].shift(period) + ep)
                    new_columns[f'{symbol}_high_change_period_{period}'] = (df[f'{symbol}_high'] - df[f'{symbol}_high'].shift(period)) / (df[f'{symbol}_high'].shift(period) + ep)
                    new_columns[f'{symbol}_low_change_period_{period}'] = (df[f'{symbol}_low'] - df[f'{symbol}_low'].shift(period)) / (df[f'{symbol}_low'].shift(period) + ep)
                    new_columns[f'{symbol}_close_change_period_{period}'] = (df[f'{symbol}_close'] - df[f'{symbol}_close'].shift(period)) / (df[f'{symbol}_close'].shift(period) + ep)
                    new_columns[f'{symbol}_volatility_change_period_{period}'] = (total_range - total_range.shift(period)) / (total_range.shift(period) + ep)
                    new_columns[f'{symbol}_distance_from_ema_period_{period}'] = (df[f'{symbol}_close'] - ema_period) / (ema_period + ep)
                    new_columns[f'{symbol}_price_zscore_period_{period}'] = (df[f'{symbol}_close'] - close_mean) / (close_std + ep)
                    new_columns[f'{symbol}_volume_change_period_{period}'] = (df[f'{symbol}_volume'] - df[f'{symbol}_volume'].shift(period)) / (df[f'{symbol}_volume'].shift(period) + ep)
                    new_columns[f'{symbol}_volume_burst_period_{period}'] = df[f'{symbol}_volume'] / (volume_long_mean + ep)
                    new_columns[f'{symbol}_volume_concentration_period_{period}'] = (df[f'{symbol}_volume'] ** 2).rolling(period, min_periods=1).sum() / ((volume_sum + ep) ** 2)
                    new_columns[f'{symbol}_cumulative_volume_imbalance_period_{period}'] = ((df[f'{symbol}_taker_buy_ba_volume'] - df[f'{symbol}_taker_sell_ba_volume']).rolling(period, min_periods=1).sum()) / (volume_sum + ep)
                    new_columns[f'{symbol}_buyer_pressure_change_period_{period}'] = buyer_pct - buyer_pct.shift(period)
                    new_columns[f'{symbol}_vpin_period_{period}'] = (df[f'{symbol}_taker_buy_ba_volume'].rolling(period, min_periods=1).sum() - df[f'{symbol}_taker_sell_ba_volume'].rolling(period, min_periods=1).sum()).abs() / (volume_sum + ep)
                    new_columns[f'{symbol}_vwap_distance_period_{period}'] = (df[f'{symbol}_close'] - vwap_roll) / (vwap_roll + ep)
                    new_columns[f'{symbol}_quote_volume_momentum_period_{period}'] = (df[f'{symbol}_qa_volume'] - df[f'{symbol}_qa_volume'].shift(period)) / (df[f'{symbol}_qa_volume'].shift(period) + ep)
                    new_columns[f'{symbol}_avg_trade_size_change_period_{period}'] = (avg_trade - avg_trade.shift(period)) / (avg_trade.shift(period) + ep)
                    new_columns[f'{symbol}_trade_size_volatility_period_{period}'] = avg_trade.rolling(period, min_periods=1).std() / (avg_trade.rolling(period, min_periods=1).mean() + ep)
                    new_columns[f'{symbol}_roll_measure_period_{period}'] = 2 * np.sqrt(-roll_cov_clipped)
                    new_columns[f'{symbol}_volume_price_corr_period_{period}'] = returns.rolling(period, min_periods=1).corr(volume_returns)
                    new_columns[f'{symbol}_amihud_period_{period}'] = (abs(returns) / (df[f'{symbol}_volume'] * df[f'{symbol}_close'] + ep)).rolling(period, min_periods=1).mean()
                    new_columns[f'{symbol}_vw_spread_period_{period}'] = (effective_spread_proxy * volume_weights).rolling(period, min_periods=1).sum()
                    new_columns[f'{symbol}_vol_norm_volatility_period_{period}'] = returns_std / (np.log(volume_mean + 1.0) + ep)
                    new_columns[f'{symbol}_kyle_lambda_period_{period}'] = price_change.rolling(period, min_periods=1).sum() / (signed_volume.rolling(period, min_periods=1).sum().abs() + ep)
            self.df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
            return True
            
        except Exception as e:
            print(f"ERROR: unable to engineer features {e}")
            
        return False
    
    def _prepare_prediction_row(self) -> bool:

        # Transform DF to Feature Set
        try:
            non_feature_columns = ['time'] + [
                f'{symbol}_{column}' for symbol in SYMBOLS_MAP.keys()
                for column in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            feature_columns = [col for col in self.df.columns if col not in non_feature_columns]
            self.prediction_row = self.df.iloc[-1][feature_columns].values
            return True
        except Exception as e:
            print(f"ERROR: unable to prepare prediction rows {e}")

        return False
    
    def _predictions(self) -> Optional[Dict[str,float]]:

        # Use Trained Model on Current Feature Set
        try:
            predictions = {symbol:0.50000000 for symbol in SYMBOLS_MAP.keys()}
            pred_row = self.prediction_row.reshape(1,-1)
            for symbol in SYMBOLS_MAP.keys():
                predictions[symbol] = self.models[symbol].predict_proba(pred_row)[0,1]
                print(f"INFO: retrieved prediction for {symbol}")
            return predictions
        except Exception as e:
            print(f"ERROR: unable to generate predictions {e}")

        return None
    
    def model_predictions(self, dict_candle_data:Dict[str,List[CandleData]]) -> Optional[Dict[str,float]]:
        """
        Function to generate prediction probabilities to guide trading decisions

        Args:
            dict_candle_data:Dict[str,List[CandleData]] -> candle data for all symbols
        Returns:
            predictions:Dict[str,float] -> predictions for all symbols
        """

        if not dict_candle_data:
            return
        self._reset_dataframe()
        self._reset_prediction_row()
        check_1 = self._prepare_dataframe(dict_candle_data)
        if not check_1:
            return
        check_2 = self._calculate_features()
        if not check_2:
            return
        check_3 = self._prepare_prediction_row()
        if not check_3:
            return
        predictions = self._predictions()

        return predictions