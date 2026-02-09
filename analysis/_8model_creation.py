import copy
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from _5backtest import BacktestEngine
from warnings import filterwarnings
filterwarnings('ignore')

class LiveProductionModels:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.df:pd.DataFrame = pd.read_csv(config['filepath'])
        self.train:Optional[pd.DataFrame] = None
        self.validation:Optional[pd.DataFrame] = None
        self.X_train:Optional[np.ndarray] = None
        self.y_train:Optional[np.ndarray] = None
        self.X_validation:Optional[np.ndarray] = None
        self.y_validation:Optional[np.ndarray] = None
        self.models:Dict[str,xgb.XGBClassifier] = {}
        self.valid = self.__prepare_dataset()

    def __prepare_dataset(self) -> None:
        
        # Prepare Dataset
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
        self.df = self.df.set_index('time')
        self.df = self.df[(self.df.index >= start) & (self.df.index <= end)]
        start_train_date = start
        end_train_date = start_train_date + pd.DateOffset(months=self.config['training_periods'])
        start_test_date = end_train_date + pd.DateOffset(minutes=15)
        end_test_date = start_test_date + pd.DateOffset(months=self.config['testing_periods'])
        
        return end_test_date < end
    
    def _create_model(self) -> None:
        
        # Create Live Production Model
        df = self.df.copy()
        split = int(len(df) * 0.8)
        self.train = df[split:]
        self.validation = df[:split]
        
        for symbol in self.config['symbols']:

            non_feature_columns = [ 
                f'{sym}_{col}'
                for sym in self.config['symbols']
                for col in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            feature_columns = [col for col in self.df.columns if col not in non_feature_columns]
            target = f'{symbol}_target'

            self.X_train = self.train[feature_columns].values
            self.y_train = self.train[target].values.flatten()
            self.X_validation = self.validation[feature_columns].values
            self.y_validation = self.validation[target].values.flatten()

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                **self.model_parameters,
                n_jobs=4,
                verbosity=0
            )
            model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_validation, self.y_validation)],
                verbose=False
            )
            self.models[symbol] = model

        return None
    
    def _basic_model_testing(self) -> None:

        for symbol in self.config['symbols']:
            y_pred_proba = self.models[symbol].predict_proba(self.X_validation)[:, 1]
            y_target = self.y_validation
            upper_threshold = self.config['upper_threshold']
            lower_threshold = self.config['lower_threshold']
            upper_indicies = np.where(y_pred_proba >= upper_threshold)[0]
            lower_indicies = np.where(y_pred_proba <= lower_threshold)[0]
            if len(upper_indicies) > 0:
                upper_preds = np.ones(len(upper_indicies))
                upper_actuals = y_target[upper_indicies]
                upper_accs = (upper_preds == upper_actuals).mean()
                print(f"UPPER THRESHOLDS (>= {upper_threshold})")
                print(f"Signals: {len(upper_indicies)} | Accuracy: {upper_accs:.4f}")
            else: print("INFO: no signals found above upper threshold")
            if len(lower_indicies) > 0:
                lower_preds = np.ones(len(lower_indicies))
                lower_actuals = y_target[lower_indicies]
                lower_accs = (lower_preds == lower_actuals).mean()
                print(f"UPPER THRESHOLDS (>= {lower_threshold})")
                print(f"Signals: {len(lower_indicies)} | Accuracy: {lower_accs:.4f}")
            else: print("INFO: no signals found below lower threshold")

        return None
    
    def create_live_production_model(self, save:Optional[bool]=False) -> None:

        self._create_model()
        self._basic_model_testing()
        if save:
            for symbol in self.config['symbols']:
                joblib.dump(self.models[symbol], filename=f"algorithm/production/{symbol}_live_model.pkl")

        return None
    
def main() -> None:

    config = {
        'filepath': 'analysis/data/crypto_1h_training.csv',
        'start': '2022-01-01',
        'end': '2025-03-30',
        'symbols': ['BTCUSD', 'SOLUSD', 'ETHUSD', 'XRPUSD'],
        'starting_equity': 100,
        'training_periods': 4,
        'testing_periods': 1,
        'walk_forward_shift_periods': 1,
        'upper_threshold': 0.505,
        'lower_threshold': 0.495,
        'permutation_runs': 100
    }
    model_parameters = {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 20,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    engine = LiveProductionModels(config, model_parameters)
    engine.create_live_production_model()

    return None

if __name__ == "__main__":
    main()