import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Optional, Dict, Any
from warnings import filterwarnings
filterwarnings('ignore')

class LiveProductionModels:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.df:pd.DataFrame = pd.read_csv(config['filepath'])
        self.train:Optional[pd.DataFrame] = None
        self.validation:Optional[pd.DataFrame] = None
        self.X_validation:Optional[np.ndarray] = None
        self.validation_targets:Optional[Dict[str,np.ndarray]] = {}
        self.models:Dict[str,xgb.XGBClassifier] = {}
        self.valid = self.__prepare_dataset()

    def __prepare_dataset(self) -> bool:
        
        # Prepare Dataset
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
        self.df = self.df.set_index('time')
        self.df = self.df[(self.df.index >= start) & (self.df.index <= end)]
        
        date_range_months = (end.year - start.year) * 12 + (end.month - start.month)
        if date_range_months != self.config['training_periods']:
            print(f"ERROR: Date range mismatch")
            print(f"  Start: {start.date()}")
            print(f"  End: {end.date()}")
            print(f"  Actual months: {date_range_months}")
            print(f"  Expected months (training_periods): {self.config['training_periods']}")
            return False
        
        return True
        
    def _create_model(self) -> None:
        
        # Create Live Production Model
        df = self.df.copy()
        split = int(len(df) * 0.8)
        self.train = df[:split]
        self.validation = df[split:]
        
        for symbol in self.config['symbols']:

            non_feature_columns = [ 
                f'{sym}_{col}'
                for sym in self.config['symbols']
                for col in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            feature_columns = [col for col in self.df.columns if col not in non_feature_columns]
            target = f'{symbol}_target'

            X_train = self.train[feature_columns].values
            y_train = self.train[target].values.flatten()
            X_validation = self.validation[feature_columns].values
            y_validation = self.validation[target].values.flatten()
            self.validation_targets[symbol] = y_validation

            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                **self.model_parameters,
                n_jobs=4,
                verbosity=0
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_validation, y_validation)],
                verbose=False
            )
            self.models[symbol] = model
        
        self.X_validation = X_validation

        return None

    def _basic_model_testing(self) -> None:

        total_predictions = 0
        total_correct = 0

        for symbol in self.config['symbols']:
            print(f"BASIC MODEL TESTING: {symbol}")
            y_pred_proba = self.models[symbol].predict_proba(self.X_validation)[:, 1]
            y_target = self.validation_targets[symbol]
            upper_threshold = self.config['upper_threshold']
            lower_threshold = self.config['lower_threshold']
            upper_indicies = np.where(y_pred_proba >= upper_threshold)[0]
            lower_indicies = np.where(y_pred_proba <= lower_threshold)[0]
            
            if len(upper_indicies) > 0:
                upper_preds = np.ones(len(upper_indicies))
                upper_actuals = y_target[upper_indicies]
                upper_correct = (upper_preds == upper_actuals).sum()
                upper_accs = upper_correct / len(upper_indicies)
                print(f"UPPER THRESHOLD (>= {upper_threshold})")
                print(f"Signals: {len(upper_indicies)} | Accuracy: {upper_accs:.4f}")
                total_predictions += len(upper_indicies)
                total_correct += upper_correct
            else: 
                print("INFO: no signals found above upper threshold")
                
            if len(lower_indicies) > 0:
                lower_preds = np.zeros(len(lower_indicies))
                lower_actuals = y_target[lower_indicies]
                lower_correct = (lower_preds == lower_actuals).sum()
                lower_accs = lower_correct / len(lower_indicies)
                print(f"LOWER THRESHOLD (<= {lower_threshold})")
                print(f"Signals: {len(lower_indicies)} | Accuracy: {lower_accs:.4f}")
                total_predictions += len(lower_indicies)
                total_correct += lower_correct
            else: 
                print("INFO: no signals found below lower threshold")
        
        print("\nTOTAL SIGNALS:")
        if total_predictions > 0:
            overall_accuracy = total_correct / total_predictions
            print(f"Signals: {total_predictions} | Accuracy: {overall_accuracy:.4f}")
        else:
            print("No signals generated across any symbols")

        return None
    
    def create_live_production_model(self, save:Optional[bool]=False) -> None:

        if not self.valid: 
            print("ERROR: invalid datetime periods given")
            return None
        self._create_model()
        self._basic_model_testing()
        if save:
            for symbol in self.config['symbols']:
                joblib.dump(self.models[symbol], filename=f"algorithm/production/{symbol}_live_model.pkl")

        return None
    
def main() -> None:

    config = {
        'filepath': 'analysis/data/crypto_1h_testing.csv',
        'start': '2025-09-30',
        'end': '2026-01-30',
        'symbols': ['BTCUSD', 'SOLUSD', 'ETHUSD', 'XRPUSD'],
        'starting_equity': 100,
        'training_periods': 4,
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
    engine.create_live_production_model(save=True)

    return None

if __name__ == "__main__":
    main()