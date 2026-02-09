import copy
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from _5backtest import TradePosition, BacktestPortfolioManager
from _6walkforward import WalkForwardAnalysis
from warnings import filterwarnings

filterwarnings('ignore')

class PermutationResults:
    def __init__(self):
        self.results_map:Dict[str,Any] = {}
        self.base_trade_positions:List[TradePosition] = []
        self.base_probability_variances:List[float] = []
        self.base_equity_curve:List[float] = []
        self.permutation_trade_positions:Dict[int,List[TradePosition]] = {}
        self.permutation_probability_variances:Dict[int,List[float]] = {}
        self.permutation_equity_curves:Dict[int,List[float]] = {}
    
    def add_base_results(self, trade_positions:List[TradePosition], probability_variance:List[float],
                         equity_curve:List[float]) -> None:
        
        # Add Base Walk-Forward Results
        self.base_trade_positions = copy.copy(trade_positions)
        self.base_probability_variances = copy.copy(probability_variance)
        self.base_equity_curve = copy.copy(equity_curve)

        return None
    
    def add_permutation_run(self, permutation_run:int, trade_positions:List[TradePosition], 
                            probability_variance:List[float], equity_curve:List[float]) -> None:
        
        # Add Single Instance
        self.permutation_trade_positions[permutation_run] = trade_positions
        self.permutation_probability_variances[permutation_run] = probability_variance
        self.permutation_equity_curves[permutation_run] = equity_curve

        return None
    
    def _calculate_statistics(self) -> None:
        
        base_final_return = self.base_equity_curve[-1]
        perm_final_return = [curve[-1] for curve in self.permutation_equity_curves.values()]
        better_than_base = sum(1 for r in perm_final_return if r >= base_final_return)
        p_value = (better_than_base + 1) / (len(perm_final_return) + 1)
        self.results_map = {
            'base_return': base_final_return,
            'avg_perm_return': np.mean(perm_final_return),
            'max_perm_return': np.max(perm_final_return),
            'p_value': p_value,
            'n_permutations': len(perm_final_return)
        }
        
        return None
    
    def _print_results(self) -> None:

        # Permutation Analysis
        print("\nPERMUTATION ANALYSIS SUMMARY:")
        print("-----------------------------")
        print(f"Base Strategy Return: {self.results_map['base_return']}")
        print(f"Average Permutated Return: {self.results_map['avg_perm_return']}")
        print(f"Max Permutated Return: {self.results_map['max_perm_return']}")
        print(f"p-value: {self.results_map['p_value']}")
        print("-----------------------------")

        return None

    def _plot_results(self) -> None:

        # Plot Permutations
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 7))
        for label, curve in self.permutation_equity_curves.items():
            plt.plot(curve, color='gray', alpha=0.2, linewidth=0.8)
        plt.plot(self.base_equity_curve, color='gold', linewidth=2.5, label=f'Original Strategy (P={self.results_map["p_value"]:.4f})')
        plt.title(f"Permutation Test: Strategy vs. 100 Random Targets")
        plt.xlabel("Number of Trades (Sequential)")
        plt.ylabel("Equity (USD)")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()
        plt.savefig(f"analysis/results/permutation_equities.png")

        return None
    
    def return_results(self) -> None:

        # Return Results
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return None
    
class PermutationAnalysis:
    def __init__(self, config:Dict[str,Any], model_parameters:Dict[str,Any]):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.walk_forward:WalkForwardAnalysis = WalkForwardAnalysis(
            config, 
            model_parameters
        )
        self.results:PermutationResults = PermutationResults()
        self.df:pd.DataFrame = pd.read_csv(config['filepath'])
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

    def _randomise_dataset(self) -> pd.DataFrame:

        # Randomise Target (ŷ) for each symbol
        randomised_df = self.df.copy()
        for symbol in self.config['symbols']:
            target_col = f'{symbol}_target'
            if target_col in randomised_df.columns:
                randomised_df[target_col] = np.random.permutation(randomised_df[target_col].values)
        
        return randomised_df
    
    def _permutation_engine(self) -> None:

        print("STARTING PERMUTATION ANALYSIS:")
        print("------------------------------")
        print("RUNNING BASELINE WALKFORWARD: ")
        print("------------------------------")
        trade_positions, prob_var, equity = self.walk_forward.run_walkforward_analysis(save=True)
        self.results.add_base_results(trade_positions, prob_var, equity)
        print("------------------------------")

        print("RUNNING PERMUTATION...")
        permutation_runs = self.config['permutation_runs']
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        training_periods = self.config['training_periods']
        testing_periods = self.config['testing_periods']
        walk_forward_periods = self.config['walk_forward_shift_periods']
        first_train_start = start
        first_train_end = first_train_start + pd.DateOffset(months=training_periods)
        first_test_start = first_train_end
        first_test_end = first_test_start + pd.DateOffset(months=testing_periods)
        total_months = (end.year - start.year) * 12 + (end.month - start.month)
        min_months_needed = training_periods + testing_periods
        if total_months < min_months_needed:
            print(f"ERROR: minimum months was not met to start analysis: got {total_months}, need {min_months_needed}")
            return None
        
        # Permutation
        for i in range(1, permutation_runs+1):

            portfolio_manager = BacktestPortfolioManager(self.config['starting_equity'])
            normal_df = self.df.copy()
            randomised_df = self._randomise_dataset()
            non_feature_columns = [
                f'{sym}_{col}'
                for sym in self.config['symbols']
                for col in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            feature_columns = [col for col in normal_df.columns if col not in non_feature_columns]
            target = {symbol: f'{symbol}_target' for symbol in self.config['symbols']}

            curr_period = 0
            curr_shift = 0
            while True:

                curr_period += 1
                walk_train_start = first_train_start + pd.DateOffset(months=curr_shift)
                walk_train_end = walk_train_start + pd.DateOffset(months=training_periods)
                walk_test_start = walk_train_end + pd.DateOffset(minutes=15)
                walk_test_end = walk_test_start + pd.DateOffset(months=testing_periods)
                if walk_test_end > end:
                    break

                # Splice Datasets
                training = randomised_df[walk_train_start:walk_train_end]
                if training.empty: break
                testing = normal_df[walk_test_start:walk_test_end]
                if testing.empty: break
                models:Optional[Dict[str,xgb.XGBClassifier]] = {symbol: None for symbol in self.config['symbols']}

                # Train Models
                for symbol in self.config['symbols']:

                    X = training[feature_columns].values
                    y = training[target[symbol]].values
                    data_len = int(len(X) * 0.8)
                    X_train, y_train = X[:data_len], y[:data_len].flatten()
                    X_validation, y_validation = X[data_len:], y[data_len:].flatten()

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
                    models[symbol] = model
                
                # Testing Model
                pending_entries:Optional[Dict[str,TradePosition]] = {symbol: None for symbol in self.config['symbols']}
                current_positions:Optional[Dict[str,TradePosition]] = {symbol: None for symbol in self.config['symbols']}
                
                for index, row in testing.iterrows():
                    current_close = {symbol: row[f'{symbol}_close'] for symbol in self.config['symbols']}
                    current_open = {symbol: row[f'{symbol}_open'] for symbol in self.config['symbols']}

                    trade_positions = {}

                    for symbol in self.config['symbols']:

                        if pending_entries[symbol]:
                            current_positions[symbol] = pending_entries[symbol]
                            current_positions[symbol].time = int(index.timestamp())
                            current_positions[symbol].entry_price = current_open[symbol]
                            pending_entries[symbol] = None
                        
                        if current_positions[symbol]:
                            pos = current_positions[symbol]
                            pos.exit_price = current_close[symbol]
                            
                            if pos.direction == "UP" and pos.entry_price <= current_close[symbol]:
                                pos.returns = 1
                            elif pos.direction == "DOWN" and pos.entry_price >= current_close[symbol]:
                                pos.returns = 1
                            else:
                                pos.returns = -1
                                
                            trade_positions[symbol] = copy.copy(pos)
                            current_positions[symbol] = None

                        features = row[feature_columns].values.reshape(1, -1)
                        y_pred_proba = models[symbol].predict_proba(features)[0, 1]

                        if y_pred_proba >= self.config['upper_threshold']:
                            pending_entries[symbol] = TradePosition(
                                symbol=symbol,
                                time=None,
                                pred_proba=y_pred_proba,
                                direction="UP",
                                entry_price=None,
                                exit_price=None,
                                returns=None
                            )
                        elif y_pred_proba <= self.config['lower_threshold']:
                            pending_entries[symbol] = TradePosition(
                                symbol=symbol,
                                time=None,
                                pred_proba=y_pred_proba,
                                direction="DOWN",
                                entry_price=None,
                                exit_price=None,
                                returns=None
                            )
                    
                    portfolio_manager.add_trade_position(trade_positions)

                curr_shift += walk_forward_periods

            # Append Walk-Forward Permutation Results
            perm_trade_positions, perm_prob_var, perm_equity = portfolio_manager.return_portfolio_results()
            self.results.add_permutation_run(i, perm_trade_positions, perm_prob_var, perm_equity)
            print(f"COMPLETED: {i}/{permutation_runs}")
        
        # Results
        self.results.return_results()

        return None
    
    def run_permutation_analysis(self) -> None:

        # Run Full Permutation Analysis
        if not self.valid: 
            print("ERROR: invalid datetime periods given")
            return None
        self._permutation_engine()

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
    permutation_engine = PermutationAnalysis(config, model_parameters)
    permutation_engine.run_permutation_analysis()

if __name__ == "__main__":
    main()