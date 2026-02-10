import copy
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class TradePosition:
    symbol:str
    time:float
    pred_proba:float
    direction:str
    entry_price:float
    exit_price:float
    returns:float

class BacktestPortfolioManager:
    def __init__(self, starting_equity:float):
        self.full_trade_positions:List[TradePosition] = []
        self.full_probability_variance:List[float] = []
        self.full_equity:List[float] = [starting_equity]

    def add_trade_position(self, trade_positions:Dict[str,TradePosition]) -> None:
        
        # Add list of Trade Positions (each symbol)
        if not trade_positions: return None
        trade_positions = copy.copy(trade_positions)
        for _, trade_position in trade_positions.items():
            self.full_trade_positions.append(copy.copy(trade_position))
            prob_var = abs(trade_position.pred_proba - 0.5)
            self.full_probability_variance.append(prob_var)
            new_equity = self.full_equity[-1] + trade_position.returns
            self.full_equity.append(new_equity)

        return None
    
    def return_portfolio_results(self) -> Tuple[List[TradePosition],List[float],List[float]]:
        
        # Copy Arrays & Return
        trade_positions = copy.copy(self.full_trade_positions)
        probability_variance = copy.copy(self.full_probability_variance)
        equity_curve = copy.copy(self.full_equity)
        
        return trade_positions, probability_variance, equity_curve

class BacktestResults:
    def __init__(self):
        self.results_map:Dict[str,Any] = {}
        self.trade_positions:List[TradePosition] = []
        self.probability_variances:List[float] = []
        self.equity_curve:List[float] = []

    def _calculate_statistics(self) -> None:

        # Calculate Statistics
        total_trade_returns = [trade.returns for trade in self.trade_positions]
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
            drawdown = (peak - val) / peak
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

        return None
    
    def _print_results(self) -> None:

        # Print Results
        print("BACKTEST RESULTS")
        print("----------------")
        print(f"TOTAL TRADES: {self.results_map['total_trades']}")
        print(f"TOTAL WINS: {self.results_map['total_wins']}")
        print(f"WIN RATE: {self.results_map['win_rate']:.2f}%")
        print(f"TOTAL RETURNS: {self.results_map['total_returns']:.2f}%")
        print(f"MAX CONSEC. WINS: {self.results_map['max_consec_wins']}")
        print(f"MAX CONSEC. LOSSES: {self.results_map['max_consec_losses']}")
        print(f"MAX DRAWDOWN: {self.results_map['max_drawdown']:.2f}%")
        print(f"AVG CONSEC. WINS: {self.results_map['avg_consec_wins']:.2f}")
        print(f"AVG CONSEC LOSSES: {self.results_map['avg_consec_losses']:.2f}")
        print(f"AVG DRAWDOWN: {self.results_map['avg_drawdown']:.2f}%")
        print(f"FULL KELLY FRACTION: {self.results_map['full_kelly']:.2f}%")
        print(f"1/4 KELLY FRACTION: {self.results_map['quarter_kelly']:.2f}%")
        print("----------------")

        return None
    
    def _plot_results(self) -> None:

        # Plot Results
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8), sharex=False)
        fig.subplots_adjust(hspace=0.4)
        ax1.plot(self.equity_curve, color='royalblue', label='Equity')
        ax1.set_title("Strategy Equity Curve")
        ax1.set_ylabel("Price (USD)")
        ax2.set_xlabel("Number of Trades (n)")
        ax1.legend()
        extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f"analysis/results/backtest_equity.png", bbox_inches=extent1.expanded(1.1, 1.2))
        data = np.array(self.probability_variance)
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
        fig.savefig(f"analysis/results/backtest_probability_variances.png", bbox_inches=extent2.expanded(1.1, 1.2))

        plt.tight_layout()
        plt.show()

        return None
    
    def return_results(self, trade_positions:List[TradePosition], probability_variance:List[float], equity_curve:List[float]) -> None:

        # Copy Metrics
        self.trade_positions = copy.copy(trade_positions)
        self.probability_variance = copy.copy(probability_variance)
        self.equity_curve = copy.copy(equity_curve)
        
        # Calculate & Return Results
        self._calculate_statistics()
        self._print_results()
        self._plot_results()

        return None
    
class BacktestEngine:
    def __init__(self, config:str, model_parameters:str):
        self.config:Dict[str,Any] = config
        self.model_parameters:Dict[str,Any] = model_parameters
        self.current_positions:Dict[str, Optional[TradePosition]] = {
            symbol: None for symbol in self.config['symbols']
        }
        self.portfolio_manager:BacktestPortfolioManager = BacktestPortfolioManager(self.config['starting_equity'])
        self.results:BacktestResults = BacktestResults()
        self.df:pd.DataFrame = pd.read_csv(self.config['filepath'])
        self.training_df:Optional[pd.DataFrame] = None
        self.testing_df:Optional[pd.DataFrame] = None
        self.models:Dict[str,xgb.XGBClassifier] = {}
        self.valid:bool = self.__prepare_dataset()

    def __prepare_dataset(self) -> bool:
        
        # Prepare and Initalize Dataset
        start = pd.to_datetime(self.config['start'])
        end = pd.to_datetime(self.config['end'])
        self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
        self.df = self.df.set_index('time')
        self.df = self.df[(self.df.index >= start) & (self.df.index <= end)]
        start_train_date = start
        end_train_date = start_train_date + pd.DateOffset(months=self.config['training_periods'])
        start_test_date = end_train_date + pd.DateOffset(minutes=15)
        end_test_date = start_test_date + pd.DateOffset(months=self.config['testing_periods'])
        self.training_df = self.df.loc[start_train_date:end_train_date].copy()
        self.testing_df = self.df.loc[start_test_date:end_test_date].copy()

        return end_test_date < end
    
    def _train_model(self) -> None:
        
        # Training Function
        train_df = self.training_df.copy()
        train_df = train_df.dropna()
        for symbol in self.config['symbols']:
            non_feature_columns = [
                f'{sym}_{col}'
                for sym in self.config['symbols']
                for col in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            features_columns = [col for col in train_df.columns if col not in non_feature_columns]
            target = f'{symbol}_target'
            X = train_df[features_columns].values
            y = train_df[target].values
            data_len = int(len(X) * 0.8)
            X_train = X[:data_len]
            y_train = y[:data_len].flatten()
            X_validation = X[data_len:]
            y_validation = y[data_len:].flatten()
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
                verbose=True
            )
            self.models[symbol] = model

        return None
    
    def _backtest_engine(self) -> None:

        # Backtest Engine
        test_df = self.testing_df.copy()
        test_df = test_df.dropna()
        non_feature_columns = [
            f'{sym}_{col}'
            for sym in self.config['symbols']
            for col in ['open', 'high', 'low', 'close', 'volume', 'target']
        ]
        feature_columns = [col for col in test_df.columns if col not in non_feature_columns]
        
        # Pending entries per symbol
        pending_entries = {symbol: None for symbol in self.config['symbols']}

        for index, row in test_df.iterrows():
            current_close = {symbol: row[f'{symbol}_close'] for symbol in self.config['symbols']}
            current_open = {symbol: row[f'{symbol}_open'] for symbol in self.config['symbols']}
            
            trade_positions = {}
            
            for symbol in self.config['symbols']:

                # Entry if pending signal for this symbol
                if pending_entries[symbol]:
                    self.current_positions[symbol] = pending_entries[symbol]
                    self.current_positions[symbol].time = int(index.timestamp())
                    self.current_positions[symbol].entry_price = current_open[symbol]
                    pending_entries[symbol] = None
                
                # Exit if in trade for this symbol
                if self.current_positions[symbol]:
                    pos = self.current_positions[symbol]
                    pos.exit_price = current_close[symbol]  # Use specific symbol price
                    
                    if pos.direction == "UP" and pos.entry_price <= current_close[symbol]:
                        pos.returns = 1
                    elif pos.direction == "DOWN" and pos.entry_price >= current_close[symbol]:
                        pos.returns = 1
                    else:
                        pos.returns = -1
                        
                    trade_positions[symbol] = copy.copy(pos)
                    self.current_positions[symbol] = None

                # Generate Entry Probability
                features = row[feature_columns].values.reshape(1, -1)
                y_pred_proba = self.models[symbol].predict_proba(features)[0, 1]

                # Pending Order Generation (per symbol)
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
            
            self.portfolio_manager.add_trade_position(trade_positions)
        
        # Return Results
        trade_pos, prob_var, equity = self.portfolio_manager.return_portfolio_results()
        self.results.return_results(trade_pos, prob_var, equity)

        return None
    
    def run_backtest(self) -> None:

        # Run Single Instance
        if not self.valid:
            print("ERROR: invalid datetime periods given")
            return None
        try:
            self._train_model()
            self._backtest_engine()
            for symbol in self.config['symbols']:
                joblib.dump(self.models[symbol], f'analysis/models/{symbol}_backtested_model.pkl')
        except Exception as e:
            print(f"ERROR: {e}")

        return None
    
    @classmethod
    def run_backtest_alt(cls, training_df:pd.DataFrame, testing_df:pd.DataFrame, config:Dict[str,Any], model_parameters:Dict[str,Any]
                         ) -> Tuple[List[TradePosition],List[float],List[float],Dict[str,xgb.XGBClassifier],List[str]]:
        
        train_df = training_df.dropna().copy()
        models:Optional[Dict[str,xgb.XGBClassifier]] = {symbol: None for symbol in config['symbols']}
        for symbol in config['symbols']:
            non_feature_columns = [
                f'{sym}_{col}'
                for sym in config['symbols']
                for col in ['open', 'high', 'low', 'close', 'volume', 'target']
            ]
            feature_columns = [col for col in train_df.columns if col not in non_feature_columns]
            target = f'{symbol}_target'
            X = train_df[feature_columns].values
            y = train_df[target].values
            data_len = int(len(X) * 0.8)
            X_train = X[:data_len]
            y_train = y[:data_len].flatten()
            X_validation = X[data_len:]
            y_validation = y[data_len:].flatten()
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                **model_parameters,
                n_jobs=4,
                verbosity=0
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_validation, y_validation)],
                verbose=False
            )
            models[symbol] = model

        ## Backtest Model
        portfolio_manager:BacktestPortfolioManager = BacktestPortfolioManager(0)
        pending_entries:Optional[Dict[str,TradePosition]] = {symbol: None for symbol in config['symbols']}
        current_positions:Optional[Dict[str,TradePosition]] = {symbol: None for symbol in config['symbols']}
        test_df:pd.DataFrame = testing_df.dropna().copy()

        for index, row in test_df.iterrows():
            current_close = {symbol: row[f'{symbol}_close'] for symbol in config['symbols']}
            current_open = {symbol: row[f'{symbol}_open'] for symbol in config['symbols']}

            trade_positions = {}
            for symbol in config['symbols']:
                
                # Entry if pending signal for this symbol
                if pending_entries[symbol]:
                    current_positions[symbol] = pending_entries[symbol]
                    current_positions[symbol].time = int(index.timestamp())
                    current_positions[symbol].entry_price = current_open[symbol]
                    pending_entries[symbol] = None
                
                # Exit if in trade for this symbol
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

                # Generate Entry Probability
                features = row[feature_columns].values.reshape(1, -1)
                y_pred_proba = models[symbol].predict_proba(features)[0, 1]

                # Pending Order Generation (per symbol)
                if y_pred_proba >= config['upper_threshold']:
                    pending_entries[symbol] = TradePosition(
                        symbol=symbol,
                        time=None,
                        pred_proba=y_pred_proba,
                        direction="UP",
                        entry_price=None,
                        exit_price=None,
                        returns=None
                    )
                elif y_pred_proba <= config['lower_threshold']:
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
        
        trade_pos, prob_var, equity = portfolio_manager.return_portfolio_results()
    
        return trade_pos, prob_var, equity, models, feature_columns
    
def main():

    config = {
        'filepath': 'analysis/data/crypto_1h_features.csv',
        'start': '2025-07-30',
        'end': '2025-12-31',
        'symbols': ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
        'starting_equity': 100,
        'training_periods': 4,
        'testing_periods': 1,
        'upper_threshold': 0.505,
        'lower_threshold': 0.495,
    }
    model_parameters = {
        'n_estimators': 1000,
        'early_stopping_rounds': 100,
        'learning_rate': 0.01,
        'max_depth': 4,
        'min_child_weight': 20,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    engine = BacktestEngine(config, model_parameters)
    engine.run_backtest()

    return None

if __name__ == "__main__":
    main()