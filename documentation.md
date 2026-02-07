# XGBoost Directional Prediction Algorithm for BTCUSD, ETHUSD, SOLUSD, XRPUSD: 1 hour Polymarket Contract #


## Objective
- identify if cross-asset correlation is a more accurate predictor of directional betting than single asset
- identify if cross-asset correlation offsets the risk profile of the trades


## Background
- polymarket provides directional betting for the pairs BTCUSD, ETHUSD, SOLUSD, XRPUSD
- XGBoost is great at picking up on relational dependancies


## Overview:

#### Phase 1: Data Manipulation
✅ Data Collection
  * retrive candle data for all 4 assets
  * collate into csv
✅ Feature Engineering
  * Engineer features for all 4 assets
✅ Data Checking
  * visual inspection
  * missingness / error value checks
✅ Data Spliting
  * decide sample split
  * 80/20 split 

#### Phase 2: Model Creation
[ ] Define Model
  * specifications
  * hyperparameters
[ ] Build Infrastructure
  * design mapping
  * testing pipeline
  * backtest, walkforward, permutation

#### Phase 3: Validation
[ ] Walk-Forward
  * strategy generalisation & hyperparameter tuning
  * fold model creation
[ ] Permutation
  * evaluate strategy robustness (statistical significance)
  * analyse random probability density distributions
[ ] Live-Testing
  * build live infrastructure
  * evaluate as extended out sample data (~300)
[ ] Post-Evaluation
  * backtest strategy against live-tested data
  * ensure coherance

#### Phase 4: Implementation
[ ] Soft Implementation
  * implement with full trade execution
  * test over aloted period
  * capital allocation
  * monitor results
[ ] Full Implementation
  * scaling plan
  * monitor results


## Methodology

#### Phase 1: Data Manipulation

=== Data Collection: ===
- collected data from 2022-01-01 to present
- for the following assets: BTCUSD, ETHUSD, SOLUSD, XRPUSD

=== Feature Engineering: ===
- added all previous calculations but for all assets
- all included in one wide dataset
- 144 features per asset

=== Data Check: ===
INFO: len of columns: 576

INFO: unique time intervals found: 2
  - 0 days 01:00:00: 35784 occurrences
  - 0 days 02:00:00: 1 occurrences

WARNING: Inconsistent time intervals detected!

Interval distribution:
time
0 days 01:00:00    35784
0 days 02:00:00        1
Name: count, dtype: int64

Expected interval: 3600000000000 nanoseconds

Found 1 anomalous intervals:

Timestamp | Interval | Previous Timestamp
----------------------------------------------------------------------
2023-03-24 14:00:00 | 0 days 02:00:00 | 2023-03-24 12:00:00

Notes:
- one inconsistant value likely due to market error or binance error
- will leave it in as it is only one value and enkeeps with the temporal element of the dataset required for this strategy

=== Data Split: ===
- training goes from (2022-01-07 11:00:00 -> 2025-04-14 07:00:00)
- testing goes from (2022-01-07 11:00:00 -> 2025-04-14 07:00:00)

#### Phase 2: Model Creation

=== Define Model: ===

Model Specifications Overview:
- XGBoost model for relation dependancies
- using wide dataset for cross-correlational relationships

Model Hyperparameters Overview:
- model recalibration every 2-4 weeks
- model should utilize the following parameters: 
  * colsample_bytree = 0.8
  * n_estimators = 500
  * learning rate = <0.01
  * max_depth = 4-5
  * early_stopping_rounds = 50

=== Build Infrastructure: ===

Divised Infrastructure Overview:
- model creation intertwinded with testing logic
- seperate live creation model script

Plan:
- _5backtest.py -> post-model evaluation
  * contains production ready training, test, and evaluation
- _6walkforward.py -> Full backtest
  * outputs full strategy results for entire periods
  * inherits backtesting engine class
- _7permutation.py -> permutation on synthetic future created dataframes
  * can test entire period with multiple permutations
  * utilised in production phase
- test.py -> forward testing component
  * main trading infrastructure without execution component

Backtest Overview:
inputs:
- df_training_slice -> training_slice for dataset
- df_testing_slice -> testing_slice for dataset
- configuration
- model parameters
outputs:
- equity_curve (testing period only)
- returns_curve (testing period only)
- trade_positions_curve (testing period only)
- model -> trained model
classes:
- trade position
- backtest portfolio manager
- backtest results
- backtest engine

Walk-Forward Overview:
inputs:
- complete_df with start and end date
- configuration
- model parameters
outputs:
- full outcome results
  * equity curve, outcome difference distribution (outcome - prediction)
  * model stability, consistancy, feature importance
classes:
- ^backtest engine
- walkforward portfolio manager
- walkforward results
- walkforward model evaluation
- walkforward engine

Permutation Overview:
inputs:
- model
- complete_df (for randomisation)
- configuration
- model parameters
outputs:
- full distribution of equity curves
- statistical significance
classes:
- ^trade position
- ^backtest portfolio manager
- ^walkforward engine
- permutation
- permutation results

=== Testing Pipeline and Intention: ===

Backtest: 
- intention -> isolate and evaluate post-model performance
- use cases -> composite class for other testing methodologies, post implementation validation
- aims -> evaluate past performance of already-implemented strategies
Walk-Forward:
- intention -> evaluate the entire strategy for a given timeframe
- use cases -> strategy generalization and hyperparameter tuning
- aims -> provide a complete picture of the strategy and its attributes
Permutation:
- intention -> evaluate critical errors (data leakage, overfitting)
- use cases -> evaluating new strategy or parameter set
- aims -> provide statistical robustness

Pipeline:
1. data manipulation
2. walk forward analysis
  * strategy creation
  * hyperparameter tuning
3. permutation analysis
  * evaluation and robustness testing
4. live model creation
  * create live model
  * utilise recalibration logic 
5. live testing
  * evluate as extended out sample testing
  * compare to backtest post production model
6. live production
  * implement and monitor with risk managment metrics

#### Phase 3: Validation

=== Walk Forward Analysis: ===

Hypthosis Testing:
- model consistancy
  * H₀ = walk-forward strategy results fail to maintain profitability against out sample data
  * H₁ = walk-forward strategy results maintain profitability against out sample data
- model stability
  * H₀ = walk-forward strategy features captures low inter-fold correlation between features
  * H₁ = walk-forward strategy features captures high inter-fold correlation between features

Evalutaion Metrics:
- model consistancy:
  * returns -> show consistent profitability accross time
  * win rate -> minimal degradation between in/out sample data 
  * drawdowns -> drawdowns are relatively unchanged between in/out sample data
- model stability
  * top 10 features remain dominant accross folds and between in/out sample data

Results:
- in-sample results:
- out-sample results:

Interpretation:
- model consistancy:
- model stability:

Notes:
- N/A

=== Permutation Analysis: ===

Hypthosis Testing:
- model robustness:
  * H₀ = permutation strategy proves no feature-target relationship within baseline model
  * H₁ = permutation strategy proves feature-target relationship is structure within baseline model

Evalutaion Metrics:
- model robustness:
  * baseline outperforms permutations
  * statistically significant p-value <0.05

Results:
- in-sample results:
- out-of-sample results:

Interpretation:
- model robustness:

Notes:
- observations:
- implications:

=== Forward Testing: ===

Hypthosis Testing:
- model validation
  * H₀ = strategy model and assumptions don't hold true in forward testing
  * H₁ = strategy model and assumptions are validated in forward testing                            

Evalutaion Metrics:
- model validation:
  * returns -> consistant with that of backtest
  * win rate -> at expected levels
  * doesn't exceed fat tailed event: max consec losses, max drawdown period

Implementation:
- model:
- scripts:
  * clients/
    * collection_agent.py
    * trade_entry_agent.py
    * trade_exit_agent.py
  * production/
    * live_model.pkl
  * setup/
    * claim.py
    * set_up_wallet.py
  * storage/
    * data_attributes.py
    * memory.py
    * trade_log.py
  * strategy/
    * engine.py
  * main.py
  * config.py
- design:

Results:

Interpretation:

Notes:
- implications
- observations

=== Evalutation: ===

Hypothesis Testing:
- post results evaluation:
  * H₀ = strategy doesn't conforms to post-production model evaluation 
  * H₁ = strategy conforms to post-production model evaluation

Evaluation Metrics:
- post results evaluation:
  * backtest results align as similar to forward-testing results
  * same/similar equity curve

Results:

Interpretation:

Notes:
- implications
- observations