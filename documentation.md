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
✅ Define Model
  * specifications
  * hyperparameters
✅ Build Infrastructure
  * design mapping
  * testing pipeline
  * backtest, walkforward, permutation

#### Phase 3: Validation
✅ Walk-Forward
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
- training (2022-01-07 11:00:00 -> 2025-04-14 07:00:00)
- testing (2025-04-14 08:00:00 -> 2026-02-06 13:00:00)

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
  * H₁ = walk-forward strategy results maintain profitability against out sample data           ✅
- model stability
  * H₀ = walk-forward strategy features captures low inter-fold correlation between features    ✅
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
  BACKTEST RESULTS
  ----------------
  TOTAL TRADES: 88503
  TOTAL WINS: 47519
  WIN RATE: 53.69%
  TOTAL RETURNS: 6535.00%
  MAX CONSEC. WINS: 22
  MAX CONSEC. LOSSES: 29
  MAX DRAWDOWN: 43.41%
  AVG CONSEC. WINS: 2.63
  AVG CONSEC LOSSES: 2.27
  AVG DRAWDOWN: 1.39%
  FULL KELLY FRACTION: 7.38%
  1/4 KELLY FRACTION: 1.85%
  ----------------
  MODEL EVALUATION: BTCUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 34
  TOP 20 FEATURES (BY AVG GAIN):
  BTCUSD_vwap_distance_period_12             13.325464
  BTCUSD_price_zscore_period_12              13.106827
  BTCUSD_avg_taker_sell_price_diff           12.806227
  BTCUSD_close_to_vwap                       12.397202
  BTCUSD_avg_taker_buy_price_diff            11.993665
  ETHUSD_vwap_distance_period_12             11.725910
  BTCUSD_buy_efficiency                      11.673234
  ETHUSD_avg_taker_buy_price_diff            11.556929
  ETHUSD_avg_taker_sell_price_diff           11.369646
  ETHUSD_close_to_vwap                       11.351045
  BTCUSD_distance_from_ema_period_12         11.216954
  BTCUSD_sell_efficiency                     11.206537
  BTCUSD_volume_imbalance                    11.193518
  ETHUSD_sell_efficiency                     10.932463
  BTCUSD_quote_volume_efficiency             10.924618
  SOLUSD_avg_taker_buy_price_diff            10.872423
  BTCUSD_net_taker_volume                    10.696952
  ETHUSD_price_zscore_period_12              10.621894
  BTCUSD_buyer_pressure_change_period_168    10.598793
  BTCUSD_avg_trade_size_change_period_48     10.568162
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0634
  MODEL EVALUATION: ETHUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 34
  TOP 20 FEATURES (BY AVG GAIN):
  ETHUSD_price_zscore_period_12         12.645857
  BTCUSD_price_zscore_period_12         12.530995
  BTCUSD_avg_taker_buy_price_diff       12.526382
  BTCUSD_close_to_vwap                  12.387353
  BTCUSD_avg_taker_sell_price_diff      12.113890
  BTCUSD_vwap_distance_period_12        12.024712
  ETHUSD_vwap_distance_period_12        11.949704
  ETHUSD_avg_taker_sell_price_diff      11.892491
  ETHUSD_quote_volume_efficiency        11.866060
  ETHUSD_close_to_vwap                  11.552969
  ETHUSD_buy_efficiency                 11.399676
  ETHUSD_kyle_lambda_period_12          11.292932
  BTCUSD_quote_volume_efficiency        11.277681
  ETHUSD_sell_efficiency                11.265483
  BTCUSD_sell_efficiency                11.254340
  ETHUSD_distance_from_ema_period_12    11.128010
  ETHUSD_avg_taker_buy_price_diff       10.904837
  ETHUSD_price_zscore_period_24         10.902135
  BTCUSD_buy_efficiency                 10.886974
  ETHUSD_volume_imbalance               10.820421
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0516
  MODEL EVALUATION: SOLUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 34
  TOP 20 FEATURES (BY AVG GAIN):
  BTCUSD_avg_taker_sell_price_diff           11.984945
  BTCUSD_sell_efficiency                     11.459774
  ETHUSD_quote_volume_efficiency             11.244408
  SOLUSD_vwap_distance_period_12             11.143407
  SOLUSD_avg_taker_buy_price_diff            11.039710
  ETHUSD_price_zscore_period_12              10.978004
  SOLUSD_buy_efficiency                      10.933869
  SOLUSD_avg_taker_sell_price_diff           10.921482
  BTCUSD_buy_efficiency                      10.864264
  BTCUSD_close_to_vwap                       10.854300
  BTCUSD_buyer_pressure_change_period_168    10.853134
  ETHUSD_avg_taker_buy_price_diff            10.794321
  BTCUSD_buy_price_impact                    10.788043
  SOLUSD_close_to_vwap                       10.774807
  ETHUSD_kyle_lambda_period_12               10.743128
  ETHUSD_amihud_period_12                    10.677082
  ETHUSD_avg_taker_sell_price_diff           10.632784
  SOLUSD_price_zscore_period_12              10.554637
  BTCUSD_price_zscore_period_12              10.539696
  SOLUSD_volume_imbalance                    10.535263
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0443
  MODEL EVALUATION: XRPUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 34
  TOP 20 FEATURES (BY AVG GAIN):
  XRPUSD_distance_from_ema_period_12              13.596631
  XRPUSD_price_zscore_period_12                   12.848462
  XRPUSD_vwap_distance_period_12                  12.414846
  XRPUSD_avg_taker_sell_price_diff                12.011814
  XRPUSD_distance_from_ema_period_24              11.803943
  XRPUSD_close_to_vwap                            11.767947
  XRPUSD_avg_taker_buy_price_diff                 11.507469
  XRPUSD_vwap_distance_period_24                  11.435864
  BTCUSD_price_zscore_period_12                   11.430959
  ETHUSD_kyle_lambda_period_96                    11.309346
  BTCUSD_close_to_vwap                            11.179125
  XRPUSD_price_zscore_period_24                   10.995550
  XRPUSD_cumulative_volume_imbalance_period_12    10.858816
  XRPUSD_close_change_period_12                   10.535476
  BTCUSD_buyer_pressure_change_period_168         10.524774
  XRPUSD_vwap_proxy                               10.474239
  BTCUSD_avg_taker_sell_price_diff                10.457894
  BTCUSD_vwap_distance_period_12                  10.452774
  BTCUSD_sell_price_impact                        10.449262
  BTCUSD_buy_efficiency                           10.390185
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0658
- out-sample results:
  BACKTEST RESULTS
  ----------------
  TOTAL TRADES: 12867
  TOTAL WINS: 6742
  WIN RATE: 52.40%
  TOTAL RETURNS: 617.00%
  MAX CONSEC. WINS: 17
  MAX CONSEC. LOSSES: 25
  MAX DRAWDOWN: 37.50%
  AVG CONSEC. WINS: 2.62
  AVG CONSEC LOSSES: 2.38
  AVG DRAWDOWN: 3.60%
  FULL KELLY FRACTION: 4.80%
  1/4 KELLY FRACTION: 1.20%
  ----------------
  MODEL EVALUATION: BTCUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 5
  TOP 20 FEATURES (BY AVG GAIN):
  XRPUSD_open_change_period_48             12.795036
  BTCUSD_sell_price_impact                 12.767150
  BTCUSD_close_change_period_24            12.736389
  BTCUSD_avg_taker_sell_price_diff         12.220894
  XRPUSD_vol_norm_volatility_period_24     12.040105
  ETHUSD_avg_taker_sell_price_diff         12.008086
  SOLUSD_vol_norm_volatility_period_24     11.875666
  ETHUSD_close_to_vwap                     11.870477
  BTCUSD_vwap_distance_period_12           11.862360
  SOLUSD_sell_price_impact                 11.795766
  BTCUSD_net_taker_volume                  11.771426
  XRPUSD_trade_intensity                   11.763614
  ETHUSD_price_zscore_period_12            11.748013
  SOLUSD_avg_taker_sell_price_diff         11.744688
  ETHUSD_avg_taker_buy_price_diff          11.730094
  BTCUSD_volume_concentration_period_96    11.679077
  BTCUSD_volume_change_period_168          11.662331
  BTCUSD_volatility_change_period_96       11.577769
  SOLUSD_volume_change_period_168          11.560187
  ETHUSD_vwap_distance_period_12           11.550617
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0326
  MODEL EVALUATION: ETHUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 5
  TOP 20 FEATURES (BY AVG GAIN): 
  BTCUSD_price_zscore_period_12                   12.683624
  BTCUSD_cumulative_volume_imbalance_period_24    12.503434
  ETHUSD_vwap_distance_period_12                  12.501689
  SOLUSD_sell_price_impact                        12.439655
  SOLUSD_buy_efficiency                           12.308952
  XRPUSD_kyle_lambda_period_24                    12.232067
  XRPUSD_open_change_period_24                    12.151707
  SOLUSD_distance_from_ema_period_24              11.909179
  ETHUSD_amihud_period_24                         11.891432
  XRPUSD_sell_price_impact                        11.862430
  ETHUSD_buy_efficiency                           11.798830
  SOLUSD_volume_change_period_96                  11.727270
  SOLUSD_price_zscore_period_12                   11.664739
  XRPUSD_volume_price_corr_period_12              11.660619
  ETHUSD_low_change_period_12                     11.653326
  SOLUSD_trade_size_volatility_period_24          11.629471
  ETHUSD_vwap_distance_period_24                  11.567915
  ETHUSD_avg_taker_buy_price_diff                 11.474655
  ETHUSD_avg_taker_sell_price_diff                11.464073
  XRPUSD_avg_trade_size_change_period_168         11.429744
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0073
  MODEL EVALUATION: SOLUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 5
  TOP 20 FEATURES (BY AVG GAIN):
  XRPUSD_quote_volume_momentum_period_96          12.695464
  ETHUSD_taker_buy_qa_volume                      12.475522
  BTCUSD_cumulative_volume_imbalance_period_48    12.115732
  ETHUSD_buy_efficiency                           12.086115
  SOLUSD_taker_buy_qa_volume                      12.072828
  SOLUSD_volume_price_corr_period_168             12.047272
  SOLUSD_open_change_period_96                    11.970747
  SOLUSD_vwap_distance_period_12                  11.633631
  SOLUSD_close_change_period_48                   11.629688
  SOLUSD_num_trades                               11.626886
  ETHUSD_kyle_lambda_period_48                    11.527662
  ETHUSD_vpin_period_96                           11.439997
  SOLUSD_quote_volume_momentum_period_48          11.418437
  ETHUSD_price_zscore_period_168                  11.414267
  XRPUSD_price_zscore_period_12                   11.401567
  BTCUSD_volatility_change_period_96              11.395727
  SOLUSD_taker_buy_ba_volume                      11.370590
  BTCUSD_volume_concentration_period_12           11.346526
  SOLUSD_volume_change_period_96                  11.345526
  ETHUSD_taker_sell_ba_volume                     11.308160
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0394
  MODEL EVALUATION: XRPUSD
  ----------------------------
  TOTAL MODELS EVALUATED: 5
  TOP 20 FEATURES (BY AVG GAIN):
  ETHUSD_volume_imbalance                   12.479079
  SOLUSD_volatility_change_period_96        12.162711
  ETHUSD_close_to_vwap                      12.144378
  ETHUSD_net_taker_volume                   12.100295
  SOLUSD_volume_change_period_96            12.086314
  SOLUSD_buy_efficiency                     12.029059
  SOLUSD_taker_sell_ba_volume               11.948286
  SOLUSD_buyer_pressure_change_period_12    11.771544
  BTCUSD_volume_change_period_96            11.652291
  ETHUSD_volume_price_corr_period_96        11.630504
  XRPUSD_volatility_change_period_168       11.612005
  ETHUSD_num_trades                         11.545988
  ETHUSD_kyle_lambda_period_12              11.544908
  XRPUSD_volume_change_period_12            11.543983
  BTCUSD_vol_norm_volatility_period_168     11.499334
  SOLUSD_vwap_proxy                         11.498658
  SOLUSD_close_change_period_24             11.498581
  BTCUSD_open_change_period_48              11.491145
  SOLUSD_volume_concentration_period_96     11.449979
  XRPUSD_volume_concentration_period_48     11.447361
  dtype: float64
  MODEL DRIFT (AVG INTER-FOLD CORRELATION):
  0.0332

Interpretation:
- model consistancy:
  * model seems to hold true out sample
  * model win rate is around ~53% with minor degradation in out sample
- model stability:
  * model stability is weak as features are constantly shifting
  * introduction of cross-correlation features seems to expand change in feature importance

Notes:
- selected parameters:
  config = {
    'filepath': 'analysis/data/crypto_1h_testing.csv',
    'start': '2025-04-01',
    'end': '2026-01-31',
    'symbols': ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD'],
    'starting_equity': 100,
    'training_periods': 4,
    'testing_periods': 1,
    'walk_forward_shift_periods': 1,
    'upper_threshold': 0.505,
    'lower_threshold': 0.495
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
- observeations:
  * model performance has the highest stability and consistancy seen yet
  * by diversifying to multiple assets altogether we can ensure that the volume of trades is greater -> theoretically increasing the amount of returns that are able to be generated  

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