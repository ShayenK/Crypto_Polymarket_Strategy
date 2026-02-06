# XGBoost Directional Prediction Algorithm for BTCUSD, ETHUSD, SOLUSD, XRPUSD: 1 hour Polymarket Contract #


## Objective
- identify if cross-asset correlation is a more accurate predictor of directional betting than single asset
- identify if cross-asset correlation offsets the risk profile of the trades


## Background
- polymarket provides directional betting for the pairs BTCUSD, ETHUSD, SOLUSD, XRPUSD
- XGBoost is great at picking up on relational dependancies


## Overview:

#### Phase 1: Data Manipulation
[ ] Data Collection
  * retrive candle data for all 4 assets
  * collate into csv
[ ] Data Checking
  * visual inspection
  * missingness / error value checks
[ ] Feature Engineering
  * Engineer features for all 4 assets
[ ] Data Spliting
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