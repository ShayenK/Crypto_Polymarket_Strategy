from dataclasses import dataclass

@dataclass(frozen=True)
class TradePosition:
    entry_proba:float
    entry_time:int
    entry_price:float
    entry_units:float
    direction:str
    yes_token_id:str
    no_token_id:str
    condition_id:str
    order_id:str
    neg_risk:bool
    exit_time:int
    exit_price:float
    exit_units:float
    position_status:str
    outcome:str

@dataclass(frozen=True)
class CandleData:
    symbol:str
    time:int
    open:float
    high:float
    low:float
    close:float
    volume:float
    qa_volume:float
    num_trades:int
    taker_buy_ba_volume:float
    taker_buy_qa_volume:float
    taker_sell_ba_volume:float
    taker_sell_qa_volume:float