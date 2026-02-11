import os
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

# Strategy Parameters
SYMBOLS_MAP:Dict[str,str] = {
    'BTCUSD': 'bitcoin',
    'ETHUSD': 'ethereum',
    'SOLUSD': 'solana',
    'XRPUSD': 'xrp'
}
CHAIN_ID:int = 137
SIGNATURE:int = 0
MAX_RETRY_ATTEMPTS:int = 5
UPPER_THRESHOLD:float = 0.51
LOWER_THRESHOLD:float = 0.49
TRADE_UNITS:float = 1.00
POST_REDEMPTION_PERIODS:int = 5
MAX_LIST_LEN:int = 200
STRATEGY_PERIODS:List[int] = [12, 24, 48, 96, 168]

# User Info
PUBLIC_KEY:str = os.getenv("PUBLIC_KEY")
PRIVATE_KEY:str = os.getenv("PRIVATE_KEY")

# API requests
RPC_URL:str = "https://polygon-mainnet.core.chainstack.com/f2108cb095352bfea68ab7f2aa5eba3d"
KLINES_URL:str = 'https://api.binance.com/api/v3/klines'
MARKET_URL:str = 'https://gamma-api.polymarket.com/markets/slug/'
TRADE_URL:str = "https://clob.polymarket.com"

# Directory
TRADE_LOG_FILEPATH:str = 'trade_logs.csv'
MODEL_FILEPATHS:Dict[str,str] = {symbol: f'{symbol}_live_model.pkl' for symbol, _ in SYMBOLS_MAP.items()}