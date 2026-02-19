import os
from typing import Dict, List
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from dotenv import load_dotenv
load_dotenv()

# User Info
PUBLIC_KEY:str = os.getenv("PUBLIC_KEY")
PRIVATE_KEY:str = os.getenv("PRIVATE_KEY")

# API requests
RPC_URL:str = "https://polygon-mainnet.core.chainstack.com/f2108cb095352bfea68ab7f2aa5eba3d"
KLINES_URL:str = 'https://api.binance.com/api/v3/klines'
MARKET_URL:str = 'https://gamma-api.polymarket.com/markets/slug/'
TRADE_URL:str = "https://clob.polymarket.com"

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
POST_REDEMPTION_PERIODS:int = 5
MAX_LIST_LEN:int = 200
STRATEGY_PERIODS:List[int] = [12, 24, 48, 96, 168]

# Directory
TRADE_LOG_FILEPATH:str = 'trade_logs.csv'
MODEL_FILEPATHS:Dict[str,str] = {symbol: f'{symbol}_live_model.pkl' for symbol, _ in SYMBOLS_MAP.items()}

# Trade Amount
USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
USDC_BALANCE_ABI = [
    {
        "inputs": [{"internalType": "address", "name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]
def get_usdc_balance() -> float:
    try:
        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        contract = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_BALANCE_ABI)
        balance = contract.functions.balanceOf(PUBLIC_KEY).call()
        return balance / 1e6
    except Exception as e:
        print(f"ERROR: Failed to get USDC balance: {e}")
        return 0.0