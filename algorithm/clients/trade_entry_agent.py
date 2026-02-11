import time
import copy
import json
import requests
from zoneinfo import ZoneInfo
from dataclasses import replace
from typing import Optional, Dict
from datetime import datetime, timezone, timedelta
from py_clob_client.client import ClobClient
from py_clob_client.order_builder.constants import BUY
from py_clob_client.clob_types import MarketOrderArgs, OrderType
from storage.data_attributes import TradePosition
from config import (
    SYMBOLS_MAP,
    MARKET_URL,
    TRADE_URL,
    PUBLIC_KEY,
    PRIVATE_KEY,
    CHAIN_ID,
    SIGNATURE,
    TRADE_UNITS,
    MAX_RETRY_ATTEMPTS,
    UPPER_THRESHOLD,
    LOWER_THRESHOLD
)

class TradeEntryAgent:
    def __init__(self):
        self.current_positions:Dict[str,Optional[TradePosition]] = {symbol: None for symbol, _ in SYMBOLS_MAP.items()}
        self._client:ClobClient = self.__client_authentication()

    def __client_authentication(self) -> ClobClient:

        client = ClobClient(
            host=TRADE_URL,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID
        )
        user_api_creds = client.create_or_derive_api_creds()
        client = ClobClient(
            host=TRADE_URL,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            creds=user_api_creds,
            signature_type=SIGNATURE,
            funder=PUBLIC_KEY
        )

        return client
    
    def _reset_current_positions(self) -> None:

        # Reset the Current Position (only after has been appended in-memory)
        self.current_positions = {symbol: None for symbol, _ in SYMBOLS_MAP.items()}

        return
    
    def _get_recent_market_time(self) -> int:

        # Get Recent Unix Timestamp for Market (1 hour)

        return int((time.time() // 3600) * 3600)
    
    def _get_slug_market_time(self, symbol_str:str) -> str:

        # Get Recent Formatted Datetime Slug for Market (symbol-month-dom-analouge_hour)
        now_et = datetime.now(ZoneInfo("US/Eastern"))
        month_et = now_et.strftime('%B').lower()
        day_of_month_et = now_et.day
        hour_et = now_et.strftime("%I%p").lstrip("0").lower()
        slug_str = f'{symbol_str}-up-or-down-{month_et}-{day_of_month_et}-{hour_et}-et'

        return slug_str
    
    def _get_market(self) -> bool:

        # Get the Current Market for Symbols (1 hour)
        try:
            for symbol, symbol_str in SYMBOLS_MAP.items():
                print(f"INFO: attempting to fetch market for {symbol}")
                unix_time = self._get_recent_market_time()
                slug_str = self._get_slug_market_time(symbol_str)
                url = MARKET_URL + slug_str
                itr = 0
                while True:
                    if itr >= MAX_RETRY_ATTEMPTS:
                        print(f"ERROR: could not fetch market data for {symbol}")
                        break
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        raw_token_ids = data.get('clobTokenIds')
                        token_ids = json.loads(raw_token_ids)
                        condition_id = data.get('conditionId')
                        neg_risk = data.get('negRisk', False)
                        self.current_positions[symbol] = TradePosition(
                            symbol=symbol,
                            entry_proba=None,
                            entry_time=unix_time,
                            entry_price=None,
                            entry_units=None,
                            direction=None,
                            yes_token_id=str(token_ids[0]),
                            no_token_id=str(token_ids[1]),
                            condition_id=str(condition_id),
                            order_id=None,
                            neg_risk=bool(neg_risk),
                            exit_time=(unix_time+3600),
                            exit_price=None,
                            exit_units=None,
                            position_status=None,
                            outcome=None
                        )
                        print(f"INFO: collected market data for {symbol}")
                        break
                    itr += 1
                    time.sleep(1.333*itr)
            return True
        
        except Exception as e:
            print(f"ERROR: trade entry issue {e}")
            return False
    
    def _entry_position(self, predictions:Dict[str,float]) -> None:

        for symbol, _ in SYMBOLS_MAP.items():
            prediction = predictions[symbol]

            # Check for None Predictions
            if prediction is None:
                continue

            # Signal Checking
            if prediction >= UPPER_THRESHOLD:
                token_id = self.current_positions[symbol].yes_token_id
                direction = "UP"
            elif prediction <= LOWER_THRESHOLD:
                token_id = self.current_positions[symbol].no_token_id
                direction = "DOWN"
            else:
                print(f"INFO: no signal for {symbol}: prediction value -> {prediction}")
                continue

            # Place Entry Order
            itr = 0
            while True:

                if itr >= MAX_RETRY_ATTEMPTS:
                    print(f"ERROR: could not enter trade for {symbol}")
                    self.current_positions[symbol] = None
                    break

                try: 
                    order_args = MarketOrderArgs(
                        token_id=token_id,
                        amount=TRADE_UNITS,
                        side=BUY
                    )
                    signed_order = self._client.create_market_order(order_args)
                    resp = self._client.post_order(signed_order, orderType=OrderType.FOK)
                    if resp.get('orderID'):
                        taking_amt = float(resp.get('takingAmount', 0))
                        making_amt = float(resp.get('makingAmount', 0))
                        execution_price = making_amt / taking_amt if taking_amt > 0 else 0
                        print(f"INFO: order filled for {symbol} 15 minute prediction at entry time {datetime.fromtimestamp(self.current_positions[symbol].entry_time, tz=timezone.utc)}")
                        position_updates = {
                            'entry_proba': prediction,
                            'entry_price': execution_price,
                            'entry_units': making_amt,
                            'direction': direction,
                            'order_id': resp.get('orderID'),
                            'position_status': "PENDING"
                        }
                        self.current_positions[symbol] = replace(self.current_positions[symbol], **position_updates)
                        break
                    itr += 1
                    time.sleep(1.333*itr)
                except Exception as e:
                    print(f"ERROR: unable to enter trade {e}")

        return
        

        
    def trade_entry(self, predictions:Dict[str,float]) -> Optional[Dict[str,TradePosition]]:
        """
        Trade entry function that uses prediction at the current candle to enter a trade

        Args:
            predictions:Dict[str,float] -> a dict of symbol prediction values based on symbol
        Returns:
            entered_trade_positions:Dict[str,TradePositon] -> a dict of symbol trade positions that can be stored for later redemption
        """

        if not predictions: return None
        self._reset_current_positions()
        check_1 = self._get_market()
        if not check_1: return None
        self._entry_position(predictions)
        entered_trade_positions = copy.deepcopy(self.current_positions)

        return entered_trade_positions