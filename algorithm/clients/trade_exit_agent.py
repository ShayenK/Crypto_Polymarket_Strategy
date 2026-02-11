import time
import copy
from dataclasses import replace
from typing import Optional, Dict, List
from datetime import datetime, timezone
from setup.claim import redeem_position
from storage.data_attributes import TradePosition
from config import (
    SYMBOLS_MAP,
    POST_REDEMPTION_PERIODS,
    MAX_RETRY_ATTEMPTS
)

class TradeExitAgent:
    def __init__(self):
        self.redeemed_positions:Dict[str,List[TradePosition]] = {symbol: [] for symbol in SYMBOLS_MAP.keys()}

    def _reset_redeemed_positions(self) -> None:

        # Resets all redeemed trade positions
        self.redeemed_positions = {symbol: [] for symbol in SYMBOLS_MAP.keys()}

        return
    
    def _get_recent_time(self) -> int:

        # Get Recent Unix Timestamp for Market (1 hour)

        return int(time.time())
    
    def _exit_positions(self, entery_trade_positions:Dict[str,List[TradePosition]]) -> None:

        # Sequentially Iterate Over Trade Positions and Redeem
        for symbol in SYMBOLS_MAP.keys():
            unix_time = self._get_recent_time()
            for trade_position in entery_trade_positions[symbol]:
                if (unix_time - trade_position.exit_time) >= (900 * POST_REDEMPTION_PERIODS):
                    itr = 0
                    while True:
                        if itr >= MAX_RETRY_ATTEMPTS:
                            modified_position_details = {
                                'exit_price':  0.00,
                                'exit_units': 0.00,
                                'position_status': "UNRESOLVED",
                                'outcome': "LOSS"
                            }
                            redeemed_position = replace(trade_position, **modified_position_details)
                            self.redeemed_positions[symbol].append(redeemed_position)
                            break

                        try: 
                            status, amount = redeem_position(
                                trade_position.condition_id,
                                trade_position.yes_token_id,
                                trade_position.no_token_id,
                                trade_position.neg_risk
                            )
                            if status == "UNRESOLVED":
                                self.redeemed_positions[symbol].append(trade_position)
                                print(f"INFO: {symbol} market not resolved yet, keeping as PENDING")
                                break
                            elif status == "RESOLVED":
                                modified_position_details = {
                                    'exit_price': 1.00 if amount > 0 else 0.00,
                                    'exit_units': float(amount),
                                    'position_status': "COMPLETE",
                                    'outcome': "WIN" if amount > 0 else "LOSS"
                                }
                                redeemed_position = replace(trade_position, **modified_position_details)
                                self.redeemed_positions[symbol].append(redeemed_position)
                                print(f"INFO: {symbol} position has been collected for market ending at {datetime.fromtimestamp(trade_position.exit_time, tz=timezone.utc)}")
                                break
                            elif status == "ERROR":
                                print(f"ERROR: {symbol} redemption attempt {itr+1} failed")    
                            itr += 1
                            time.sleep(1.333*itr)
                        except Exception as e:
                            print(f"ERROR: {symbol} could not redeem positions -> {e}")
                            
                        itr += 1
                        time.sleep(1.333*itr)

        return

    def trade_exit(self, entery_trade_positions:Dict[str,List[TradePosition]]) -> Optional[Dict[str,List[TradePosition]]]:
        """
        Trade exit function that redeems list of all claimable positions

        Args:
            entered_trade_positions:Dict[str,List[TradePosition]] -> a dict of entered trade positions based on the symbol
        Returns:
            entered_trade_positions:Dict[str,List[TradePosition]] -> a dict of redeemed trade positions based on the symbol
        """

        if not entery_trade_positions: return None
        self._reset_redeemed_positions()
        self._exit_positions(entery_trade_positions)
        redeemed_trade_positions = copy.deepcopy(self.redeemed_positions)

        return redeemed_trade_positions