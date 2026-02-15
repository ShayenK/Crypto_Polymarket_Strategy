import time
from clients.collection_agent import CollectionAgent
from clients.trade_entry_agent import TradeEntryAgent
from clients.trade_exit_agent import TradeExitAgent
from storage.memory import Memory
from strategy.strategy import StrategyEngine

def main() -> None:

    try:

        # Initialize Object Classes
        candles = CollectionAgent()
        entry = TradeEntryAgent()
        exit = TradeExitAgent()
        engine = StrategyEngine()

        # Run Loop with Memory Class Lifecycle as a Focus
        with Memory() as memory:
            while True:

                # Collect, Predict, and Enter Trade
                dict_candle_data = candles.data_collection()
                predictions = engine.model_predictions(dict_candle_data)
                entered_trade_positions = entry.trade_entry(predictions)

                # Add Position to Memory and Return All current Pending Positions
                memory.add_new_trade_position(entered_trade_positions)
                pending_trade_positions = memory.return_current_positions()

                # Redeem Pending Positions that Meet Criteria and Remove them from Memory
                redeemed_trade_positions = exit.trade_exit(pending_trade_positions)
                memory.remove_finished_positions(redeemed_trade_positions)

                time.sleep(1)

    except Exception as e:
        print(f"ERROR: unable to keep loop alive {e}")
    except KeyboardInterrupt:
        print("INFO: shutting down")

    return

if __name__ == "__main__":
    main()