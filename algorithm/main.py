import time
from clients.collection_agent import CollectionAgent
from clients.trade_entry_agent import TradeEntryAgent
from clients.trade_exit_agent import TradeExitAgent
from storage.data_attributes import CandleData, TradePosition
from storage.memory import Memory
from strategy.strategy import StrategyEngine

def main() -> None:

    try:
        candles = CollectionAgent()
        entry = TradeEntryAgent()
        exit = TradeExitAgent()
        engine = StrategyEngine()

        with Memory() as memory:
            while True:

                dict_candle_data = candles.data_collection()
                predictions = engine.model_predictions(dict_candle_data)
                entered_trade_positions = entry.trade_entry(predictions)

                memory.add_new_trade_position(entered_trade_positions)
                pending_trade_positions = memory.return_current_positions()

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