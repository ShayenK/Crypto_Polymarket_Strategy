import time
from clients.collection_agent import CollectionAgent
from clients.trade_entry_agent import TradeEntryAgent
from clients.trade_exit_agent import TradeExitAgent
from storage.memory import Memory
from strategy.strategy import StrategyEngine
from setup.forward_testing import ForwardTesting

def main() -> None:

    # Initialize Class Objects
    try:
        collection = CollectionAgent()
        entry = TradeEntryAgent()
        exit = TradeExitAgent()
        engine = StrategyEngine()
        testing = ForwardTesting()

        # Testing Loop with Memory Class Lifecycle as a Focus
        with Memory() as memory:
            while True:

                # Collect, Predict, and Log Prediction
                dict_candle_data = collection.data_collection()
                predictions = engine.model_predictions(dict_candle_data)
                testing.record_probabilities(predictions)

                time.sleep(1)

    except Exception as e:
        print(f"ERROR: unable to keep loop alive {e}")
    except KeyboardInterrupt:
        print("INFO: shutting down")

    return

if __name__ == "__main__":
    main()