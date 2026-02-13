from clients.collection_agent import CollectionAgent
from strategy.strategy import StrategyEngine

collection = CollectionAgent()
engine = StrategyEngine()

dict_candle_data = collection.data_collection()
predictions = engine.model_predictions(dict_candle_data)

print(predictions)