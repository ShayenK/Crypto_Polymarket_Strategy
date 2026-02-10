import pandas as pd

class DataSplit:
    def __init__(self, filepath:str):
        self.df:pd.DataFrame = pd.read_csv(filepath)

    def split_dataset(self, split_point:float, training_output_filepath:str, testing_output_filepath:str) -> None:

        # Split into Training and Testing
        df = self.df.copy()
        len_df = len(df)
        split_index = int(len_df * split_point)
        training_df = df[:split_index]
        testing_df = df[split_index:]
        print("INFO: outputing filepaths...")
        training_df.to_csv(training_output_filepath, index=False)
        testing_df.to_csv(testing_output_filepath, index=False)
        print("INFO: outputed datasets")

        return None
    
    def check_split_dates(self, training_filepath:str, testing_filepath:str) -> None:

        # Date Check on Dataset Column 'time'
        training_df = pd.read_csv(training_filepath)
        testing_df = pd.read_csv(testing_filepath)
        train_start = pd.to_datetime(training_df.iloc[0]['time'], unit='s') 
        train_end = pd.to_datetime(training_df.iloc[-1]['time'], unit='s') 
        test_start = pd.to_datetime(testing_df.iloc[0]['time'], unit='s') 
        test_end = pd.to_datetime(testing_df.iloc[-1]['time'], unit='s') 
        print(f"INFO: training ({train_start} -> {train_end})")
        print(f"INFO: testing ({test_start} -> {test_end})")

        return None

def main() -> None:
    data_split = DataSplit('analysis/data/crypto_1h_features.csv')

    # 1. Split Dataset
    data_split.split_dataset(
        0.8,
        'analysis/data/crypto_1h_training.csv',
        'analysis/data/crypto_1h_testing.csv'
    )

    # 2. Check Date Split
    data_split.check_split_dates(
        'analysis/data/crypto_1h_training.csv',
        'analysis/data/crypto_1h_testing.csv'
    )

    return None

if __name__ == "__main__":
    main()