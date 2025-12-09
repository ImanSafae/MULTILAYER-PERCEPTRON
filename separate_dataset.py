from argparse import ArgumentParser
import sys
from os.path import isfile
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser(prog="Dataset Separator", description="Separate dataset into training and validation sets")
    parser.add_argument("dataset", help ="a .csv dataset")
    args = parser.parse_args()
    dataset = args.dataset
    if not isfile(dataset):
        print(f"The file {dataset} does not exist.")
        sys.exit(1)
    if not dataset.endswith('.csv'):
        print("The dataset must be a .csv file.")
        sys.exit(1)
    data = pd.read_csv(dataset)
    train_frac = 0.8
    slice_index = int(len(data.index) * train_frac)
    training_data = data.iloc[:slice_index]
    validation_data = data.iloc[slice_index:]
    print(training_data)
    print(validation_data)
    training_data.to_csv("training_set.csv", index=False)
    validation_data.to_csv("validation_set.csv", index=False)