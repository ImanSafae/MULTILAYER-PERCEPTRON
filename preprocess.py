from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from os.path import isfile
import pandas as pd

def standardize_dataset(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    features, predictions = split_features_and_predictions(df, pred_col=1)
    scaler.fit(features)
    standardized_features = pd.DataFrame(scaler.transform(features), columns=features.columns)
    standardized_df = pd.concat([predictions, standardized_features], axis=1)
    return standardized_df

def split_features_and_predictions(df: pd.DataFrame, pred_col: int):
    features = df.drop(columns=[pred_col])
    predictions = df[pred_col]
    return features, predictions

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()  # Remove rows with NaN values
    df = df.drop(columns=[0])  # Remove ID column
    return df
    

if __name__ == "__main__":
    parser = ArgumentParser(prog="Preprocess data")
    parser.add_argument("dataset", help="a .csv dataset for training")
    # parser.add_argument("-v", "--valid", help="a .csv dataset for validation", required=True)
    args = parser.parse_args()
    dataset: chr = args.dataset
    # valid_dataset: chr = args.valid
    if (not dataset.endswith(".csv")):
        raise ValueError("Program expects .csv file.")
    if (not isfile(dataset)):
        raise FileNotFoundError(f"The file {dataset} does not exist.")
    # if (not isfile(valid_dataset)):
    #     # raise FileNotFoundError(f"The file {valid_dataset} does not exist.")
    
    scaler = StandardScaler()
    df: pd.DataFrame = pd.read_csv(dataset, header=None)
    # valid_df: pd.DataFrame = pd.read_csv(valid_dataset, header=None)
    df = clean_dataset(df)
    # valid_df = clean_dataset(valid_df)
    # print("Before scaling, training dataset:\n", train_df.head())
    df = standardize_dataset(df, scaler)
    # print("After scaling, training dataset:\n", train_df.head())
    # print('-------------------------')
    # print("Before scaling, validation dataset:\n", valid_df.head())
    # valid_df = standardize_dataset(valid_df, scaler)
    # print("After scaling, validation dataset:\n", valid_df.head())
    df.to_csv(f"preprocessed_{dataset}", index=False, header=False)
    # valid_df.to_csv("preprocessed_validation_set.csv", index=False, header=False)