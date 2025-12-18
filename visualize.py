from argparse import ArgumentParser
from os.path import isfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def boxplot(df: pd.DataFrame):
    # Get all features to plot (skip column 1)
    features = [col for col in df.columns if col != 1]
    classes = df[1].unique()
    
    # Calculate grid size
    n_features = len(features)
    n_cols = 5  # 5 columns in the grid
    n_rows = (n_features + n_cols - 1) // n_cols  # Round up
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()  # Convert to 1D array for easy indexing
    
    for idx, col in enumerate(features):
        df_to_study = df[[col, 1]].dropna()
        
        # Separate data by class (M and B)
        data_by_class = []
        for cls in classes:
            # Filter rows that belong to current class
            mask = df_to_study[1] == cls
            rows_of_class = df_to_study[mask]
            # Extract feature values for this class
            feature_values = rows_of_class[col].values
            data_by_class.append(feature_values)
        
        # Plot on the corresponding subplot
        axes[idx].boxplot(data_by_class, tick_labels=classes)
        axes[idx].set_xlabel("Class")
        axes[idx].set_ylabel(f"Feature {col}")
        axes[idx].set_title(f"Feature {col}")
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def bar_charts(df: pd.DataFrame):
    features = [(col - 1) for col in df.columns if col != 1]
    diagnostic_classes = df[1].unique()
    total_means = {}

    for feature in features:
        sub_df = df[[feature + 1, 1]]
        means = {}
        for diag in diagnostic_classes:
            mask = sub_df[1] == diag
            filtered_df = sub_df[mask]
            # print("Filtered DF for feature", feature, "and class", diag, ":\n", filtered_df)
            feature_mean = np.mean(filtered_df[feature + 1].values)
            print("Mean for feature", feature, "and class", diag, ":", feature_mean)
            means[diag] = feature_mean
        total_means[feature] = means
    
    n_features = len(features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    for idx, feature in enumerate(features):
        feature_data = total_means[feature]
        print("Plotting feature data:", feature_data)
        for diag, mean in feature_data.items():
            axes[idx].bar(diag, mean, label=f"diag {diag}")
            axes[idx].set_xlabel("Class")
            axes[idx].set_ylabel("Mean Value")
            axes[idx].legend()
            axes[idx].set_title(f"Feature {feature}")
    
    plt.tight_layout()
    plt.show()
        

        




if __name__ == "__main__":
    parser = ArgumentParser(prog="Visualize",
                            description="Visualize the dataset before training")
    parser.add_argument("dataset", help="a .csv dataset")
    args = parser.parse_args()
    dataset = args.dataset
    
    print(f"Visualizing dataset: {dataset}")
    if not dataset.endswith('.csv'):
        raise ValueError("The dataset must be a .csv file")
    if not isfile(dataset):
        raise FileNotFoundError(f"The file {dataset} does not exist")
    df = pd.read_csv(dataset, header=None).dropna()
    df = df.drop(columns=[0])
    # print(df)
    # boxplot(df)
    bar_charts(df)

    
    