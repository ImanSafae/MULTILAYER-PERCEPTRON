from argparse import ArgumentParser
import os
from MLP import MLP
import numpy as np
import pandas as pd
from preprocess import clean_dataset, standardize_dataset
from sklearn.preprocessing  import StandardScaler

def preprocess_data(dataset):
    dataset = clean_dataset(dataset)
    dataset = standardize_dataset(dataset, StandardScaler())
    return dataset

def parse_weights_and_biases(model_folder: str, weights, biases, hidden_layers_nb):
    files = [file for file in os.listdir(model_folder) if os.path.isfile(f"{model_folder}/{file}")]
    config_path = f"{model_folder}/config.txt"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            hidden_layers_nb = int(float(f.read().strip()))
    else:
        hidden_layers_nb = 2
    # print("files in model folder:", files)
    for file in files:
        if file.startswith("weights_"):
            weights.append(f"{model_folder}/{file}")
        if file.startswith("biases_"):
            biases.append(f"{model_folder}/{file}")

def init_mlp(input_size, weights, biases, hidden_layers_nb=2):
    # Determine number of hidden layers based on number of weight files - 1 (last one is output)
    # hidden_layers_nb = len(weights) - 1
    mlp = MLP(input_size, hidden_layers_nb, learning_rate=0.01)
    print("Initialized MLP with", hidden_layers_nb, "hidden layers.")
    
    # Load weights and biases for all layers
    for i in range(len(weights)):
        mlp.hidden_weights[i] = np.loadtxt(weights[i], delimiter=',')
    
    for i in range(len(biases)):
        if i < hidden_layers_nb:
            mlp.hidden_layers_biases[i] = np.loadtxt(biases[i], delimiter=',')
        else:
            mlp.final_layer_biases = np.loadtxt(biases[i], delimiter=',')
    
    return mlp

def parse_dataset(dataset):
    dataset: pd.DataFrame = pd.read_csv(dataset, header=None)
    dataset = preprocess_data(dataset)
    expected = dataset[1]
    dataset = dataset.drop(columns=[1])
    return dataset, expected

def interpret_proba(probas):
    predictions = []
    for proba in probas:
        if (proba[0] > proba[1]):
            predictions.append('B')
        else:
            predictions.append('M')
    return predictions

def predict(mlp: MLP, dataset):
    output_probas = mlp.forward(dataset.to_numpy())
    predicted_diagnoses = interpret_proba(output_probas)
    # print("Predicted diagnoses:", predicted_diagnoses)
    loss = mlp.cross_entropy_loss(output_probas, pd.get_dummies(expected, dtype=int).to_numpy())
    return predicted_diagnoses, loss

def compute_accuracy(predicted, expected):
    total = len(predicted)
    accurates = 0
    if len(predicted) != len(expected):
        raise ValueError("Length of predicted and expected arrays must be the same.")
    for i in range(total):
        if predicted[i] == expected.iloc[i]:
            accurates += 1
    percentage = accurates * 100 / total
    return percentage

if __name__ == "__main__":
    parser = ArgumentParser(prog="Predict.py", description="Prediction of a binary diagnosis")
    parser.add_argument("--dataset", help="The dataset to predict on", required=True)
    parser.add_argument("--model", help="The folder name containing the weights and biases needed", required=True)
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset

    if not os.path.exists(model):
        raise ValueError("The model folder does not exist")
    if not (dataset.endswith('.csv')):
        raise ValueError("The dataset should be a .csv file")
    if not os.path.exists(dataset):
        raise ValueError("The provided dataset file does not exist")
    weights = []
    biases = []
    hidden_layers_nb = 0
    parse_weights_and_biases(model, weights, biases, hidden_layers_nb)
    dataset, expected = parse_dataset(dataset)
    # print("weights:", weights)
    # print("biases:", biases)
    mlp: MLP = init_mlp(len(dataset.columns), weights, biases)
    predicted_diagnoses, loss = predict(mlp, dataset)
    accuracy = compute_accuracy(predicted_diagnoses, expected)
    print(f"Accuracy of the model on the provided dataset: {accuracy:.2f}%")
    print(f"Loss of the model on the provided dataset: {loss:.4f}")