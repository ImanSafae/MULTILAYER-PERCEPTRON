from argparse import ArgumentParser
import pandas as pd
from MLP import MLP
import numpy as np
import matplotlib.pyplot as plt
import os

def interpret_proba(probas):
    predictions = []
    for proba in probas:
        if (proba[0] > proba[1]):
            predictions.append('B')
        else:
            predictions.append('M')
    return predictions

def predict(mlp: MLP, valid_dataset, expected):
    predictions_probabilities = mlp.forward(valid_dataset)
    predictions = interpret_proba(predictions_probabilities)
    return predictions, predictions_probabilities

def compute_accuracy(predicted, expected):
    total = len(predicted)
    accurates = 0
    if len(predicted) != len(expected):
        raise ValueError("Length of predicted and expected arrays must be the same.")
    for i in range(total):
        if predicted[i] == expected[i]:
            accurates += 1
    percentage = accurates * 100 / total
    return percentage

def export_weights_and_biases(mlp: MLP, folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    i = 0
    np.savetxt(f"{folder_name}/weights_{i}.csv", mlp.input_hidden_weights, delimiter=',')
    i += 1
    np.savetxt(f"{folder_name}/weights_{i}.csv", mlp.hidden_hidden_weights, delimiter=',')
    i += 1
    np.savetxt(f"{folder_name}/weights_{i}.csv", mlp.hidden_output_weights, delimiter=',')
    i = 0
    np.savetxt(f"{folder_name}/biases_{i}.csv", mlp.first_hidden_biases, delimiter=',')
    i += 1
    np.savetxt(f"{folder_name}/biases_{i}.csv", mlp.second_hidden_biases, delimiter=',')
    i += 1
    np.savetxt(f"{folder_name}/biases_{i}.csv", mlp.final_layer_biases, delimiter=',')


def plot_all_metrics(val_accuracies, val_losses, accuracies, losses):
    fig, axes = plt.subplots(1,2)
    axes[0].set_title("Loss over epochs")
    axes[0].plot(val_losses, label="Validation Loss")
    axes[0].plot(losses, label="Training Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()
    axes[1].set_title("Accuracy over epochs")
    axes[1].plot(val_accuracies, label="Validation Accuracy")
    axes[1].plot(accuracies, label="Training Accuracy")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()
    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser(prog="Training", description="Training of a multilayer perceptron model")
    parser.add_argument("--train", help="a .csv preprocessed training dataset", required=True)
    parser.add_argument("--valid", help="a .csv preprocessed validation dataset", required=True)
    args = parser.parse_args()
    train_dataset = args.train
    valid_dataset = args.valid
    train_dataset = pd.read_csv(train_dataset, header=None)
    valid_dataset = pd.read_csv(valid_dataset, header=None)
    expected_valid_dataset = valid_dataset[0].to_numpy()
    expected_valid_dataset_onehot = pd.get_dummies(valid_dataset[0], dtype=int).to_numpy()
    valid_dataset = valid_dataset.drop(columns=0).to_numpy()

    epochs = 300
    mlp = MLP(30)
    expected = train_dataset[0].to_numpy()
    expected_onehot = pd.get_dummies(train_dataset[0], dtype=int).to_numpy()
    inputs = train_dataset.drop(columns=0).to_numpy()
    val_losses = []
    val_accuracies = []
    losses = []
    accuracies = []
    
    for i in range(epochs):
        outputs = mlp.forward(inputs)
        outputs_interpreted = interpret_proba(outputs)
        mlp.retropropagate(outputs, expected_onehot)

        predictions, predictions_proba = predict(mlp, valid_dataset, expected_valid_dataset)

        loss = mlp.cross_entropy_loss(outputs, expected_onehot)
        accuracy = compute_accuracy(outputs_interpreted, expected)
        losses.append(loss)
        accuracies.append(accuracy)
        
        val_loss = mlp.cross_entropy_loss(predictions_proba, expected_valid_dataset_onehot)
        val_accuracy = compute_accuracy(predictions, expected_valid_dataset)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        if (i + 1) % 50 == 0:
            print(f"Epoch {i + 1}/{epochs} - Accuracy: {accuracy:.2f}% - Loss: {loss:.4f} - Val_Accuracy: {val_accuracy:.2f}% - Val_Loss: {val_loss:.4f}")
    plot_all_metrics(val_accuracies, val_losses, accuracies, losses)
    export_weights_and_biases(mlp, "model")