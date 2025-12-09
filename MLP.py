import numpy as np

class MLP:
    input_size: int
    hidden_layers_nb: int
    input_hidden_weights: list[list[float]]
    hidden_hidden_weights: list[list[float]]
    hidden_output_weights: list[list[float]]
    first_hidden_biases: list[float]
    second_hidden_biases: list[float]
    output_biases: list[float]
    learning_rate: float

    def __init__(self, hidden_layers_nb=2):
        self.hidden_layers_nb = hidden_layers_nb

    def forward(self, inputs: list[float]) -> list[float]:
        #first_hidden_layer = relu(inputs * output_hidden_weigths + first_hidden_layer_biases)
        #second_hidden_layer = relu(first_hidden_layer * hidden_hidden_weights + second_layer_biases)
        #output = softmax(second_hidden_layer * hidden_output_weights + output_biases)
        pass

    def backward(self, inputs: list[float], expected: list[float]) -> None:
        pass

    def softmax(self, input):
        pass
    
    def relu(self, input):
        pass