import numpy as np

class MLP:
    input_size: int
    first_hidden_size: int
    second_hidden_size: int
    final_layer_size: int = 2
    hidden_layers_nb: int

    inputs: np.ndarray
    first_hidden_activation: np.ndarray
    second_hidden_activation: np.ndarray
    outputs: np.ndarray

    input_hidden_weights: np.ndarray
    hidden_hidden_weights: np.ndarray
    hidden_output_weights: np.ndarray

    first_hidden_biases: np.ndarray
    second_hidden_biases: np.ndarray
    final_layer_biases: np.ndarray

    learning_rate: float


    def __init__(self, input_size, hidden_layers_nb=2, learning_rate=0.01):
        self.hidden_layers_nb = hidden_layers_nb
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.first_hidden_size = input_size * 2
        self.second_hidden_size = input_size
        self.final_layer_size = 2
        # Initialize weights and biases here
        self.input_hidden_weights = np.random.randn(self.input_size, self.first_hidden_size)
        self.hidden_hidden_weights = np.random.randn(self.first_hidden_size, self.second_hidden_size)
        self.hidden_output_weights = np.random.randn(self.second_hidden_size, self.final_layer_size)
        self.first_hidden_biases = np.zeros(self.first_hidden_size)
        self.second_hidden_biases = np.zeros(self.second_hidden_size)
        self.final_layer_biases = np.zeros(self.final_layer_size)

    def forward(self, inputs: np.array) -> np.array:
        self.inputs = inputs
        first_hidden_input = np.dot(inputs, self.input_hidden_weights) + self.first_hidden_biases
        self.first_hidden_activation = self.sigmoid(first_hidden_input)
        second_hidden_input = np.dot(self.first_hidden_activation, self.hidden_hidden_weights) + self.second_hidden_biases
        self.second_hidden_activation = self.sigmoid(second_hidden_input)
        final_layer_input = np.dot(self.second_hidden_activation, self.hidden_output_weights) + self.final_layer_biases
        self.outputs = self.softmax(final_layer_input)
        return self.outputs

    def retropropagate(self, outputs: np.array, expected: np.array):
        batch_size = outputs.shape[0]
        
        # Gradient of loss w.r.t output
        grad_output = outputs - expected  # (batch_size, 2)
        
        # Gradient for hidden->output weights
        grad_hidden_output = np.dot(self.second_hidden_activation.T, grad_output) / batch_size  # (30, 2)
        
        # Backprop to second hidden layer
        error_second_hidden = np.dot(grad_output, self.hidden_output_weights.T)  # (batch_size, 30)
        grad_second_hidden = error_second_hidden * self.sigmoid(self.second_hidden_activation, derivative=True)  # (batch_size, 30)
        
        # Gradient for first_hidden->second_hidden weights
        grad_hidden_hidden = np.dot(self.first_hidden_activation.T, grad_second_hidden) / batch_size  # (60, 30)
        
        # Backprop to first hidden layer
        error_first_hidden = np.dot(grad_second_hidden, self.hidden_hidden_weights.T)  # (batch_size, 60)
        grad_first_hidden = error_first_hidden * self.sigmoid(self.first_hidden_activation, derivative=True)  # (batch_size, 60)
        
        # Gradient for input->first_hidden weights
        grad_input_hidden = np.dot(self.inputs.T, grad_first_hidden) / batch_size
        
        # Update weights and biases
        self.hidden_output_weights -= self.learning_rate * grad_hidden_output
        self.final_layer_biases -= self.learning_rate * np.mean(grad_output, axis=0)
        self.hidden_hidden_weights -= self.learning_rate * grad_hidden_hidden
        self.second_hidden_biases -= self.learning_rate * np.mean(grad_second_hidden, axis=0)
        self.input_hidden_weights -= self.learning_rate * grad_input_hidden
        self.first_hidden_biases -= self.learning_rate * np.mean(grad_first_hidden, axis=0)   


    def softmax(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))  # Stabilité numérique
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def sigmoid(self, input, derivative=False):
        if derivative:
            sig = self.sigmoid(input)
            return sig * (1 - sig)
        return 1 / (1 + np.exp(input * -1))
    
    def cross_entropy_loss(self, outputs: np.ndarray, expected: np.ndarray) -> float:
        return -np.mean(expected * np.log(outputs))