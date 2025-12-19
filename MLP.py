import numpy as np

class MLP:
    input_size: int
    hidden_layer_sizes: list[int]
    # first_hidden_size: int
    # second_hidden_size: int
    final_layer_size: int = 2
    hidden_layers_nb: int

    inputs: np.ndarray
    hidden_layers_activations: list[np.ndarray]
    outputs: np.ndarray

    hidden_weights: list[np.ndarray]
    hidden_layers_biases: list[np.ndarray]
    final_layer_biases: np.ndarray

    learning_rate: float


    def __init__(self, input_size, hidden_layers_nb=2, learning_rate=0.01):
        if hidden_layers_nb < 1:
            raise ValueError("There must be at least one hidden layer")
        self.hidden_layers_nb = hidden_layers_nb
        self.learning_rate = learning_rate
        self.input_size = input_size
        
        # Initialize lists
        self.hidden_layer_sizes = []
        self.hidden_weights = []
        self.hidden_layers_biases = []
        self.hidden_layers_activations = []
        
        # Define hidden layer sizes
        self.hidden_layer_sizes.append(input_size * 2)
        for i in range(1, hidden_layers_nb):
            self.hidden_layer_sizes.append(input_size)
        
        self.final_layer_size = 2
        
        # Initialize weights and biases for each hidden layer
        for i in range(hidden_layers_nb):
            if i == 0:
                # Input to first hidden layer
                self.hidden_weights.append(np.random.randn(self.input_size, self.hidden_layer_sizes[0]))
            else:
                # Hidden to hidden
                self.hidden_weights.append(np.random.randn(self.hidden_layer_sizes[i - 1], self.hidden_layer_sizes[i]))
            
            # Biases for this hidden layer
            self.hidden_layers_biases.append(np.zeros(self.hidden_layer_sizes[i]))
        
        # Last hidden layer to output
        self.hidden_weights.append(np.random.randn(self.hidden_layer_sizes[-1], self.final_layer_size))
        self.final_layer_biases = np.zeros(self.final_layer_size)
        # self.hidden_hidden_weights = np.random.randn(self.first_hidden_size, self.second_hidden_size)
        # self.hidden_output_weights = np.random.randn(self.second_hidden_size, self.final_layer_size)
        # self.first_hidden_biases = np.zeros(self.first_hidden_size)
        # self.second_hidden_biases = np.zeros(self.second_hidden_size)
        # self.final_layer_biases = np.zeros(self.final_layer_size)

    def forward(self, inputs: np.array) -> np.array:
        self.inputs = inputs
        self.hidden_layers_activations = []
        
        for i in range(self.hidden_layers_nb):
            if i == 0:
                # First hidden layer: input -> hidden
                layer_input = np.dot(inputs, self.hidden_weights[0]) + self.hidden_layers_biases[0]
                self.hidden_layers_activations.append(self.sigmoid(layer_input))
            else:
                # Hidden -> hidden
                layer_input = np.dot(self.hidden_layers_activations[i - 1], self.hidden_weights[i]) + self.hidden_layers_biases[i]
                self.hidden_layers_activations.append(self.sigmoid(layer_input))
        
        # Final layer: last hidden -> output
        final_layer_input = np.dot(self.hidden_layers_activations[-1], self.hidden_weights[self.hidden_layers_nb]) + self.final_layer_biases
        self.outputs = self.softmax(final_layer_input)
        
        return self.outputs
        # first_hidden_input = np.dot(inputs, self.input_hidden_weights) + self.first_hidden_biases
        # self.first_hidden_activation = self.sigmoid(first_hidden_input)
        # second_hidden_input = np.dot(self.first_hidden_activation, self.hidden_hidden_weights) + self.second_hidden_biases
        # self.second_hidden_activation = self.sigmoid(second_hidden_input)
        # final_layer_input = np.dot(self.second_hidden_activation, self.hidden_output_weights) + self.final_layer_biases
        # self.outputs = self.softmax(final_layer_input)
        # return self.outputs

    def retropropagate(self, outputs: np.array, expected: np.array):
        # Gradient of loss w.r.t output
        grad_output = outputs - expected
        batch_size = self.inputs.shape[0]
        
        # Gradient for last hidden -> output weights
        grad_weights_output = np.dot(self.hidden_layers_activations[-1].T, grad_output) / batch_size
        self.hidden_weights[self.hidden_layers_nb] -= self.learning_rate * grad_weights_output
        self.final_layer_biases -= self.learning_rate * np.mean(grad_output, axis=0)
        
        # Backpropagate error to last hidden layer
        error_hidden = np.dot(grad_output, self.hidden_weights[self.hidden_layers_nb].T)
        grad_hidden = error_hidden * self.sigmoid(self.hidden_layers_activations[-1], derivative=True)
        
        # Backpropagate through all hidden layers in reverse order
        for i in reversed(range(self.hidden_layers_nb)):
            if i == 0:
                # First hidden layer: input -> hidden
                grad_weights = np.dot(self.inputs.T, grad_hidden) / batch_size
                self.hidden_weights[0] -= self.learning_rate * grad_weights
                self.hidden_layers_biases[0] -= self.learning_rate * np.mean(grad_hidden, axis=0)
            else:
                # Middle hidden layers
                grad_weights = np.dot(self.hidden_layers_activations[i - 1].T, grad_hidden) / batch_size
                self.hidden_weights[i] -= self.learning_rate * grad_weights
                self.hidden_layers_biases[i] -= self.learning_rate * np.mean(grad_hidden, axis=0)
                
                # Backpropagate to previous layer
                error_hidden = np.dot(grad_hidden, self.hidden_weights[i].T)
                grad_hidden = error_hidden * self.sigmoid(self.hidden_layers_activations[i - 1], derivative=True)   


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