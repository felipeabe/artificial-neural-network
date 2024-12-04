"""
Arquivo neural_network.py
Contém a implementação de uma Rede Neural Artificial para tarefas de classificação binária,
classificação multiclasse e regressão.
"""

import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Derivadas das funções de ativação
def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# Funções de perda
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

# Classe da Rede Neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation="relu", loss="mse"):
        # Inicializa pesos e biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

        # Configurar função de ativação
        if activation == "relu":
            self.activation_func = relu
            self.activation_derivative = relu_derivative
        elif activation == "sigmoid":
            self.activation_func = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "tanh":
            self.activation_func = tanh
            self.activation_derivative = tanh_derivative
        else:
            raise ValueError("Ativação inválida! Escolha entre 'relu', 'sigmoid' ou 'tanh'.")

        # Configurar função de perda
        if loss == "mse":
            self.loss_func = mse_loss
        elif loss == "cross_entropy":
            self.loss_func = cross_entropy_loss
        else:
            raise ValueError("Função de perda inválida! Escolha entre 'mse' ou 'cross_entropy'.")

    def forward(self, X):
        # Propagação para frente
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = self.activation_func(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = sigmoid(self.z2)  # A saída final sempre usa sigmoid
        return self.a2

    def backward(self, X, y, learning_rate):
        # Retropropagação
        m = X.shape[0]

        # Erro da camada de saída
        output_error = self.a2 - y

        # Gradientes para a camada de saída
        dw2 = np.dot(self.a1.T, output_error) / m
        db2 = np.sum(output_error, axis=0, keepdims=True) / m

        # Erro da camada oculta
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.activation_derivative(self.z1)

        # Gradientes para a camada oculta
        dw1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0, keepdims=True) / m

        # Atualizar pesos e biases
        self.weights_hidden_output -= learning_rate * dw2
        self.bias_output -= learning_rate * db2
        self.weights_input_hidden -= learning_rate * dw1
        self.bias_hidden -= learning_rate * db1

    def compute_loss(self, y_true, y_pred):
        return self.loss_func(y_true, y_pred)

    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Calcular perda
            loss = self.compute_loss(y, y_pred)

            # Backward pass
            self.backward(X, y, learning_rate)

            # Exibir progresso
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        # Realiza predição
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)  # Para classificação binária

