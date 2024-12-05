import numpy as np

# Função Sigmoid
def sigmoid(x):
    """
    Função de ativação Sigmoid.
    Retorna valores no intervalo (0, 1).
    """
    return 1 / (1 + np.exp(-x))

# Derivada da função Sigmoid
def sigmoid_derivative(x):
    """
    Derivada da função Sigmoid.
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

# Função ReLU (Rectified Linear Unit)
def relu(x):
    """
    Função de ativação ReLU.
    Retorna valores positivos ou zero.
    """
    return np.maximum(0, x)

# Derivada da função ReLU
def relu_derivative(x):
    """
    Derivada da função ReLU.
    Retorna 1 para x > 0 e 0 para x <= 0.
    """
    return np.where(x > 0, 1, 0)

# Função Tanh (Tangente Hiperbólica)
def tanh(x):
    """
    Função de ativação Tanh.
    Retorna valores no intervalo (-1, 1).
    """
    return np.tanh(x)

# Derivada da função Tanh
def tanh_derivative(x):
    """
    Derivada da função Tanh.
    """
    return 1 - np.tanh(x) ** 2
