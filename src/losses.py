import numpy as np

# Função de perda Cross-Entropy
def cross_entropy_loss(y_true, y_pred):
    """
    Função de perda Cross-Entropy.
    Mede a discrepância entre as previsões do modelo (y_pred) e os valores reais (y_true).
    Ideal para problemas de classificação.
    """
    epsilon = 1e-8  # Evitar log de 0
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Derivada da função Cross-Entropy
def cross_entropy_derivative(y_true, y_pred):
    """
    Derivada da função de perda Cross-Entropy.
    """
    epsilon = 1e-8  # Evitar divisão por 0
    return -(y_true / (y_pred + epsilon)) + ((1 - y_true) / (1 - y_pred + epsilon))

# Função de perda MSE (Mean Squared Error)
def mse_loss(y_true, y_pred):
    """
    Função de perda Mean Squared Error (MSE).
    Mede o erro médio quadrático entre as previsões (y_pred) e os valores reais (y_true).
    Ideal para problemas de regressão.
    """
    return np.mean((y_true - y_pred) ** 2)

# Derivada da função MSE
def mse_derivative(y_true, y_pred):
    """
    Derivada da função de perda Mean Squared Error (MSE).
    """
    return 2 * (y_pred - y_true) / y_true.size
