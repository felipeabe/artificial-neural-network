import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Divide os dados em conjuntos de treino e teste.
    
    Parâmetros:
    - X: ndarray
        Dados de entrada (features).
    - y: ndarray
        Rótulos/saídas correspondentes aos dados de entrada.
    - test_size: float (0 a 1)
        Proporção dos dados que será usada para teste.
    - random_state: int (opcional)
        Semente para replicação dos resultados.
    
    Retorna:
    - X_train: ndarray
        Conjunto de treino (features).
    - X_test: ndarray
        Conjunto de teste (features).
    - y_train: ndarray
        Rótulos do conjunto de treino.
    - y_test: ndarray
        Rótulos do conjunto de teste.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize_data(data):
    """
    Normaliza os dados para o intervalo [0, 1].
    
    Parâmetros:
    - data: ndarray
        Dados numéricos a serem normalizados.
    
    Retorna:
    - ndarray
        Dados normalizados.
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia do modelo.
    
    Parâmetros:
    - y_true: ndarray
        Valores reais (rótulos).
    - y_pred: ndarray
        Previsões do modelo.
    
    Retorna:
    - float
        Acurácia como uma proporção (0 a 1).
    """
    return np.mean((y_pred > 0.5) == y_true)

def one_hot_encode(y, num_classes):
    """
    Realiza o One-Hot Encoding de rótulos.
    
    Parâmetros:
    - y: ndarray
        Rótulos (inteiros).
    - num_classes: int
        Número de classes distintas.
    
    Retorna:
    - ndarray
        Rótulos codificados no formato One-Hot.
    """
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

def one_hot_decode(y_one_hot):
    """
    Decodifica rótulos no formato One-Hot para inteiros.
    
    Parâmetros:
    - y_one_hot: ndarray
        Rótulos no formato One-Hot.
    
    Retorna:
    - ndarray
        Rótulos como inteiros.
    """
    return np.argmax(y_one_hot, axis=1)
