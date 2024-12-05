"""
Arquivo neural_network.py
Contém a implementação de uma Rede Neural Artificial para tarefas de classificação binária,
classificação multiclasse e regressão. Implementado para fins educacionais, utilizando 
conceitos básicos de aprendizado de máquina, como propagação para frente, retropropagação 
e atualização de pesos.
"""

import numpy as np

# Funções de ativação
# Introduzem não-linearidade no modelo, permitindo que a rede aprenda padrões complexos.
def ativacao_sigmoid(x):
    """Função Sigmoid: mapeia valores reais para o intervalo (0, 1)."""
    return 1 / (1 + np.exp(-x))

def ativacao_relu(x):
    """Função ReLU: retorna 0 para valores negativos e o valor original para positivos."""
    return np.maximum(0, x)

def ativacao_tanh(x):
    """Função Tanh: mapeia valores para o intervalo (-1, 1), útil para normalização."""
    return np.tanh(x)

# Derivadas das funções de ativação
# Necessárias para calcular os gradientes durante a retropropagação.
def derivada_sigmoid(x):
    """Derivada da Sigmoid."""
    sig = ativacao_sigmoid(x)
    return sig * (1 - sig)

def derivada_relu(x):
    """Derivada da ReLU."""
    return np.where(x > 0, 1, 0)

def derivada_tanh(x):
    """Derivada da Tanh."""
    return 1 - np.tanh(x)**2

# Funções de perda
# Medem a discrepância entre as previsões do modelo e os valores reais.
def perda_mse(y_real, y_predito):
    """Erro Médio Quadrático: penaliza grandes diferenças entre valores reais e previstos."""
    return np.mean((y_real - y_predito) ** 2)

def perda_cross_entropy(y_real, y_predito):
    """Cross-Entropy: mede a discrepância entre distribuições de probabilidade."""
    return -np.mean(y_real * np.log(y_predito + 1e-8) + (1 - y_real) * np.log(1 - y_predito + 1e-8))

# Classe Rede Neural
class RedeNeural:
    """
    Implementação de uma Rede Neural Artificial com uma camada oculta.
    Suporta propagação para frente, retropropagação e treinamento com gradiente descendente.
    """
    def __init__(self, entrada_tamanho, oculta_tamanho, saida_tamanho, ativacao="relu", perda="mse"):
        # Inicializar pesos e biases de forma aleatória para evitar simetria.
        self.pesos_entrada_oculta = np.random.randn(entrada_tamanho, oculta_tamanho) * 0.01
        self.bias_oculta = np.zeros((1, oculta_tamanho))
        self.pesos_oculta_saida = np.random.randn(oculta_tamanho, saida_tamanho) * 0.01
        self.bias_saida = np.zeros((1, saida_tamanho))

        # Configuração das funções de ativação e perda.
        if ativacao == "relu":
            self.funcao_ativacao = ativacao_relu
            self.derivada_ativacao = derivada_relu
        elif ativacao == "sigmoid":
            self.funcao_ativacao = ativacao_sigmoid
            self.derivada_ativacao = derivada_sigmoid
        elif ativacao == "tanh":
            self.funcao_ativacao = ativacao_tanh
            self.derivada_ativacao = derivada_tanh
        else:
            raise ValueError("Função de ativação inválida! Use 'relu', 'sigmoid' ou 'tanh'.")

        if perda == "mse":
            self.funcao_perda = perda_mse
        elif perda == "cross_entropy":
            self.funcao_perda = perda_cross_entropy
        else:
            raise ValueError("Função de perda inválida! Use 'mse' ou 'cross_entropy'.")

    def propagacao_frente(self, X):
        """
        Executa a propagação para frente.
        Calcula as ativações das camadas oculta e de saída.
        """
        self.z1 = np.dot(X, self.pesos_entrada_oculta) + self.bias_oculta
        self.a1 = self.funcao_ativacao(self.z1)
        self.z2 = np.dot(self.a1, self.pesos_oculta_saida) + self.bias_saida
        self.a2 = ativacao_sigmoid(self.z2)  # A saída final usa sigmoid.
        return self.a2

    def retropropagacao(self, X, y, taxa_aprendizado):
        """
        Executa a retropropagação.
        Calcula os gradientes e ajusta os pesos para minimizar a perda.
        """
        m = X.shape[0]

        # Erro da camada de saída
        erro_saida = self.a2 - y

        # Gradientes da camada de saída
        dw2 = np.dot(self.a1.T, erro_saida) / m
        db2 = np.sum(erro_saida, axis=0, keepdims=True) / m

        # Erro da camada oculta
        erro_oculta = np.dot(erro_saida, self.pesos_oculta_saida.T) * self.derivada_ativacao(self.z1)

        # Gradientes da camada oculta
        dw1 = np.dot(X.T, erro_oculta) / m
        db1 = np.sum(erro_oculta, axis=0, keepdims=True) / m

        # Atualizar pesos e biases
        self.pesos_oculta_saida -= taxa_aprendizado * dw2
        self.bias_saida -= taxa_aprendizado * db2
        self.pesos_entrada_oculta -= taxa_aprendizado * dw1
        self.bias_oculta -= taxa_aprendizado * db1

    def calcular_perda(self, y_real, y_predito):
        """
        Calcula a perda com base na função configurada.
        """
        return self.funcao_perda(y_real, y_predito)

    def treinar(self, X, y, epocas, taxa_aprendizado):
        """
        Treina a rede neural ajustando os pesos a cada época.
        """
        for epoca in range(epocas):
            y_predito = self.propagacao_frente(X)
            perda = self.calcular_perda(y, y_predito)
            self.retropropagacao(X, y, taxa_aprendizado)
            if epoca % 100 == 0:
                print(f"Época {epoca}/{epocas}, Perda: {perda:.4f}")

    def prever(self, X):
        """
        Realiza predições.
        Retorna a classificação binária baseada nas ativações da camada de saída.
        """
        y_predito = self.propagacao_frente(X)
        return (y_predito > 0.5).astype(int)
