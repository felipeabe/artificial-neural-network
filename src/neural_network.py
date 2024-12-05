"""
Arquivo neural_network.py
Implementação de uma Rede Neural Artificial para classificação binária, multiclasse e regressão.
Este código foi desenvolvido para fins acadêmicos e enfatiza conceitos fundamentais como 
propagação para frente, retropropagação e ajuste de pesos por gradiente descendente.
"""

import numpy as np

# Funções de ativação
# Introduzem não-linearidade no modelo, essencial para aproximar padrões complexos.
def ativacao_sigmoid(x):
    """Função Sigmoid: mapeia valores reais para o intervalo (0, 1)."""
    return 1 / (1 + np.exp(-x))

def ativacao_relu(x):
    """Função ReLU: retorna o valor original para positivos e 0 para negativos."""
    return np.maximum(0, x)

def ativacao_tanh(x):
    """Função Tanh: mapeia valores reais para o intervalo (-1, 1)."""
    return np.tanh(x)
    
def softmax(x):
    """
    Função de ativação Softmax.
    Retorna uma distribuição de probabilidades para uma entrada.
    """
    # Subtrair o valor máximo de x para maior estabilidade numérica
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
# Derivadas das funções de ativação
# Usadas na retropropagação para ajustar pesos com base nos gradientes.
def derivada_sigmoid(x):
    """Derivada da função Sigmoid."""
    sig = ativacao_sigmoid(x)
    return sig * (1 - sig)

def derivada_relu(x):
    """Derivada da função ReLU."""
    return np.where(x > 0, 1, 0)

def derivada_tanh(x):
    """Derivada da função Tanh."""
    return 1 - np.tanh(x)**2

# Funções de perda
# Avaliam a discrepância entre valores reais e previstos, guiando o aprendizado.
def perda_mse(y_real, y_predito):
    """Erro Médio Quadrático: penaliza diferenças quadráticas entre previsões e valores reais."""
    return np.mean((y_real - y_predito) ** 2)

def perda_cross_entropy(y_real, y_predito):
    """Cross-Entropy: mede a divergência entre distribuições de probabilidade."""
    return -np.mean(y_real * np.log(y_predito + 1e-8) + (1 - y_real) * np.log(1 - y_predito + 1e-8))

# Classe Rede Neural
class RedeNeural:
    """
    Rede Neural Artificial com uma única camada oculta.
    Implementa propagação para frente, retropropagação e ajuste de pesos via gradiente descendente.
    """
    def __init__(self, entrada_tamanho, oculta_tamanho, saida_tamanho, ativacao="relu", perda="mse"):
        # Inicialização de pesos e biases para evitar simetria
        self.pesos_entrada_oculta = np.random.randn(entrada_tamanho, oculta_tamanho) * 0.01
        self.bias_oculta = np.zeros((1, oculta_tamanho))
        self.pesos_oculta_saida = np.random.randn(oculta_tamanho, saida_tamanho) * 0.01
        self.bias_saida = np.zeros((1, saida_tamanho))

        # Configuração de funções de ativação e perda
        self.historico_perda = []  # Armazena a perda de cada época
        if ativacao == "relu":
            self.funcao_ativacao = ativacao_relu
            self.derivada_ativacao = derivada_relu
        elif ativacao == "sigmoid":
            self.funcao_ativacao = ativacao_sigmoid
            self.derivada_ativacao = derivada_sigmoid
        elif ativacao == "tanh":
            self.funcao_ativacao = ativacao_tanh
            self.derivada_ativacao = derivada_tanh
        elif ativacao == "softmax":
            self.funcao_ativacao = softmax
        else:
            raise ValueError("Função de ativação inválida! Escolha 'relu', 'sigmoid' ou 'tanh'.")

        if perda == "mse":
            self.funcao_perda = perda_mse
        elif perda == "cross_entropy":
            self.funcao_perda = perda_cross_entropy
        else:
            raise ValueError("Função de perda inválida! Escolha 'mse' ou 'cross_entropy'.")

    def propagacao_frente(self, X):
        """
        Propagação para frente: calcula ativações da camada oculta e de saída.
        """
        self.z1 = np.dot(X, self.pesos_entrada_oculta) + self.bias_oculta
        self.a1 = self.funcao_ativacao(self.z1)
        self.z2 = np.dot(self.a1, self.pesos_oculta_saida) + self.bias_saida
        self.a2 = ativacao_sigmoid(self.z2)  # Sigmoid é usado na saída final
        return self.a2

    def retropropagacao(self, X, y, taxa_aprendizado):
        """
        Retropropagação: ajusta pesos e biases para reduzir a perda.
        """
        m = X.shape[0]
        y_predito = self.propagacao_frente(X)
        erro_saida = y_predito - y

        # Gradientes da camada de saída
        grad_w_saida = np.dot(self.a1.T, erro_saida) / m
        grad_b_saida = np.sum(erro_saida, axis=0, keepdims=True) / m

        # Gradientes da camada oculta
        erro_oculta = np.dot(erro_saida, self.pesos_oculta_saida.T) * self.derivada_ativacao(self.z1)
        grad_w_oculta = np.dot(X.T, erro_oculta) / m
        grad_b_oculta = np.sum(erro_oculta, axis=0, keepdims=True) / m

        # Atualizar pesos e biases
        self.pesos_oculta_saida -= taxa_aprendizado * grad_w_saida
        self.bias_saida -= taxa_aprendizado * grad_b_saida
        self.pesos_entrada_oculta -= taxa_aprendizado * grad_w_oculta
        self.bias_oculta -= taxa_aprendizado * grad_b_oculta

        # Armazenar perda
        perda = self.funcao_perda(y, y_predito)
        self.historico_perda.append(perda)

    def calcular_perda(self, y_real, y_predito):
        """
        Calcula a perda para o modelo.
        """
        return self.funcao_perda(y_real, y_predito)

    def treinar(self, X, y, epocas, taxa_aprendizado):
        """
        Treina a rede neural ajustando pesos a cada época.
        """
        for epoca in range(epocas):
            self.retropropagacao(X, y, taxa_aprendizado)
            if epoca % 100 == 0:
                print(f"Época {epoca}/{epocas}, Perda: {self.historico_perda[-1]:.4f}")

    def prever(self, X):
        """
        Predições para classificação binária.
        """
        y_predito = self.propagacao_frente(X)
        return (y_predito > 0.5).astype(int)
