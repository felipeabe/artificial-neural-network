"""
Arquivo neural_network.py
Contém a implementação de uma Rede Neural Artificial para tarefas de classificação binária,
classificação multiclasse e regressão.
"""

import numpy as np

# Funções de ativação
# Funções de ativação introduzem não-linearidade no modelo, permitindo a aproximação
# de funções complexas. Cada função tem suas características e usos específicos.
def ativacao_sigmoid(x):
    # A função sigmoid mapeia qualquer valor real para o intervalo (0, 1),
    # sendo amplamente utilizada para problemas de classificação binária.
    return 1 / (1 + np.exp(-x))

def ativacao_relu(x):
    # A função ReLU (Rectified Linear Unit) retorna 0 para valores negativos
    # e o próprio valor para positivos. É amplamente utilizada em redes profundas
    # devido à sua simplicidade e eficiência computacional.
    return np.maximum(0, x)

def ativacao_tanh(x):
    # A função tanh mapeia valores para o intervalo (-1, 1), sendo útil
    # para normalizar saídas em torno de zero, facilitando o aprendizado.
    return np.tanh(x)

# Derivadas das funções de ativação
# As derivadas são utilizadas durante o algoritmo de retropropagação para calcular
# os gradientes necessários para a atualização dos pesos.
def derivada_sigmoid(x):
    sig = ativacao_sigmoid(x)
    return sig * (1 - sig)

def derivada_relu(x):
    return np.where(x > 0, 1, 0)

def derivada_tanh(x):
    return 1 - np.tanh(x)**2

# Funções de perda
# A função de perda mede o desempenho do modelo, comparando as previsões com os valores reais.
def perda_mse(y_real, y_predito):
    # Mean Squared Error (Erro Médio Quadrático) é utilizado em tarefas de regressão,
    # penalizando grandes diferenças entre valores previstos e reais.
    return np.mean((y_real - y_predito) ** 2)

def perda_cross_entropy(y_real, y_predito):
    # Cross-Entropy Loss é utilizada para tarefas de classificação, medindo a discrepância
    # entre distribuições de probabilidade previstas e reais.
    return -np.mean(y_real * np.log(y_predito + 1e-8) + (1 - y_real) * np.log(1 - y_predito + 1e-8))

# Classe Rede Neural
# Implementação de uma rede neural feedforward com uma única camada oculta,
# utilizando backpropagation para ajustar os pesos com base na função de perda.
class RedeNeural:
    def __init__(self, entrada_tamanho, oculta_tamanho, saida_tamanho, ativacao="relu", perda="mse"):
        # Inicializar pesos e biases de forma aleatória para evitar simetria
        self.pesos_entrada_oculta = np.random.randn(entrada_tamanho, oculta_tamanho) * 0.01
        self.bias_oculta = np.zeros((1, oculta_tamanho))
        self.pesos_oculta_saida = np.random.randn(oculta_tamanho, saida_tamanho) * 0.01
        self.bias_saida = np.zeros((1, saida_tamanho))

        # Configurar função de ativação
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

        # Configurar função de perda
        if perda == "mse":
            self.funcao_perda = perda_mse
        elif perda == "cross_entropy":
            self.funcao_perda = perda_cross_entropy
        else:
            raise ValueError("Função de perda inválida! Use 'mse' ou 'cross_entropy'.")

    def propagacao_frente(self, X):
        # Propagação para frente
        # Calcula as ativações da camada oculta e da saída.
        self.z1 = np.dot(X, self.pesos_entrada_oculta) + self.bias_oculta
        self.a1 = self.funcao_ativacao(self.z1)
        self.z2 = np.dot(self.a1, self.pesos_oculta_saida) + self.bias_saida
        self.a2 = ativacao_sigmoid(self.z2)  # A saída final é sempre sigmoid
        return self.a2

    def retropropagacao(self, X, y, taxa_aprendizado):
        # Retropropagação
        # Calcula os gradientes e atualiza os pesos para minimizar a função de perda.
        m = X.shape[0]

        # Erro na saída
        erro_saida = self.a2 - y

        # Gradientes da saída
        dw2 = np.dot(self.a1.T, erro_saida) / m
        db2 = np.sum(erro_saida, axis=0, keepdims=True) / m

        # Erro na camada oculta
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
        # Calcula a perda utilizando a função configurada.
        return self.funcao_perda(y_real, y_predito)

    def treinar(self, X, y, epocas, taxa_aprendizado):
        # Processo de treinamento da rede
        # Itera por múltiplas épocas, ajustando os pesos a cada iteração.
        for epoca in range(epocas):
            # Passo de propagação
            y_predito = self.propagacao_frente(X)

            # Calcular perda
            perda = self.calcular_perda(y, y_predito)

            # Passo de retropropagação
            self.retropropagacao(X, y, taxa_aprendizado)

            # Exibir progresso
            if epoca % 100 == 0:
                print(f"Época {epoca}/{epocas}, Perda: {perda:.4f}")

    def prever(self, X):
        # Realizar predição
        # Retorna os valores previstos pela rede para classificação binária.
        y_predito = self.propagacao_frente(X)
        return (y_predito > 0.5).astype(int)  # Para classificação binária
