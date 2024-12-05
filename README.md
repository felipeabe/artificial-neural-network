# ![INE5664 - Aprendizado de Máquina](https://upload.wikimedia.org/wikipedia/commons/8/82/Ufsc_bras%C3%A3o.svg)

# Projeto Final - Implementação de uma Rede Neural Artificial

## INE5664-07238 (20242) - Aprendizado de Máquina  
*Universidade Federal de Santa Catarina (UFSC)*  
*Curso de Sistemas de Informação*  

### Alunos  
- *Diogo Henrique* (Matrícula: 16203891)  
- *Bruno Rafael Leal Machado* (Matrícula: 17100897)  
- *Felipe Hiroyuki Abe* (Matrícula: 21202327)  

### Professor  
- *Eduardo Camilo Inácio*

---

## Descrição do Projeto  
Este projeto consiste na implementação de uma Rede Neural Artificial (RNA) utilizando Python, com foco em prever dados para três tipos de tarefas:  
1. *Regressão*  
2. *Classificação Binária*  
3. *Classificação Multiclasse*

A RNA foi implementada do zero, seguindo os conceitos teóricos estudados na disciplina e utilizando recursos de baixo nível para respeitar os princípios matemáticos do modelo. Ferramentas como *NumPy, **Pandas, **Scikit-learn* e *Matplotlib* foram utilizadas para manipulação e visualização de dados.

O desempenho foi avaliado com datasets públicos e validados com métricas como Erro Médio Quadrático (MSE) para regressão e Acurácia para classificação.

---

## Estrutura do Repositório  
O repositório contém os seguintes itens:

- *Código-fonte da Rede Neural*:
  - Implementação da RNA em Python (neural_network.py) com suporte a retropropagação, gradiente descendente e funções de ativação/perda.
- *Notebooks Jupyter*:
  - Fluxo completo de treinamento e avaliação para os três tipos de modelos. Compatível com Google Colab.
- *Conjuntos de Dados*:
  - datasets/ com os arquivos utilizados para avaliação do modelo:
    - alzheimer.csv (Classificação Binária)
    - houses.csv (Regressão)
    - multiclass_dataset.csv (Classificação Multiclasse)
- *Arquivo README*:
  - Explicação do projeto, instruções de uso e detalhes da implementação.

---

## Modelos de Predição Treinados  
1. *Regressão*:
   - Previsão de valores contínuos (e.g., preço de imóveis).
   - Métrica: Erro Médio Quadrático (MSE).
2. *Classificação Binária*:
   - Previsão de duas classes (e.g., diagnóstico de Alzheimer).
   - Métrica: Acurácia.
3. *Classificação Multiclasse*:
   - Previsão de múltiplas categorias.
   - Métrica: Acurácia.

---

## Requisitos do Sistema  
- *Python 3.10 ou superior*  
- Bibliotecas:  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - Scikit-learn  

---

## Instruções de Uso  
1. *Clonar o Repositório*:
   ```bash
   git clone https://github.com/felipeabe/artificial-neural-network
   cd artificial-neural-network


