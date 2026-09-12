# Telco Customer Churn — Pipeline de Machine Learning

Obs: Utilizei o projeto para praticar meu python que ainda está em um nível intermediário e estou buscando melhorá-lo, nas etapas do projeto fui documentando e sempre aprendendo algo novo sobre pandas e sobre o próprio python em si. Lembrando que o projeto ainda tem continuação chegando até a etapa 5/6 do pipeline de Machine Learning.

Projeto de prática de pipeline de Machine Learning, passando pelas etapas de 
**Data Collection**, **Data Preparation** e **Model Experimentation**, aplicado 
a um problema de classificação com dados reais (não sintéticos).

## Dataset

**Telco Customer Churn** (IBM / Kaggle) — dados reais de clientes de uma operadora 
de telecomunicações, com o objetivo de prever se um cliente vai cancelar o serviço 
(churn) ou não.

- Fonte: [Kaggle - blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7043 registros, 21 colunas (na versão bruta)

## Estrutura do repositório


## 01 - Data Collection

Etapa de diagnóstico do dataset bruto, sem nenhuma alteração nos dados. Foi verificado:

- Dimensão do dataset: 7043 linhas x 21 colunas
- Tipos de dado: 18 colunas `object`, 2 `int64`, 1 `float64`
- Nenhum valor nulo identificado inicialmente via `.isnull().sum()`
- Nenhuma linha duplicada

## 02 - Data Preparation

Com base no diagnóstico da etapa anterior, foram identificados e tratados os 
seguintes pontos:

**Coluna `TotalCharges` como `object`**
Apesar de conter números, a coluna estava sendo lida como texto. Ao investigar, 
identificamos 11 valores ausentes "escondidos" (armazenados como espaço em branco 
em vez de `NaN` reconhecido pelo pandas), o que fazia a coluna inteira ser 
interpretada como texto. Corrigido com `pd.to_numeric(..., errors='coerce')`.

Reparamos uma possível correlação entre esses 11 valores ausentes e clientes com 
`tenure` igual a zero (clientes recém-cadastrados, sem cobrança ainda gerada). 
Fica registrada como hipótese para análise futura — no momento, não parece viável 
aprofundar essa investigação.

Como os 11 valores representam apenas ~0,16% do total de registros, não teriam 
impacto estatisticamente relevante para os modelos futuros — optou-se por removê-los 
(`dropna(subset=['TotalCharges'])`). Dataset passou de 7043 para 7032 linhas.

**Coluna `customerID`**
Removida por ser um identificador único por cliente, sem valor preditivo para o 
modelo. Dataset passou de 21 para 20 colunas.

**Distribuição da variável alvo (`Churn`)**
Analisada via `.value_counts()`: aproximadamente 73% dos clientes não dão churn 
e 27% dão churn — um desbalanceamento moderado. Por conta disso, métricas como 
acurácia pura serão enganosas na etapa de modelagem (um modelo que sempre prevê 
"não-churn" já acertaria 73% sem aprender nada de fato); métricas como F1-score, 
precision/recall e matriz de confusão serão priorizadas na avaliação.

**Separação em treino e teste**
Antes do encoding, o dataset foi dividido em `X_train`/`X_test` e `y_train`/`y_test` 
(`test_size=0.3`, `random_state=47`, `stratify=y`), para que qualquer transformação 
subsequente (encoding, normalização) fosse ajustada apenas com base no treino, evitando 
data leakage. O `stratify=y` garantiu que a proporção de ~73%/27% de `Churn` fosse 
mantida tanto no treino quanto no teste.

**Encoding das variáveis categóricas**
Aplicado via `ColumnTransformer` + `OneHotEncoder(drop='first')`, ajustado (`fit`) 
apenas no `X_train` e reaplicado no `X_test` apenas com `transform`. O `drop='first'` 
evita a "armadilha da variável dummy" (multicolinearidade), removendo uma categoria de 
referência por coluna original.

**Normalização das variáveis numéricas**
Como a etapa de Model Experimentation prevista inclui modelos sensíveis à escala dos 
dados (Regressão Logística, KNN, SVM) além de modelos baseados em árvore (Árvore de 
Decisão, Random Forest), as colunas numéricas (`tenure`, `MonthlyCharges`, 
`TotalCharges`) foram normalizadas com `StandardScaler` (média 0, desvio padrão 1), 
incorporado ao mesmo `ColumnTransformer` usado no encoding — também ajustado apenas 
no treino.

Após essas transformações, `X_train_processado` e `X_test_processado` ficaram com 
30 colunas cada, prontos para a etapa de modelagem.

## 03 - Model Experimentation

Com o dataset já tratado e pré-processado na etapa anterior, esta etapa teve como 
objetivo instanciar e treinar diferentes algoritmos de classificação, comparando 
seu desempenho para decidir qual utilizar no projeto final.

**Modelos testados:** Regressão Logística, Árvore de Decisão, Random Forest, KNN e SVM, 
todos com `random_state=47` (quando aplicável) para reprodutibilidade.

**Metodologia de avaliação:** dado o desbalanceamento da variável alvo identificado 
na etapa 2 (~73% não-churn / 27% churn), a acurácia isolada não foi utilizada como 
critério de decisão — um modelo que sempre prevê "não-churn" já acertaria ~73% sem 
aprender nenhum padrão real. Priorizou-se o F1-score da classe `Yes` como métrica 
principal de ranqueamento, por representar o equilíbrio entre precision (evitar 
falsos alarmes) e recall (não deixar passar clientes que realmente cancelariam).

### Resultados comparativos

| Modelo               | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) |
|-----------------------|----------|------------------|----------------|------------------|
| Regressão Logística   | 0.801    | 0.645            | 0.561          | 0.601            |
| KNN                   | 0.763    | 0.558            | 0.515          | 0.536            |
| Random Forest         | 0.780    | 0.614            | 0.467          | 0.530            |
| SVM                   | 0.790    | 0.656            | 0.439          | 0.526            |
| Árvore de Decisão     | 0.724    | 0.482            | 0.503          | 0.492            |

### Modelo selecionado: Regressão Logística

Obteve o maior F1-score (0.60) e o melhor recall (0.56) da classe `Yes` entre os 5 
modelos testados. Sua precision (0.645) é mais que o dobro da precision esperada de 
um classificador aleatório nesse dataset (~0.27, dado o desbalanceamento) — indicando 
que o modelo captura padrões reais, não apenas ruído.

**Matriz de confusão:**
          Previsto: No   Previsto: Yes
          Real: No 1376 173
          Real: Yes 246 315


**Limitações identificadas:** mesmo sendo o melhor entre os 5, o modelo ainda deixa 
de identificar 246 dos 561 clientes que realmente cancelaram (recall de 56%) — uma 
limitação relevante para o objetivo de negócio de retenção, a ser explorada em 
etapas futuras (ajuste de hiperparâmetros, engenharia de atributos, ou balanceamento 
de classes).

**Hipótese para a diferença de performance:** o bom desempenho relativo de um modelo 
linear simples (Regressão Logística) frente a modelos mais complexos sugere que as 
relações entre as features e o churn, após o pré-processamento, podem ser majoritariamente 
lineares — hipótese a ser validada em etapas futuras.

## 04 - Data Pipeline & Feature Engineering

*Em andamento.*

## 05 - Model Building

*Em andamento.*

