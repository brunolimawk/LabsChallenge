
<h1 align="center"> Machine Learning com Python </h1>
A solução de machine learning apresentada é um modelo de classificação que tem como objetivo identificar se um e-mail é considerado spam ou não. O modelo é treinado utilizando um conjunto de dados que possui informações sobre os e-mails, incluindo a presença de palavras específicas, o tamanho do e-mail em termos de número de palavras, a frequência de ocorrência de determinadas palavras, entre outras.


# Índice

* [Introdução](#introdução)
* [Problema](#problema)
* [Objetivo](#objetivo)
* [Solução Proposta](#objetivo)
  
 
# Introdução

A solução implementada de machine learning com Python consiste em um modelo capaz de classificar e-mails como spam ou não spam. Para isso, foram utilizados diversos algoritmos de classificação, como Regressão Logística, Árvores de Decisão e Naive Bayes, que foram treinados em um conjunto de dados de e-mails previamente rotulados.
O conjunto de dados é pré-processado para transformar as informações em um formato adequado para a alimentação do modelo de aprendizado de máquina. Em seguida, um modelo de classificação é escolhido e treinado usando os dados de treinamento.
Após o treinamento, o modelo é avaliado usando um conjunto de dados de teste separado. A precisão do modelo é calculada a partir das previsões feitas pelo modelo para o conjunto de teste. Uma vez que o modelo foi avaliado e ajustado para obter o melhor desempenho possível, ele está pronto para ser utilizado em novos dados.
No exemplo apresentado, foram utilizados vários algoritmos de classificação, como Regressão Logística, Árvores de Decisão, e Naive Bayes. Também foram utilizadas técnicas de pré-processamento de dados, como vetorização de texto, que é uma técnica comum para processar dados de texto para alimentação de modelos de aprendizado de máquina.
Em geral, a solução de machine learning é um processo iterativo que envolve a escolha do conjunto de dados, a pré-processamento dos dados, a escolha e treinamento do modelo, e a avaliação do modelo. A escolha de um bom conjunto de dados e a seleção do modelo apropriado são essenciais para o sucesso da solução de machine learning.


# Problema

Classificação de e-mails como spam ou não spam. 


# Objetivo

Criar algoritimo através da programação em Python utilizando machine learning que seja capaz de classificar e-mails como spam ou não spam de forma automatizada e eficiente.  

# Solução  

```python
import pandas as pd

#Url contendo os dados que serão utilizamos para criação do modelo
urlDbGit = "https://raw.githubusercontent.com/brunolimawk/LabsChallenge/main/DataBase.csv"
 
#Busca os dados da tabela no formato CSV e cria um DataFrame
df = pd.read_csv(urlDbGit, delimiter=";")

#Criar colunas contendo o mínimo, a média, o máximo, o desvio padrão, a mediana e a variação de palavras da coluna Full_Text

df['Min_palavras'] = df['Full_Text'].str.split().apply(lambda x: len(x)).min()
df['Media_palavras'] = df['Full_Text'].str.split().apply(lambda x: len(x)).mean()
df['Max_palavras'] = df['Full_Text'].str.split().apply(lambda x: len(x)).max()
df['Desvio_palavras'] = df['Full_Text'].str.split().apply(lambda x: len(x)).std()
df['Mediana_palavras'] = df['Full_Text'].str.split().apply(lambda x: len(x)).median()
#df['Variacao_palavras'] = df['Full_Text'].str.split().apply(lambda x: max(x) - min(x)) 

#Definir a coluna "IsSpam" como variável alvo e transformá-la em binarios
df['IsSpam'] = df['IsSpam'].apply(lambda x: 1 if x == 'Yes' else 0)


#Separar o conjunto de dados em treinamento e teste
df['IsSpam'] = df['IsSpam'].apply(lambda x: 1 if x == 'Yes' else 0)

#Divide o modelo e prepara o treinamento
from sklearn.model_selection import train_test_split

treinamento, teste = train_test_split(df, test_size=0.2, random_state=42)


#Importar o CountVectorizer e utiliza-lo para transformar a coluna "Full_Text" em uma matriz
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
X_treinamento = count_vectorizer.fit_transform(treinamento['Full_Text'])
X_teste = count_vectorizer.transform(teste['Full_Text'])
y_treinamento = treinamento['IsSpam']
y_teste = teste['IsSpam']


#Adicionar dados de treinamento e teste
import numpy as np

X_treinamento = np.hstack((X_treinamento.toarray(), treinamento[['Min_palavras', 'Media_palavras', 'Max_palavras', 'Desvio_palavras', 'Mediana_palavras']].values))
X_teste = np.hstack((X_teste.toarray(), teste[['Min_palavras', 'Media_palavras', 'Max_palavras', 'Desvio_palavras', 'Mediana_palavras']].values))

#Treinar um modelo de classificação, utilizando Naive Bayes
from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(X_treinamento, y_treinamento)

#Avaliar o desempenho do modelo
from sklearn.metrics import accuracy_score 

#Fazer as previsões com o modelo treinado
y_pred = modelo.predict(X_teste)


# Calcular acurácia
acuracia = accuracy_score(y_teste, y_pred)

 
print("Acurácia:", acuracia)
```



