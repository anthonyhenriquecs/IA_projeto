# IA_projeto

import pandas as pd

tabela = pd.read_csv("clientes.csv")
display(tabela)

#Vendo os dados importantes da tabelha
display(tabela.info())

#Tratando os dados e transformando em numero
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])

display(tabela.info())

#Aprendizadoda de maquina

#A coluna id_cliente não vai ser usada por ser um numero random

#y é a coluna que voce quer prever    
y = tabela["score_credito"]

#x é as colunas que vai usar pra fazer a previsão
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection  import train_test_split

x_treino, y_treino, x_teste, y_teste = train_test_split(x, y)
