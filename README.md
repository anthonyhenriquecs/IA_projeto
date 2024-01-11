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

#Criar a IA
#Importando os modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Arvore de decisão
modelo_arvore = RandomForestClassifier()
##treino
modelo_arvore.fit(x_treino,y_treino)
#KNN - Vizinhos proximos
modelo_KNN = KNeighborsClassifier()
##treino
modelo_KNN.fit(x_treino,y_treino)

#Testar os modelos e acuracia]
from sklearn.metrics import accuracy_score

prev_arvore = modelo_arvore.predict(x_teste)
prev_knn = modelo_KNN.predict(x_teste.to_numpy())

print(accuracy_score(y_teste, prev_arvore))
print(accuracy_score(y_teste, prev_knn))

# Fazer novas previsões
##melhor modelo foi a arvore##

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
display(tabela_novos_clientes)

#codficar novos clientes
tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])

#fazer as novas previsões
previsoes = modelo_arvore.predict(tabela_novos_clientes)
display(previsoes)


