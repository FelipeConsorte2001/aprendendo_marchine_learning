import pandas as pd
from pyECLAT import ECLAT

base_mercado1 = pd.read_csv("./../../database/mercado.csv", header=None)
transacoes = []
for i in range(len(base_mercado1)):
    transacoes.append(
        [str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])]
    )
eclat = ECLAT(data=base_mercado1)
print(eclat.uniq_)
indices, suporte = eclat.fit(min_combination=1, min_support=0.3, max_combination=3)
print(indices)
print(suporte)
