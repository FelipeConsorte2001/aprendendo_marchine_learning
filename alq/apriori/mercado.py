import pandas as pd
from apyori import apriori


base_mercado1 = pd.read_csv("./../../database/mercado.csv", header=None)
transacoes = []
for i in range(len(base_mercado1)):
    transacoes.append(
        [str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])]
    )

print(transacoes)
regras = apriori(transacoes, min_support=0.3, min_confidence=0.8, min_lift=2)
resultados = list(regras)
print(len(resultados))
regra1 = []
regra2 = []
suport = []
confieca = []
lift = []
for resultado in resultados:
    suport_regra = resultado[1]
    s = resultado[1]
    result_rules = resultado[2]
    for result_rule in result_rules:
        a = list(result_rule[0])
        b = list(result_rule[1])
        c = result_rule[2]
        l = result_rule[3]
        regra1.append(a)
        regra2.append(b)
        confieca.append(c)
        suport.append(s)
        lift.append(l)

print(len(regra1), len(regra2), len(suport), len(confieca), len(lift))

rules_df = pd.DataFrame(
    {"a": regra1, "b": regra2, "suporte": suport, "confianca": confieca, "lift": lift}
)
print(rules_df)
