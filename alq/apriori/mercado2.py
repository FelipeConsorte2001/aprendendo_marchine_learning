import pandas as pd
from apyori import apriori


base_mercado2 = pd.read_csv("./../../database/mercado2.csv", header=None)
transacoes = []
for i in range(base_mercado2.shape[0]):
    transacoes.append(
        [str(base_mercado2.values[i, j]) for j in range(base_mercado2.shape[1])]
    )

regras = apriori(transacoes, min_support=0.003, min_confidence=0.2, min_lift=3)
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
