import Orange
import Orange.classification
import Orange.evaluation


base_risco_credito = Orange.data.Table("./../../database/risco_credito_regras.csv")
print(base_risco_credito.domain)
cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)
for regras in regras_risco_credito.rule_list:
    print(regras)


previsoes = regras_risco_credito(
    [["boa", "alta", "nenhuma", "acima_35"], ["ruim", "alta", "adequada", "0_15"]]
)
print(previsoes)
print(base_risco_credito.domain.class_var.values)

for i in previsoes:
    print(i)


base_credit = Orange.data.Table("./../../database/credit_data_regras.csv")

base_dividida = Orange.evaluation.testing.sample(base_credit, n=0.25)

base_treinamento = base_dividida[1]
base_test = base_dividida[0]
regras_credit = cn2(base_treinamento)
for regras in regras_credit.rule_list:
    print(regras)

previsoes_credit = Orange.evaluation.testing.TestOnTestData(
    base_treinamento, base_test, [lambda testdara: regras_credit]
)
print(Orange.evaluation.CA(previsoes_credit))


base_censo = Orange.data.Table("./../../database/census_regras.csv")
base_censo_dividida = Orange.evaluation.testing.sample(base_censo, n=0.70)
print(base_censo_dividida[1], base_censo_dividida[0])
regras_censo = cn2(base_censo_dividida[1])
for regras in regras_censo.rule_list:
    print(regras)

print("here")
previsoes_censo = Orange.evaluation.testing.TestOnTestData(
    base_censo_dividida[1], base_censo_dividida[0], [lambda testdara: regras_censo]
)
print(Orange.evaluation.CA(previsoes_censo))
