import Orange
import Orange.classification
import Orange.evaluation
from collections import Counter

base_credit = Orange.data.Table("./../../database/credit_data_regras.csv")
majority = Orange.classification.MajorityLearner()
previsoes = Orange.evaluation.testing.TestOnTestData(
    base_credit, base_credit, [majority]
)
print(Orange.evaluation.CA(previsoes))
for registro in base_credit:
    print(registro.get_class())
print(Counter(str(registro.get_class()) for registro in base_credit))

# ---------------------- census --------------------------

base_census = Orange.data.Table("./../../database/census_regras.csv")
majority_census = Orange.classification.MajorityLearner()
previsoes_census = Orange.evaluation.testing.TestOnTestData(
    base_census, base_census, [majority_census]
)
print(Orange.evaluation.CA(previsoes_census))
print(Counter(str(registro.get_class()) for registro in base_census))
print(24720 / (24720 + 7841))
