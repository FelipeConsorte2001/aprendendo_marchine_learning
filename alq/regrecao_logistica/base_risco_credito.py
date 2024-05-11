import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

with open("../../risco_credito.pkl", "rb") as f:
    x_risco_credito, y_risco_credito = pickle.load(f)


x_risco_credito = np.delete(x_risco_credito, [2, 7, 11], axis=0)
y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)


logistc_risco_credit = LogisticRegression(random_state=1)
logistc_risco_credit.fit(x_risco_credito, y_risco_credito)
print(logistc_risco_credit.intercept_)
print(logistc_risco_credit.coef_)

previsoes1 = logistc_risco_credit.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsoes1)
