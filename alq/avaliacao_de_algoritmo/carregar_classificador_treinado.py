import pickle
import numpy as np

with open("./../../credit.pkl", "rb") as f:
    x_credit_treinamento, y_credito_treinamento, x_credit_teste, y_credit_teste = (
        pickle.load(f)
    )
x_credit = np.concatenate((x_credit_treinamento, x_credit_teste), axis=0)
y_credit = np.concatenate((y_credito_treinamento, y_credit_teste), axis=0)
rede_neural = pickle.load(
    open("../../dump_finalizado/rede_neural_finalizado.sav", "rb")
)
arvore = pickle.load(open("../../dump_finalizado/arvore_finalizado.sav", "rb"))
svm = pickle.load(open("../../dump_finalizado/svm_finalizado.sav", "rb"))
novo_registro = x_credit[1999]

novo_registro = novo_registro.reshape(1, -1)
print(rede_neural.predict(novo_registro))
print(arvore.predict(novo_registro))
print(svm.predict(novo_registro))
