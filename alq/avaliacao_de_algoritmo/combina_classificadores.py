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
novo_registro = x_credit[0]


novo_registro = novo_registro.reshape(1, -1)
resposta_rede = rede_neural.predict(novo_registro)
resposta_arvore = arvore.predict(novo_registro)
resposta_svm = svm.predict(novo_registro)

paga = 0
nao_paga = 0
if resposta_rede[0] == 1:
    nao_paga += 1
else:
    paga += 1
if resposta_arvore[0] == 1:
    nao_paga += 1
else:
    paga += 1
if resposta_svm[0] == 1:
    nao_paga += 1
else:
    paga += 1
if paga > nao_paga:
    print("cliente pagara o empresetimo")
elif paga == nao_paga:
    print("empate")
else:
    print("cliente não pagara o emprestimo")
