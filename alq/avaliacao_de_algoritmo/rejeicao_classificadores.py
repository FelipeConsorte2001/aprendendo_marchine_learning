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
resposta_rede = rede_neural.predict(novo_registro)
resposta_arvore = arvore.predict(novo_registro)
resposta_svm = svm.predict(novo_registro)

probabilidade_rede_neural = rede_neural.predict_proba(novo_registro)
probabilidade_arvore = arvore.predict_proba(novo_registro)
probabilidade_svm = svm.predict_proba(novo_registro)
print("rede\n", probabilidade_rede_neural)
confianca_rede_neural = probabilidade_rede_neural.max()
print(confianca_rede_neural)

print("arvore\n", probabilidade_arvore)
confianca_arvore = probabilidade_arvore.max()
print(confianca_arvore)

print("svm\n", probabilidade_svm)
confianca_svm = probabilidade_svm.max()
print(confianca_svm)

paga = 0
nao_paga = 0
confianca_minima = 0.999999
algoritmo = 0
if confianca_rede_neural >= confianca_minima:
    algoritmo += 1
    if resposta_rede[0] == 1:
        nao_paga += 1
    else:
        paga += 1
if confianca_arvore >= confianca_minima:
    algoritmo += 1
    if resposta_arvore[0] == 1:
        nao_paga += 1
    else:
        paga += 1
if confianca_svm >= confianca_minima:
    algoritmo += 1
    if resposta_svm[0] == 1:
        nao_paga += 1
    else:
        paga += 1
if paga > nao_paga:
    print(f"cliente pagara o empresetimo, baseado em {algoritmo} algoritmos")
elif paga == nao_paga:
    print(f"empate, baseado em {algoritmo} algoritmos")
else:
    print(f"cliente não pagara o emprestimo, baseado em {algoritmo} algoritmos")
