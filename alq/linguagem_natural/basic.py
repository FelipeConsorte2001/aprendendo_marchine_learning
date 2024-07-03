import nltk
import spacy

pln = spacy.load("pt_core_news_sm")
print(pln)
documento = pln("Estou aprendendo processamento de liguagem natural, curso em Curitiba")
print(type(documento))
for token in documento:
    print(token.text, token.pos_)

print("new\n")
for token in documento:
    print(token.text, token.lemma_)

doc = pln("encontrei encontraram encontrarão encontrariam cursando curso cursei")
words = [token.lemma_ for token in doc]
print(words)
nltk.download("rslp")
stemmer = nltk.stem.RSLPStemmer()
stemmer.stem("aprender")
for token in documento:
    print(token.text, token.lemma_, stemmer.stem(token.text))
