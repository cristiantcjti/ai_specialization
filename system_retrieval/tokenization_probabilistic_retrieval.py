import nltk
import numpy as np
from rank_bm25 import BM25Okapi


documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]



def preprocess(text):
    # 1. Converter para minúsculas
    text = text.lower()

    # 2. Tokenizar
    tokens = nltk.word_tokenize(text)
    
    # 3. Filtrar apenas tokens alfanuméricos
    return [word for word in tokens if word.isalnum()]


tokenized_docs = [preprocess(doc) for doc in documents]
tokenized_docs

bm25 = BM25Okapi(tokenized_docs)

query = "machine learning"

def search_bm25(query, bm25):
    tokenized_query = preprocess(query)
    return bm25.get_scores(tokenized_query)

results = search_bm25(query=query, bm25=bm25)
results

sorted_results = np.argsort(results)[::-1]
for sorted_result in sorted_results:
    print(f"Document: {sorted_result}: {documents[sorted_result]}")

