import nltk

nltk.download("punkt_tab")  # Modelo de tokenização punkt

text = "Machine Learning é um campo da inteligência artificial que permite que computadores aprendam."

#Tokenização por palavras
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

#Output: ['Machine', 'Learning', 'é', 'um', 'campo', 'da', 'inteligência', ...]


text = "Machine Learning é um campo da inteligência artificial. Ele permite que os sistemas façam previsões."

# Tokenização por sentenças
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)

# Output:
# ['Machine Learning é um campo da inteligência artificial.',
#  'Ele permite que os sistemas façam previsões.']


def preprocess(text):
    # Tokenização por palavras
    tokens = nltk.word_tokenize(text)

    # Retorna apenas tokens alfanuméricos em minúsculas
    return [word.lower() for word in tokens if word.isalnum()]


# Texto de exemplo
text = "Machine Learning é incrível!"
result = preprocess(text)
print(result)
# Output: ['machine', 'learning', 'é', 'incrível']
# Note que '!' foi removido por não ser alfanumérico


# Lista de documentos/sentenças
documents = [
    "Machine learning é o aprendizado automático.",
    "Ele permite que os sistemas façam previsões.",
    "É fundamental para IA moderna.",
]

# Aplicar preprocessing em todos os documentos
preprocessed_docs = [preprocess(doc) for doc in documents]
print(preprocessed_docs)

# Output:
# [['machine', 'learning', 'é', 'o', 'aprendizado', 'automático'],S
#  ['ele', 'permite', 'que', 'os', 'sistemas', 'façam', 'previsões'],
#  ['é', 'fundamental', 'para', 'ia', 'moderna']]
