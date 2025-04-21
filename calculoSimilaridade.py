import math
import numpy as np
from matplotlib import pyplot as plt
'''
#Número de vezes que a palavra aparece dividido pelo número de palavras no documento
def tf(word, document):
    return document.count(word) / len(document)

def idf(word, corpus):
    count_of_documents = len(corpus) + 1
    count_of_documents_with_word = sum([1 for doc in corpus if word in doc]) + 1
    IDF = np.log10(count_of_documents/count_of_documents_with_word) + 1
    return IDF

def TF_IDF(word, document, corpus):
    #print(f"{word} | TF: {tf(word, document):.4f}, IDF: {idf(word, corpus):.4f}, TF-IDF: {tf(word, document) }")

    return tf(word, document) #* idf(word, corpus) para ser linear

def tf_idf_vectorizer(corpus):
    if type(corpus[0]) == str:
        corpus = [corpus]  # se for uma só linha de palavras

    word_set = list(set(sum(corpus, []))) #lista de palavras unicas que aparecem em todos os documentos do corpus, ou seja, sem repetições
    print("word_set: ", word_set)
    word_to_index = {word: i for i, word in enumerate(word_set)}  #mapeia cada palavra do vocabulário para um índice numérico, ou seja a primeira da palavra de word_set será uma chave que corresponderá ao seu indice na lista que é 0 {"1ºpalavra":0,..}
    print(f"word_to_index: {word_to_index}")
    num_words = len(word_set) #Número total de palavras únicas (dimensão dos vetores)

    word_vectors = [] #onde vão ser guardados os vetores TF-IDF de cada documento
    for document in corpus:
        # for our new document create a new word vector
        new_word_vector = [0 for i in range(num_words)] #cria vetor de 0 do tamanho do vocabulário
        for word in document:
            # get the score
            tf_idf_score = TF_IDF(word, document, corpus)
            # next get the index for this word in our word vector
            word_index = word_to_index[word] #para saber em que posição do vetor deve ficar o score
            # populate the vector
            new_word_vector[word_index] = tf_idf_score #coloca o score na posição do vetor certo

        word_vectors.append(new_word_vector)
    print(f"word_vectors: {word_vectors}")

    return word_vectors


import numpy as np


def cosine_similarity(query_vector, doc_vectors):
    # Converter para arrays numpy para operações vetorizadas
    query_array = np.array(query_vector)
    doc_matrix = np.array(doc_vectors)

    # Calcular produto escalar (dot product) entre query e cada documento
    dot_products = np.dot(doc_matrix, query_array)

    # Calcular normas (magnitudes) dos vetores
    query_norm = np.linalg.norm(query_array)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    print(f"query_norm: {query_norm}, doc_norms: {doc_norms}")

    # Calcular similaridades (evitando divisão por zero)
    similarities = []
    for (dot, doc_norm) in zip(dot_products, doc_norms):
        if query_norm == 0 or doc_norm == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot / (query_norm * doc_norm)) #produto escalar dos vetores a dividir pelo produto das normas

    print(f"similarities: {similarities}")
    return similarities


def query_to_vector(query, word_to_index, corpus):

    num_words = len(word_to_index)
    query_vector = [0.0] * num_words

    for word in query:
        if word in word_to_index:
            # Calcula TF-IDF apenas para palavras presentes no vocabulário
            tf_idf_score = TF_IDF(word, query, corpus)
            print("Aqui:", tf_idf_score )
            query_vector[word_to_index[word]] = tf_idf_score

    print(f"query_vector: {query_vector}")
    return query_vector


# Função completa para calcular similaridades entre query e documentos
def calculate_similarities(query, corpus, word_set, word_to_index, doc_vectors):

    # Converter query para vetor TF-IDF
    query_vec = query_to_vector(query, word_to_index, corpus)

    # Calcular similaridades
    return cosine_similarity(query_vec, doc_vectors)

# Exemplo de uso integrado:
def search_with_custom_tfidf(query, documents):
    # Pré-processamento (tokenização já feita)
    tokenized_docs = [doc.split() for doc in documents]
    print(f"tokenized_docs: {tokenized_docs}")
    tokenized_query = query.split()

    # Construir vocabulário e vetores
    word_set = list(set(sum(tokenized_docs, [])))
    word_to_index = {word: i for i, word in enumerate(word_set)}
    doc_vectors = tf_idf_vectorizer(tokenized_docs)

    # Calcular similaridades
    sim_scores = calculate_similarities(tokenized_query, tokenized_docs,
                                        word_set, word_to_index, doc_vectors)

    # Ordenar resultados por relevância
    results = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    return results

# === TESTES ===
documents = [
    "o gato preto dorme no sofá",
    "o cão cão ladra no quintal",
    "o rato corre no campo",
    "gato e cão são animais de estimação",
    "ninguém mencionou quintal ou gato aqui"
]

for elem in documents:
    print(elem.split())

query = "cão sofá "

results = search_with_custom_tfidf(query, documents)

print(f"\n📊 Resultados da query: '{query}'\n")
for doc_id, score in results:
    print(f"📄 Documento {doc_id} | Score: {score:.4f}")
    print(f"   → {documents[doc_id]}\n")

import ujson
with open('scraper_results.json', 'r') as f1, open('scraper_results_with_abstracts.json', 'r') as f2:
    original = ujson.load(f1)
    updated = ujson.load(f2)
    print("Original:", len(original))
    print("Atualizado:", len(updated))'''

from nltk.tokenize import word_tokenize
import nltk

text = "NLTK is a great package for working with natural language data."

print("Bigramas e Trigramas")
n = 50
print(list(nltk.bigrams(word_tokenize(text)[:n])))
[('but', 'he'), ('he', 'swiftly'), ('swiftly', 'calls'), ('calls', 'away'), ('away', 'the'), ('the', 'captain'), ('captain',
'from'), ...]
print(list(nltk.trigrams(word_tokenize(text)[:n])))
[('but', 'he', 'swiftly'), ('he', 'swiftly', 'calls'), ('swiftly', 'calls', 'away'), ('calls', 'away', 'the'), ('away', 'the',
'captain'), ('the', 'captain', 'from'), ...]
