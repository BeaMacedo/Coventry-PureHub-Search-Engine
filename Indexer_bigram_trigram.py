import ujson
from nltk.util import ngrams
from collections import defaultdict

#Sem aplicar lematização e stem para ser uma pesquisa exata com aspas

# Carregar nomes das publicações
with open('pubname_list.json', 'r') as f:
    publication = f.read()

pubName = ujson.loads(publication)

# Criar índices de n-gramas (sem stemming/lemmatização)
def create_ngram_indices_from_pubname():
    bigram_index = defaultdict(list)
    trigram_index = defaultdict(list)

    for doc_id, text in enumerate(pubName):
        tokens = text.lower().split()  # transformar em minúsculas
        for bg in ngrams(tokens, 2):
            bg_key = " ".join(bg)
            if doc_id not in bigram_index[bg_key]:
                bigram_index[bg_key].append(doc_id)
        for tg in ngrams(tokens, 3):
            tg_key = " ".join(tg)
            if doc_id not in trigram_index[tg_key]:
                trigram_index[tg_key].append(doc_id)

    # Guardar os índices
    with open('pubname_bigram_index.json', 'w') as f:
        ujson.dump(bigram_index, f)

    with open('pubname_trigram_index.json', 'w') as f:
        ujson.dump(trigram_index, f)

    return bigram_index, trigram_index

#create_ngram_indices_from_pubname()

# Carregar abstracts das publicações
with open('abstract_list.json', 'r') as f:
    abstract = f.read()

abstracts = ujson.loads(abstract)

def create_ngram_indices_from_abstract():
    bigram_index = defaultdict(list)
    trigram_index = defaultdict(list)

    for doc_id, text in enumerate(abstracts):
        tokens = text.lower().split()  # transformar em minúsculas
        for bg in ngrams(tokens, 2):
            bg_key = " ".join(bg)
            if doc_id not in bigram_index[bg_key]:
                bigram_index[bg_key].append(doc_id)
        for tg in ngrams(tokens, 3):
            tg_key = " ".join(tg)
            if doc_id not in trigram_index[tg_key]:
                trigram_index[tg_key].append(doc_id)

    # Guardar os índices
    with open('abstract_bigram_index.json', 'w') as f:
        ujson.dump(bigram_index, f)

    with open('abstract_trigram_index.json', 'w') as f:
        ujson.dump(trigram_index, f)

    return bigram_index, trigram_index

# Criar e guardar os índices
#create_ngram_indices_from_abstract()
