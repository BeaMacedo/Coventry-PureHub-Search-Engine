# Adicionar no início do código, após outros imports
import nltk #NLTK for natural language processing tasks
import ujson
import wordnet
from nltk.corpus import stopwords # list of stop word
from nltk.tokenize import word_tokenize # To tokenize each word
from nltk.stem import PorterStemmer # For specific rules to transform words to their stems
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.util import ngrams
from collections import defaultdict

# Load the data
with open('publication_list_stemmed.json', 'r') as f:
    pub_list_first_stem = ujson.load(f)
with open('publication_list_lemma.json', 'r') as f:
    pub_list_first_lemma = ujson.load(f)

# Adicionar após carregar os dados existentes
# Carregar ou criar índices de n-gramas
def load_or_create_ngram_indices():
    try:
        with open('bigram_index.json', 'r') as f:
            bigram_index = ujson.load(f)
        with open('trigram_index.json', 'r') as f:
            trigram_index = ujson.load(f)
        with open('bigram_index_lemma.json', 'r') as f:
            bigram_index_lemma = ujson.load(f)
        with open('trigram_index_lemma.json', 'r') as f:
            trigram_index_lemma = ujson.load(f)
    except FileNotFoundError:
        # Criar índices se não existirem
        bigram_index = defaultdict(list)
        trigram_index = defaultdict(list)
        bigram_index_lemma = defaultdict(list)
        trigram_index_lemma = defaultdict(list)

        # Processar para stemming
        for doc_id, text in enumerate(pub_list_first_stem):
            tokens = text.split()
            # Bigramas
            for bg in list(ngrams(tokens, 2)):
                bigram_index[" ".join(bg)].append(doc_id)
            # Trigramas
            for tg in list(ngrams(tokens, 3)):
                trigram_index[" ".join(tg)].append(doc_id)

        # Processar para lematização
        for doc_id, text in enumerate(pub_list_first_lemma):
            tokens = text.split()
            # Bigramas
            for bg in list(ngrams(tokens, 2)):
                bigram_index_lemma[" ".join(bg)].append(doc_id)
            # Trigramas
            for tg in list(ngrams(tokens, 3)):
                trigram_index_lemma[" ".join(tg)].append(doc_id)

        # Salvar os índices
        with open('pub_bigram_index.json', 'w') as f:
            ujson.dump(bigram_index, f)
        with open('pub_trigram_index.json', 'w') as f:
            ujson.dump(trigram_index, f)
        with open('pub_bigram_index_lemma.json', 'w') as f:
            ujson.dump(bigram_index_lemma, f)
        with open('pub_trigram_index_lemma.json', 'w') as f:
            ujson.dump(trigram_index_lemma, f)

    return bigram_index, trigram_index, bigram_index_lemma, trigram_index_lemma

load_or_create_ngram_indices()