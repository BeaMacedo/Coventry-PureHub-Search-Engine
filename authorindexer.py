import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ujson
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

# Load the JSON file containing scraped results
with open('scraper_results.json', 'r') as doc:
    scraper_results = doc.read() #contém dados coletados de artigos científicos

# Extract author names from the JSON data
authors = []
data_dict = ujson.loads(scraper_results)
for item in data_dict:
    authors.append(item["cu_author"]) #Extrai os nomes dos autores de scraper_results.json (chave "cu_author")

# Write the author names to a JSON file
with open('author_names.json', 'w') as f:
    ujson.dump(authors, f) #Salva a lista de autores extraídos no arquivo author_names.json

# Download necessary NLTK resources
nltk.download('stopwords') #Para remover stopwords (palavras comuns como "the", "and")
nltk.download('punkt') #Para dividir os nomes em palavras (tokenização)

# Load the JSON file containing author names
with open('author_names.json', 'r') as f: #Lê os autores do arquivo author_names.json para processar os nomes
    author_data = f.read()

# Load JSON data
authors = ujson.loads(author_data) #Lê o conteúdo da variável author_data, que contém uma string no formato JSON.


# Função para converter POS tags para o formato WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Padrão para substantivo

#Função para a lematização:
def enhanced_lemmatize(text):
    lemmatizer = WordNetLemmatizer()

    # Tokenizar e obter POS tags
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    lemmas = []
    for token, tag in pos_tags:
        # Obter a tag no formato WordNet
        wn_tag = get_wordnet_pos(tag)
        # Lematizar com a tag apropriada
        lemma = lemmatizer.lemmatize(token, wn_tag)
        lemmas.append(lemma)

    return ' '.join(lemmas)

# Preprocess the author names
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
authors_list_first_stem = []
authors_list_first_lemma = []
authors_list = []

for author in authors:
    words = word_tokenize(author) #O nome do autor é tokenizado (separado em palavras).
    stem_word = ""
    lemma_word = ""
    for word in words:
        if word.lower() not in stop_words: # Remove stopwords (palavras sem significado relevante).
            stem_word += stemmer.stem(word) + " " # Aplica stemming (exemplo: "running" → "run").
    lemma_word = enhanced_lemmatize(' '.join([w.lower() for w in words if w.lower() not in stop_words]))
    authors_list_first_stem.append(stem_word)
    authors_list_first_lemma.append(lemma_word)
    authors_list.append(author)

# Indexing process
#Cria um índice invertido, um dicionário onde cada palavra processada aponta para os autores que a contêm.
data_dict_stem = {}
for i in range(len(authors_list_first_stem)):
    for word in authors_list_first_stem[i].split():
        if word not in data_dict_stem:
            data_dict_stem[word] = [i]
        else:
            data_dict_stem[word].append(i)

data_dict_lemma = {}
for i in range(len(authors_list_first_lemma)):
    for word in authors_list_first_lemma[i].split():
        if word not in data_dict_lemma:
            data_dict_lemma[word] = [i]
        else:
            data_dict_lemma[word].append(i)


# Write the preprocessed author names and indexed dictionary to JSON files
with open('author_list_stemmed.json', 'w') as f:
    ujson.dump(authors_list_first_stem, f)

with open('author_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict_stem, f)

with open('author_list_lemma.json', 'w') as f:
    ujson.dump(authors_list_first_lemma, f)

with open('author_indexed_dictionary_lemma.json', 'w') as f:
    ujson.dump(data_dict_lemma, f)