import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import ujson

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

# Preprocess the author names
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
authors_list_first_stem = []
authors_list = []

for author in authors:
    words = word_tokenize(author) #O nome do autor é tokenizado (separado em palavras).
    stem_word = ""
    for word in words:
        if word.lower() not in stop_words: # Remove stopwords (palavras sem significado relevante).
            stem_word += stemmer.stem(word) + " " # Aplica stemming (exemplo: "running" → "run").
    authors_list_first_stem.append(stem_word)
    authors_list.append(author)

# Indexing process
#Cria um índice invertido, um dicionário onde cada palavra processada aponta para os autores que a contêm.
data_dict = {}
for i in range(len(authors_list_first_stem)):
    for word in authors_list_first_stem[i].split():
        if word not in data_dict:
            data_dict[word] = [i]
        else:
            data_dict[word].append(i)

#ex:
#authors_list_first_stem = ["john doe", "jane smith", "doe smith"]
#indice: {
#    "john": [0],
#    "doe": [0, 2],
#    "jane": [1],
#    "smith": [1, 2]
#} Isto permite encontrar todos os autores que contem um determinado nome

# Write the preprocessed author names and indexed dictionary to JSON files
with open('author_list_stemmed.json', 'w') as f:
    ujson.dump(authors_list_first_stem, f)

with open('author_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict, f)

#resumo:
#Extrai os autores de scraper_results.json
#Salva os nomes em author_names.json
#Limpa e processa os nomes (remove stopwords, aplica stemming)
#Cria um índice invertido para pesquisas eficientes
#Salva os resultados em JSONs para uso posterior.