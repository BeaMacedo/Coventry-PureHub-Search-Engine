
import nltk #NLTK for natural language processing tasks
import ujson
from nltk.corpus import stopwords # list of stop word 
from nltk.tokenize import word_tokenize # To tokenize each word
from nltk.stem import PorterStemmer # For specific rules to transform words to their stems


# Preprosessing data before indexing
with open('scraper_results_with_abstracts.json', 'r') as doc: scraper_results=doc.read()

# Initialize empty lists to store publication name, URL, author, and date
pubAbstract = []

# Load the scraped results using ujson
data_dict = ujson.loads(scraper_results) #cada item deste dicionario contem as informações sobre uma publicação

# Get the length of the data_dict (number of publications)
array_length = len(data_dict)
# Print the number of publications
print(array_length)

for item in data_dict:
    if "abstract" in item and item["abstract"]:  # Verifica se há um abstract presente
        pubAbstract.append(item["abstract"])

with open('pub_abstract.json', 'w') as f: ujson.dump(pubAbstract, f) #vai guardar os abstracts das publicações neste ficheiro

with open('pub_abstract.json','r') as f:publication=f.read()

#Load JSON File
pubAbstract = ujson.loads(publication)

#Downloading libraries to use its methods
nltk.download('stopwords')
nltk.download('punkt')

#Predefined stopwords in nltk are used
stop_words = stopwords.words('english')
#print(stop_words)
stemmer = PorterStemmer()

pub_abstract_list = [] #abstracts sem modificações
pub_abstract_list_first_stem = [] #abstracts sem stop words e com steming
pub_abstract_list_wo_sc = [] #sem caracteres especiais
pub_abstract_list_stem_wo_sw = [] #sem caracteres especiais, com steming, sem stop words
print(len(pubAbstract))

#o código tokeniza o nome de cada publicação, remove as stopwords e aplica o stemming nas palavras, criando uma versão "limpa" da publicação:
for abstract in pubAbstract:
    #Splitting strings to tokens(words)
    words = word_tokenize(abstract)
    stem_word = ""
    for i in words:
        if i.lower() not in stop_words:
            stem_word += stemmer.stem(i) + " "
    pub_abstract_list_first_stem.append(stem_word)
    pub_abstract_list.append(abstract)

#Removing all below characters (dos nomes das publicações)
special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~0123456789+=’‘'''
for abstract in pub_abstract_list:
    word_wo_sc = ""
    if len(abstract.split()) ==1 : pub_abstract_list_wo_sc.append(abstract)
    else:
        for a in abstract:
            if a in special_characters:
                word_wo_sc += ' '
            else:
                word_wo_sc += a
        #print(word_wo_sc)
        pub_abstract_list_wo_sc.append(word_wo_sc)

#Stemming Process
for abstract in pub_abstract_list_wo_sc:
    words = word_tokenize(abstract)
    stem_word = ""
    for a in words:
        if a.lower() not in stop_words:
            stem_word += stemmer.stem(a) + ' '
    pub_abstract_list_stem_wo_sw.append(stem_word.lower())

data_dict = {} #Inverted Index holder

# Indexing process
# indexação invertida, onde cada palavra é mapeada para os índices das publicações que a contêm.
#vai ficar cada palavra do nome das publicações e o numero dos documentos em que aparece.
for a in range(len(pub_abstract_list_stem_wo_sw)):
    for b in pub_abstract_list_stem_wo_sw[a].split():
        if b not in data_dict:
             data_dict[b] = [a]
        else:
            data_dict[b].append(a)


# printing the lenght
print(len(pub_abstract_list))
print(len(pub_abstract_list_first_stem))
print(len(pub_abstract_list_wo_sc))
print(len(pub_abstract_list_stem_wo_sw))

# with open('publication_list.json', 'w') as f:
#     ujson.dump(pub_list, f)

with open('publication_abstract_list_stemmed_abstract.json', 'w') as f: #sem stop words e com stem
    ujson.dump(pub_abstract_list_first_stem, f)

with open('publication_indexed_dictionary_abstract.json', 'w') as f: #indice invertido, onde se removeu caracteres especiais, stop words e com stem
    ujson.dump(data_dict, f)

