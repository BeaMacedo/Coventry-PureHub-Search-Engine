
import nltk #NLTK for natural language processing tasks
import ujson
from nltk.corpus import stopwords # list of stop word 
from nltk.tokenize import word_tokenize # To tokenize each word
from nltk.stem import PorterStemmer # For specific rules to transform words to their stems


with open('scraper_results.json', 'r') as doc: scraper_results=doc.read()
with open('scraper_results_with_abstracts.json', 'r') as doc: scraper_results_abs=doc.read()


# Initialize empty lists to store publication name, URL, author, and date
    
pubName = []
pubURL = []
pubCUAuthor = []
pubDate = []
pubAbstract = []

data_dict = ujson.loads(scraper_results) ## Converte JSON (string) para um dicionário, onde cada item deste dicionario contem as informações sobre uma publicação
data_dict_abs = ujson.loads(scraper_results_abs) #cada item deste dicionario contem as informações sobre uma publicação


# Get the length of the data_dict (number of publications)
array_length = len(data_dict)
array_abs_length = len(data_dict_abs)
print(f"length of data dict: {array_length} and data dict abs: {array_abs_length}")


#Seperate name, url, date, author in different file
i = 0
for item in data_dict: #nao faz mais sentido só ir BUSCAR TUDO ao scraper_results_abs?????????????
    pubName.append(item["name"])
    pubURL.append(item["pub_url"])
    pubCUAuthor.append(item["cu_author"])
    pubDate.append(item["date"])
    if 'abstract' in data_dict_abs[i].keys():   # Se a publicação no segundo ficheiro (scraper_results_with_abstracts.json) tiver um "abstract", adiciona-o à lista pubResumo
        pubAbstract.append(data_dict_abs[i]["abstract"])
    i+=1

with open('pub_name.json', 'w') as f:ujson.dump(pubName, f)
with open('pub_url.json', 'w') as f:ujson.dump(pubURL, f)
with open('pub_cu_author.json', 'w') as f:ujson.dump(pubCUAuthor, f)
with open('pub_date.json', 'w') as f: ujson.dump(pubDate, f)
with open('pub_abstract.json', 'w') as f: ujson.dump(pubAbstract, f)
#cada um destes ficheiros json armazenam todos os nomes das publicações/url/autor/data

#Open a file with publication names in read mode
with open('pub_name.json','r') as f:publication=f.read()

#Load JSON File
pubName = ujson.loads(publication)

#Downloading libraries to use its methods
nltk.download('stopwords')
nltk.download('punkt')

#Predefined stopwords in nltk are used
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
pub_list_first_stem = [] #nomes das publicações após tokenização, remoção de stopwords e stemming.
pub_list = [] # Mantém os nomes das publicações originais
pub_list_wo_sc = [] #Contém nomes das publicações sem caracteres especiais
print(len(pubName))

#o código tokeniza o nome de cada publicação, remove as stopwords e aplica o stemming nas palavras
for file in pubName:
    #Splitting strings to tokens(words)
    words = word_tokenize(file) # Divide o nome da publicação em palavras
    stem_word = ""
    for i in words:
        if i.lower() not in stop_words: # Remove stopwords
            stem_word += stemmer.stem(i) + " " # # Aplica stemming
    pub_list_first_stem.append(stem_word)
    pub_list.append(file)

#Removing all below characters (dos nomes das publicações)
special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~0123456789+=’‘'''
for file in pub_list: #vai a cada nome original das publicações
    word_wo_sc = "" #versão modificada do nome da publicação, sem caracteres especiais.
    if len(file.split()) ==1 : pub_list_wo_sc.append(file) #se so tiver uma palavra essa palavra é adicionada à lista sem modificação
    else:
        for a in file: #percorre cada caracter
            if a in special_characters:
                word_wo_sc += ' ' #substitui caracteres especiais por espaço
            else:
                word_wo_sc += a
        #print(word_wo_sc)
        pub_list_wo_sc.append(word_wo_sc)

#Stemming Process
#aplica o stemming e remove stopwords novamente, após a remoção dos caracteres especiais
pub_list_stem_wo_sw = [] #nomes das publicações sem caracteres especiais e com stemming e sem stop words
for name in pub_list_wo_sc: #vai à lista com os nomes das publicações sem caracteres especiais
    words = word_tokenize(name)
    stem_word = ""
    for a in words:
        if a.lower() not in stop_words:
            stem_word += stemmer.stem(a) + ' '
    pub_list_stem_wo_sw.append(stem_word.lower())

data_dict = {} #Inverted Index holder

# Indexing process
# indexação invertida, onde cada palavra é mapeada para os índices das publicações que a contêm.
#vai ficar cada palavra do nome das publicações e o numero dos documentos em que aparece.
for a in range(len(pub_list_stem_wo_sw)):
    for b in pub_list_stem_wo_sw[a].split(): #percorre cada palavra do nome de uma publicação
        if b not in data_dict:
             data_dict[b] = [a] # Se a palavra não existe, cria uma nova entrada
        else:
            data_dict[b].append(a)

# printing the lenght
print(len(pub_list_wo_sc))
print(len(pub_list_stem_wo_sw))
print(len(pub_list_first_stem))
print(len(pub_list))

# with open('publication_list.json', 'w') as f:
#     ujson.dump(pub_list, f)

with open('publication_list_stemmed.json', 'w') as f:
    ujson.dump(pub_list_first_stem, f)

with open('publication_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict, f)

