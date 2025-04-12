
import nltk #NLTK for natural language processing tasks
import ujson
from nltk.corpus import stopwords # list of stop word 
from nltk.tokenize import word_tokenize # To tokenize each word
from nltk.stem import PorterStemmer # For specific rules to transform words to their stems
from nltk.stem import WordNetLemmatizer

with open('scraper_results.json', 'r') as doc: scraper_results=doc.read()
with open('scraper_results_with_abstracts.json', 'r') as doc: scraper_results_abs=doc.read()
with open('scraper_results_groups_links.json', 'r') as doc: scraper_results_groups_links=doc.read()

# Initialize empty lists to store publication name, URL, author, and date
    
pubName = []
pubURL = []
pubCUAuthor = []
pubDate = []
pubAbstract = []
pubGroup = []
pubLinkPDF = []


data_dict = ujson.loads(scraper_results) ## Converte JSON (string) para um dicionário, onde cada item deste dicionario contem as informações sobre uma publicação
data_dict_abs = ujson.loads(scraper_results_abs) #cada item deste dicionario contem as informações sobre uma publicação
data_dict_gr_lin = ujson.loads(scraper_results_groups_links) #cada item deste dicionario contem as informações sobre uma publicação


# Get the length of the data_dict (number of publications)
array_length = len(data_dict)
array_abs_length = len(data_dict_abs)
array_gr_lin_length = len(data_dict_gr_lin)
print(f"length of data dict: {array_length} and data dict abs: {array_abs_length} and data dict gr lin: {array_gr_lin_length}")


#Seperate name, url, date, author in different file
i = 0
for item in data_dict:
    pubName.append(item["name"])
    pubURL.append(item["pub_url"])
    pubCUAuthor.append(item["cu_author"])
    pubDate.append(item["date"])
    pubAbstract.append(data_dict_abs[i].get("abstract", "")) #se "abstract" não existir, vai pôr uma string vazia ""
    pubGroup.append(data_dict_gr_lin[i].get("research_group", ""))
    pubLinkPDF.append(data_dict_gr_lin[i].get("link", ""))
    i+=1

with open('pub_name.json', 'w') as f:ujson.dump(pubName, f)
with open('pub_url.json', 'w') as f:ujson.dump(pubURL, f)
with open('pub_cu_author.json', 'w') as f:ujson.dump(pubCUAuthor, f)
with open('pub_date.json', 'w') as f: ujson.dump(pubDate, f)
with open('pub_abstract.json', 'w') as f: ujson.dump(pubAbstract, f)
with open('pub_groups.json', 'w') as f: ujson.dump(pubGroup, f)
with open('pub_linksPDF.json', 'w') as f: ujson.dump(pubLinkPDF, f)
#cada um destes ficheiros json armazenam todos os nomes das publicações/url/autor/data



#-------------------------------------------------------Indice invertido dos nomes das publicações-------------------------------------------------------------------
#Open a file with publication names in read mode
with open('pub_name.json','r') as f:publication=f.read()

#Load JSON File
pubName = ujson.loads(publication)

#Downloading libraries to use its methods
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#Predefined stopwords in nltk are used
stop_words = stopwords.words('english')
lemma = WordNetLemmatizer()

'''# Função para converter POS tags para o formato WordNet
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
'''
stemmer = PorterStemmer()
pub_list_first_stem = [] #nomes das publicações após tokenização, remoção de stopwords e stemming.
pub_list_first_lemma = []
pub_list = [] # Mantém os nomes das publicações originais
pub_list_wo_sc = [] #Contém nomes das publicações sem caracteres especiais
print(len(pubName))

#o código tokeniza o nome de cada publicação, remove as stopwords e aplica o stemming nas palavras
for file in pubName:
    #Splitting strings to tokens(words)
    words = word_tokenize(file) # Divide o nome da publicação em palavras
    stem_word = ""
    lemma_word = ""
    for i in words:
        if i.lower() not in stop_words: # Remove stopwords
            stem_word += stemmer.stem(i) + " " # # Aplica stemming
            lemma_word += lemma.lemmatize(i) + " "
    pub_list_first_stem.append(stem_word)
    pub_list_first_lemma.append(lemma_word)
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
pub_list_lemma_wo_sw = []

for name in pub_list_wo_sc: #vai à lista com os nomes das publicações sem caracteres especiais
    words = word_tokenize(name)
    stem_word = ""
    lemma_word = ""
    for a in words:
        if a.lower() not in stop_words:
            stem_word += stemmer.stem(a) + ' '
            lemma_word += lemma.lemmatize(a) + ' '
    pub_list_stem_wo_sw.append(stem_word.lower())
    pub_list_lemma_wo_sw.append(lemma_word.lower())

data_dict_stemm = {} #Inverted Index holder

# Indexing process
# indexação invertida, onde cada palavra é mapeada para os índices das publicações que a contêm.
#vai ficar cada palavra do nome das publicações e o numero dos documentos em que aparece.
for a in range(len(pub_list_stem_wo_sw)):
    for b in pub_list_stem_wo_sw[a].split(): #percorre cada palavra do nome de uma publicação
        if b not in data_dict_stemm:
             data_dict_stemm[b] = [a] # Se a palavra não existe, cria uma nova entrada
        else:
            data_dict_stemm[b].append(a)

data_dict_lemma = {}

for a in range(len(pub_list_lemma_wo_sw)):
    for b in pub_list_lemma_wo_sw[a].split():
        if b not in data_dict_lemma:
             data_dict_lemma[b] = [a]
        else:
            data_dict_lemma[b].append(a)

# printing the lenght
print(len(pub_list_wo_sc))
print(len(pub_list_stem_wo_sw))
print(len(pub_list_first_stem))
print(len(pub_list_first_lemma))
print(len(pub_list))

# with open('publication_list.json', 'w') as f:
#     ujson.dump(pub_list, f)

with open('publication_list_stemmed.json', 'w') as f:
    ujson.dump(pub_list_first_stem, f)

with open('publication_indexed_dictionary.json', 'w') as f:
    ujson.dump(data_dict_stemm, f)

with open('publication_list_lemma.json', 'w') as f:
    ujson.dump(pub_list_first_lemma, f)

with open('publication_indexed_dictionary_lemma.json', 'w') as f:
    ujson.dump(data_dict_lemma, f)


#-------------------------------------------------------Indice invertido dos abstract das publicações-------------------------------------------------------------------

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
lemma = WordNetLemmatizer()

pub_abstract_list = [] #abstracts sem modificações
pub_abstract_list_first_stem = [] #abstracts sem stop words e com steming
pub_abstract_list_first_lemma = []
pub_abstract_list_wo_sc = [] #sem caracteres especiais
print(len(pubAbstract))

#o código tokeniza o nome de cada publicação, remove as stopwords e aplica o stemming nas palavras, criando uma versão "limpa" da publicação:
for abstract in pubAbstract:

    if not abstract.strip():  # Ignora abstracts vazios ou com só espaços
        pub_abstract_list_first_stem.append("")
        pub_abstract_list_first_lemma.append("")
        pub_abstract_list.append("")
        continue

    #Splitting strings to tokens(words)
    words = word_tokenize(abstract)
    stem_word = ""
    lemma_word = ""
    for i in words:
        if i.lower() not in stop_words:
            stem_word += stemmer.stem(i) + " "
            lemma_word += lemma.lemmatize(i) + " "
    pub_abstract_list_first_stem.append(stem_word)
    pub_abstract_list_first_lemma.append(lemma_word)
    pub_abstract_list.append(abstract)

#Removing all below characters (dos nomes das publicações)
special_characters = '''!()-—[]{};:'"\, <>./?@#$%^&*_~0123456789+=’‘'''
for abstract in pub_abstract_list:

    if not abstract.strip():
        pub_abstract_list_wo_sc.append("")
        continue

    word_wo_sc = ""
    #if len(abstract.split()) ==1 : pub_abstract_list_wo_sc.append(abstract)
    #else:
    for a in abstract:
        if a in special_characters:
            word_wo_sc += ' '
        else:
            word_wo_sc += a
    #print(word_wo_sc)
    pub_abstract_list_wo_sc.append(word_wo_sc)

pub_abstract_list_stem_wo_sw = [] #sem caracteres especiais, com steming, sem stop words
pub_abstract_list_lemma_wo_sw = []
#Stemming Process
for abstract in pub_abstract_list_wo_sc:

    if not abstract.strip():
        pub_abstract_list_stem_wo_sw.append("")
        pub_abstract_list_lemma_wo_sw.append("")
        continue

    words = word_tokenize(abstract)
    stem_word = ""
    lemma_word = ""
    for a in words:
        if a.lower() not in stop_words:
            stem_word += stemmer.stem(a) + ' '
            lemma_word += lemma.lemmatize(a) + ' '
    pub_abstract_list_stem_wo_sw.append(stem_word.lower())
    pub_abstract_list_lemma_wo_sw.append(lemma_word.lower())

data_dict_stemm = {} #Inverted Index holder

# Indexing process
# indexação invertida, onde cada palavra é mapeada para os índices das publicações que a contêm.
#vai ficar cada palavra do nome das publicações e o numero dos documentos em que aparece.
for a in range(len(pub_abstract_list_stem_wo_sw)):

    if not pub_abstract_list_stem_wo_sw[a].strip():
        continue  # ignora se o abstract for vazio

    for b in pub_abstract_list_stem_wo_sw[a].split():
        if b not in data_dict_stemm:
            if b != 'background':
                data_dict_stemm[b] = [a]
        else:
            data_dict_stemm[b].append(a)

data_dict_lemma = {} #Inverted Index holder

# Indexing process
# indexação invertida, onde cada palavra é mapeada para os índices das publicações que a contêm.
#vai ficar cada palavra do nome das publicações e o numero dos documentos em que aparece.
for a in range(len(pub_abstract_list_lemma_wo_sw)):

    if not pub_abstract_list_lemma_wo_sw[a].strip():
        continue  # ignora se o abstract for vazio

    for b in pub_abstract_list_lemma_wo_sw[a].split():
        if b not in data_dict_lemma:
            if b != 'background':
                data_dict_lemma[b] = [a]
        else:
            data_dict_lemma[b].append(a)

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
    ujson.dump(data_dict_stemm, f)

with open('publication_abstract_list_lemma_abstract.json', 'w') as f: #sem stop words e com stem
    ujson.dump(pub_abstract_list_first_lemma, f)

with open('publication_indexed_dictionary_abstract_lemma.json', 'w') as f: #indice invertido, onde se removeu caracteres especiais, stop words e com stem
    ujson.dump(data_dict_lemma, f)
