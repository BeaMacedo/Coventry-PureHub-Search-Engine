
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
#pubGroups = []  #FAZER?

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
    pubAbstract.append(data_dict_abs[i].get("abstract", "")) #se "abstract" não existir, vai pôr uma string vazia ""
    i+=1

with open('pub_name.json', 'w') as f:ujson.dump(pubName, f)
with open('pub_url.json', 'w') as f:ujson.dump(pubURL, f)
with open('pub_cu_author.json', 'w') as f:ujson.dump(pubCUAuthor, f)
with open('pub_date.json', 'w') as f: ujson.dump(pubDate, f)
with open('pub_abstract.json', 'w') as f: ujson.dump(pubAbstract, f)
#cada um destes ficheiros json armazenam todos os nomes das publicações/url/autor/data





#-------------------------------------------------------Indice invertido dos nomes das publicações-------------------------------------------------------------------
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

pub_abstract_list = [] #abstracts sem modificações
pub_abstract_list_first_stem = [] #abstracts sem stop words e com steming
pub_abstract_list_wo_sc = [] #sem caracteres especiais
pub_abstract_list_stem_wo_sw = [] #sem caracteres especiais, com steming, sem stop words
print(len(pubAbstract))

#o código tokeniza o nome de cada publicação, remove as stopwords e aplica o stemming nas palavras, criando uma versão "limpa" da publicação:
for abstract in pubAbstract:

    if not abstract.strip():  # Ignora abstracts vazios ou com só espaços
        pub_abstract_list_first_stem.append("")
        pub_abstract_list.append("")
        continue

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


#Stemming Process
for abstract in pub_abstract_list_wo_sc:

    if not abstract.strip():
        pub_abstract_list_stem_wo_sw.append("")
        continue

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

    if not pub_abstract_list_stem_wo_sw[a].strip():
        continue  # ignora se o abstract for vazio

    for b in pub_abstract_list_stem_wo_sw[a].split():
        if b not in data_dict:
            if b != 'background':
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



#--------------------Índice invertido pdfs-------------------------------
import os
import re
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ujson

# Baixar stopwords e punkt do NLTK, caso não tenha feito antes
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Definir o stemmer e as stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def limpar_texto(texto):
    # Remover todos os caracteres não alfabéticos e deixar apenas letras e espaços
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)

    # Converter o texto para minúsculas
    texto = texto.lower()

    # Tokenizar o texto em palavras
    palavras = word_tokenize(texto)

    # Remover stopwords e aplicar stemming
    palavras_limpa = [stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words]

    # Retornar as palavras limpas como uma string
    return ' '.join(palavras_limpa)


def extrair_texto_pdf(caminho_pdf):
    try:
        with open(caminho_pdf, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            texto = ""
            for pagina in range(len(reader.pages)):
                texto += reader.pages[pagina].extract_text()
            return texto
    except Exception as e:
        print(f"Erro ao processar o PDF {caminho_pdf}: {e}")
        return ""


def processar_pdfs():
    # Diretório onde os PDFs estão localizados
    pasta_pdfs = "pdfs_LIBMathematicsSupportCentre"

    # Lista para armazenar textos extraídos e os índices dos PDFs
    textos_limpos = []
    index_documentos = []

    # Percorrer todos os PDFs na pasta
    for nome_arquivo in os.listdir(pasta_pdfs):
        caminho_pdf = os.path.join(pasta_pdfs, nome_arquivo)

        if nome_arquivo.endswith(".pdf"):
            print(f"Processando PDF: {nome_arquivo}")

            # Extrair o texto do PDF
            texto_extraido = extrair_texto_pdf(caminho_pdf)

            # Limpar o texto extraído
            texto_limpo = limpar_texto(texto_extraido)

            if texto_limpo:
                textos_limpos.append(texto_limpo)
                index_documentos.append(nome_arquivo)  # Guardar o nome do arquivo como identificador

    # Criar o índice invertido
    data_dict = {}
    for i, texto in enumerate(textos_limpos):
        for palavra in texto.split():
            if palavra not in data_dict:
                data_dict[palavra] = [i]
            else:
                data_dict[palavra].append(i)

    # Salvar o índice invertido e os textos limpos
    with open('pdf_indexado.json', 'w', encoding='utf-8') as f:
        ujson.dump(data_dict, f, indent=2)

    with open('pdf_textos_limpos.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_limpos, f, indent=2)

    print(f"\n[✅] Processamento completo. {len(textos_limpos)} PDFs processados e indexados.")


# Chamar a função para processar os PDFs
processar_pdfs()
