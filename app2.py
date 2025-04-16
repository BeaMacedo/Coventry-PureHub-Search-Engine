import nltk
import streamlit as st #cria a interface web
from PIL import Image
import ujson
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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


#Número de vezes que a palavra aparece dividido pelo número de palavras no documento

def tf(word, document):
    return document.count(word) / len(document)

def idf(word, corpus):
    count_of_documents = len(corpus) + 1
    count_of_documents_with_word = sum([1 for doc in corpus if word in doc]) + 1
    IDF = np.log10(count_of_documents/count_of_documents_with_word) + 1
    return IDF

def TF_IDF(word, document, corpus):
    print(
        f"{word} | TF: {tf(word, document):.4f}, IDF: {idf(word, corpus):.4f}, TF-IDF: {tf(word, document) }")

    return tf(word, document) #* idf(word, corpus) para ser linear

def tf_idf_vectorizer(corpus):
    if isinstance(corpus[0], str):
        corpus = [corpus]

    word_set = list(set(sum(corpus, []))) #lista de palavras unicas que aparecem em todos os documentos do corpus, ou seja, sem repetições
    word_to_index = {word: i for i, word in enumerate(word_set)}  #mapeia cada palavra do vocabulário para um índice numérico, ou seja a primeira da palavra de word_set será uma chave que corresponderá ao seu indice na lista que é 0 {"1ºpalavra":0,..}
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

    # Calcular similaridades (evitando divisão por zero)
    similarities = []
    for dot, doc_norm in zip(dot_products, doc_norms):
        if query_norm == 0 or doc_norm == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot / (query_norm * doc_norm)) #produto escalar dos vetores a dividir pelo produto das normas

    return similarities

def cos_sim(mat_A, mat_B):
    matriz_sim = []
    magnitude_B = [sum(b[i]*b[i] for b in mat_B)**0.5 for i in range(len(mat_B[0]))]
    for linha in mat_A:
        dot_products = [sum(a*b for a, b in zip(linha, col)) for col in zip(*mat_B)]
        magnitude_A = sum(a*a for a in linha)**0.5
        cosine_similarities = [dot_product / (magnitude_A * magnitude_B[i]) for i, dot_product in enumerate(dot_products)]
        matriz_sim.append(cosine_similarities)
    return matriz_sim

def query_to_vector(query, word_to_index, corpus):

    num_words = len(word_to_index)
    query_vector = [0.0] * num_words

    for word in query:
        if word in word_to_index:
            # Calcula TF-IDF apenas para palavras presentes no vocabulário
            tf_idf_score = TF_IDF(word, query, corpus)
            query_vector[word_to_index[word]] = tf_idf_score

    return query_vector


def search_with_operators(input_text, search_type, stem_lema, rank_by="Sklearn function"):
    # Configuração dos índices
    if stem_lema == 2:  # lematização
        pub_index = pub_index_lemma
        pub_list_first = pub_list_first_lemma
        abstract_index = pub_abstract_index_lemma
        pub_abstract_list_first = pub_abstract_list_first_lemma
        author_index = author_index_lemma
        author_list_first = author_list_first_lemma
    else:  # stemming
        pub_index = pub_index_stem
        pub_list_first = pub_list_first_stem
        abstract_index = pub_abstract_index_stem
        pub_abstract_list_first = pub_abstract_list_first_stem
        author_index = author_index_stem
        author_list_first = author_list_first_stem

    # Parse da query
    and_groups, not_terms = parse_query(input_text)

    # 1. Processar NOTs primeiro
    docs_to_exclude = set()
    for term in not_terms:
        stemmed_term = process_term(term, stem_lema) #aplica stemming ou lematização ao termo
        #vai ver todos os documentos onde esse termo aparece e guarda-os em docs_to_exclude
        if search_type == "publication" and stemmed_term in pub_index:
            docs_to_exclude.update(set(pub_index[stemmed_term]))
        elif search_type == "author" and stemmed_term in author_index:
            docs_to_exclude.update(set(author_index[stemmed_term]))
        elif search_type == "abstract" and stemmed_term in abstract_index:
            docs_to_exclude.update(set(abstract_index[stemmed_term]))

    # 2. Processar OR groups
    all_matching_docs = set() #começa como conjunto vazio

    for group in and_groups: #cada grupo é uma lista de termos unidos por and
        # Ordenar termos por frequência (do mais raro para o mais comum)
        terms_with_freq = []
        for term in group:
            stemmed_term = process_term(term, stem_lema) #aplica stemming ou lematização ao termo
            freq = 0
            if search_type == "publication" and stemmed_term in pub_index:
                freq = len(pub_index[stemmed_term])
            elif search_type == "author" and stemmed_term in author_index:
                freq = len(author_index[stemmed_term])
            elif search_type == "abstract" and stemmed_term in abstract_index:
                freq = len(abstract_index[stemmed_term])
            terms_with_freq.append((stemmed_term, freq))

        # Ordenar termos pela frequência (ascendente)
        terms_sorted = sorted(terms_with_freq, key=lambda x: x[1])

        # Processar ANDs na ordem otimizada
        group_docs = None
        for term, _ in terms_sorted:
            term_docs = set()
            if search_type == "publication" and term in pub_index:
                term_docs = set(pub_index[term])
            elif search_type == "author" and term in author_index:
                term_docs = set(author_index[term])
            elif search_type == "abstract" and term in abstract_index:
                term_docs = set(abstract_index[term])

            if group_docs is None:
                group_docs = term_docs
            else:
                group_docs.intersection_update(term_docs) #Vai reduzindo group_docs aos documentos que contém todos os termos do grupo (AND)
                if not group_docs:  # Early exit se conjunto vazio
                    break

        if group_docs:
            all_matching_docs.update(group_docs) #adiciona todos os elementos de group_docs ao conjunto

    # 3. Aplicar NOTs
    final_docs = all_matching_docs - docs_to_exclude

    # 4. Calcular similaridade do cosseno
    output_data = {}
    if final_docs:
        # Preparar query para TF-IDF
        query_terms = []
        for group in and_groups:
            for term in group:
                stemmed_term = process_term(term, stem_lema)
                query_terms.append(stemmed_term)
        query_text = ' '.join(query_terms)
        print(f"query_text: {query_text}")

        # Preparar textos dos documentos
        docs_texts = []
        doc_ids = []
        for doc_id in final_docs:
            if search_type == "publication":
                docs_texts.append(pub_list_first[doc_id])
            elif search_type == "author":
                docs_texts.append(author_list_first[doc_id])
            elif search_type == "abstract":
                docs_texts.append(pub_abstract_list_first[doc_id])
            doc_ids.append(doc_id)

        if rank_by == "Sklearn function":
            # Calcular TF-IDF e similaridade do cosseno
            tfidf_matrix = tfidf.fit_transform(docs_texts)
            query_vector = tfidf.transform([query_text]) #transforma um texto (neste caso uma query) num vetor TF-IDF
            cosine_scores = cosine_similarity(tfidf_matrix, query_vector)

            # Atribuir scores aos documentos
            for idx, doc_id in enumerate(doc_ids): # idx corresponde à posição daquele documento na lista cosine_scores
                output_data[doc_id] = cosine_scores[idx][0]
        else:
            tokenized_docs = [doc.split() for doc in docs_texts]
            word_set = list(set(sum(tokenized_docs,[])))  # lista de palavras unicas que aparecem em todos os documentos do corpus, ou seja, sem repetições
            word_to_index = {word: i for i, word in enumerate(
                word_set)}  # mapeia cada palavra do vocabulário para um índice numérico, ou seja a primeira da palavra de word_set será uma chave que corresponderá ao seu indice na lista que é 0 {"1ºpalavra":0,..}
            doc_vectors = tf_idf_vectorizer(docs_texts)  # Calcula os vetores TF-IDF manualmente

            # Calcular a similaridade de cosseno entre o vetor da pesquisa e os vetores dos documentos
            query_tokens = query_text.split()
            query_vec = query_to_vector(query_tokens, word_to_index, docs_texts)
            cosine_output = cosine_similarity(query_vec, doc_vectors)

            # Armazena a similaridade de cosseno no dicionário de resultados
            for idx, doc_id in enumerate(doc_ids):
                output_data[doc_id] = cosine_output[idx]

    return output_data


def process_term(term, stem_lema):
    # Processa um termo (stemming ou lematização)
    word_list = word_tokenize(term)
    stem_temp = ""

    if stem_lema == 1:  # stemming
        for x in word_list:
            if x not in stop_words:
                stem_temp += stemmer.stem(x) + " "
    else:  # lematização
        stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))

    return stem_temp.strip()


def parse_query(query):
    # Processa a query e divide em partes AND, OR e NOT
    query = query.lower().strip()

    # Processar NOTs
    not_terms = []
    not_pattern = re.compile(r'not\s+([^\s]+)') #encontra termos/palavras precedidos pela palavra not
    for match in not_pattern.finditer(query): #percorre todas as ocorrencias da regex na query
        not_terms.append(match.group(1)) #adiciona essas ocorrencias à lista not_terms

    # Remover NOTs para processar o resto
    query_without_nots = not_pattern.sub('', query)

    # Processar ORs (tem precedência sobre AND)
    or_groups = [group.strip() for group in re.split(r'\bor\b', query_without_nots) if group.strip()]

    # Processar ANDs em cada grupo OR
    and_groups = []
    for group in or_groups:
        # Dividir por AND explícito ou espaços
        terms = []
        for part in re.split(r'\band\b|\s+', group):
            if part.strip():
                terms.append(part.strip())
        if terms:
            and_groups.append(terms)

    return and_groups, not_terms

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# Set up the NLTK components
stemmer = PorterStemmer()
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer()
lemmatizer = WordNetLemmatizer()

# Load the data
with open('publication_list_stemmed.json', 'r') as f:
    pub_list_first_stem = ujson.load(f)
with open('publication_list_lemma.json', 'r') as f:
    pub_list_first_lemma = ujson.load(f)
with open('publication_indexed_dictionary.json', 'r') as f: #{nome publicação:[0,1,5,..] publicações em que aparece}
    pub_index_stem = ujson.load(f)
with open('publication_indexed_dictionary_lemma.json', 'r') as f: #{nome publicação:[0,1,5,..] publicações em que aparece}
    pub_index_lemma = ujson.load(f)
with open('author_list_stemmed.json', 'r') as f:
    author_list_first_stem = ujson.load(f)
with open('author_list_lemma.json', 'r') as f:
    author_list_first_lemma = ujson.load(f)
with open('author_indexed_dictionary.json', 'r') as f:
    author_index_stem = ujson.load(f)
with open('author_indexed_dictionary_lemma.json', 'r') as f:
    author_index_lemma = ujson.load(f)
with open('publication_abstract_list_stemmed_abstract.json', 'r') as f:
    pub_abstract_list_first_stem = ujson.load(f)
with open('publication_abstract_list_lemma_abstract.json', 'r') as f:
    pub_abstract_list_first_lemma = ujson.load(f)
with open('publication_indexed_dictionary_abstract.json', 'r') as f:
    pub_abstract_index_stem = ujson.load(f)
with open('publication_indexed_dictionary_abstract_lemma.json', 'r') as f:
    pub_abstract_index_lemma = ujson.load(f)
# Carregar os índices específicos para o grupo LIB
with open('pdfs_indexed_dictionary.json', 'r', encoding='utf-8') as f:
    lib_index = ujson.load(f)
with open('pdf_list_stemmed.json', 'r', encoding='utf-8') as f:
    lib_texts = ujson.load(f)
with open('author_names.json', 'r') as f:
    author_name = ujson.load(f)
with open('pub_name.json', 'r') as f:
    pub_name = ujson.load(f)
with open('pub_url.json', 'r') as f:
    pub_url = ujson.load(f)
with open('pub_cu_author.json', 'r') as f:
    pub_cu_author = ujson.load(f)
with open('pub_date.json', 'r') as f:
    pub_date = ujson.load(f)
with open('pub_abstract.json', 'r') as f:
    pub_abstract = ujson.load(f)
with open('pub_groups.json', 'r') as f:
    pub_groups = ujson.load(f)
with open('pub_linksPDF.json', 'r') as f:
    pub_linksPDF = ujson.load(f)

with open('scraper_results_groups_links.json', 'r', encoding='utf-8') as f:
    pub_group_links = ujson.load(f)


def search_data(input_text, operator_val, search_type, stem_lema, rank_by="Sklearn function"): #função de procura

    if stem_lema == 2:  # se lematizacao
        pub_index = pub_index_lemma
        pub_list_first = pub_list_first_lemma
        abstract_index = pub_abstract_index_lemma
        pub_abstract_list_first = pub_abstract_list_first_lemma
        author_index = author_index_lemma
        author_list_first = author_list_first_lemma
    else:
        pub_index = pub_index_stem
        pub_list_first = pub_list_first_stem
        abstract_index = pub_abstract_index_stem
        pub_abstract_list_first = pub_abstract_list_first_stem
        author_index = author_index_stem
        author_list_first = author_list_first_stem

    output_data = {}
    if operator_val == 2: #operador or
        input_text = input_text.lower().split() #separa a frase por espaços
        pointer = []
        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break
            stem_temp = ""
            stem_word_file = []
            temp_file = []
            word_list = word_tokenize(token) #ex. machine-learning!, o tokenize faz ['machine','-','learning','!'] quando o split dava tudo junto

            if stem_lema == 1:
                for x in word_list:
                    if x not in stop_words:
                        stem_temp += stemmer.stem(x) + " "
            elif stem_lema == 2:
                stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))

            stem_word_file.append(stem_temp.strip())
            #print(f"stem_word {stem_word_file}")

            if search_type == "publication" and pub_index.get(stem_word_file[0].strip()): #Se for "publication", pesquisa no pub_index
                pointer = pub_index.get(stem_word_file[0].strip())
                #print(pointer)
            elif search_type == "author" and author_index.get(stem_word_file[0].strip()): #Se for "author", pesquisa no author_index
                pointer = author_index.get(stem_word_file[0].strip())
            elif search_type == "abstract" and abstract_index.get(stem_word_file[0].strip()): #Se for "author", pesquisa no author_index
                pointer = abstract_index.get(stem_word_file[0].strip())

            print(f"pointeragora:{pointer}")

            if len(pointer) == 0: #se nao encontrou nada no indice, sem resultados
                output_data = {}
            else:
                for j in pointer: #indice de cada documento que contem a palavra
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j]) #texto do ficheiro publication_list_stemmed.json que é o texto do documento sem stop words e com stem
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])

                if rank_by == "Sklearn function":
                    temp_file = tfidf.fit_transform(temp_file) #Transforma os textos em vetores TF-IDF
                    #print(f"stem_word_file_or: {stem_word_file}")
                    cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file)) #Calcula a similaridade do cosseno entre a pesquisa e os textos encontrados

                    #print(f"pointer_or:{pointer}")
                    for j in pointer:
                        output_data[j] = cosine_output[pointer.index(j)] #primeira posição em que um indice de um documento aparece, para nao haver duplicados
                    #print(f"output_data_or:{output_data}")

                else:
                    i =0
                    matrix = []
                    for elem in temp_file:
                        elem = tf_idf_vectorizer(elem.split())[0]
                        matrix.append(elem)
                        i+=1

                    mat_inv = tf_idf_vectorizer(stem_word_file)
                    cos_out = cos_sim(matrix, mat_inv) #similaridade entre cada vetor do documento e a query
                    for j in pointer:
                        output_data[j] = cos_out[pointer.index(j)]

    elif operator_val == 1:  # operador and
        #se quiser colocar a query toda poderia criar uma lista antes do ciclo for como  stem_word_file2 = [] e colocar todas as palavras da query e usá-la depois com tfidf.transform
        input_text = input_text.lower().split()
        pointer = [] #documentos encontrados
        match_word = [] # Apenas os documentos que contêm TODAS as palavras (interseção)
        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break
            # if len(token) <= 3:
            #     st.warning("Please enter more than 4 characters.")
            #     break
            temp_file = [] #armazena documentos temporários
            set2 = set() #conjunto de documentos encontrados usado para garantir que apenas documentos comuns a todos os tokens sejam mantidos.
            stem_word_file = [] #armazenar palavras após stemming
            word_list = word_tokenize(token)
            stem_temp = ""
            if stem_lema == 1:
                for x in word_list:
                    if x not in stop_words:
                        stem_temp += stemmer.stem(x) + " "
            else:
                stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))
            stem_word_file.append(stem_temp.strip())
            print(f"stem_word {stem_word_file}")

            if search_type == "publication" and pub_index.get(stem_word_file[0].strip()):
                set1 = set(pub_index.get(stem_word_file[0].strip())) #set 1 é o conjunto dos indices dos documentos onde a palavra processada aparece
                pointer.extend(list(set1)) #Adiciona o conjunto set1 ao pointer
                print(f"pointer: {pointer}")
            elif search_type == "author" and author_index.get(stem_word_file[0].strip()):
                set1 = set(author_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))
            elif search_type == "abstract" and abstract_index.get(stem_word_file[0].strip()):
                set1 = set(abstract_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1)) #adiciona os indices dos documentos onde aparece o token em questão

            if match_word == []: #se match_word estiver vazia - 1ºtoken, ela será preenchida com documentos que já aparecem em pointer
                match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
            else: #match_word vai conter no final os documentos onde aparecem TODOS os termos de pesquisa (interseção de set1 ao longo do loop)
                match_word.extend(list(set1))
                match_word = list({z for z in match_word if z in set2 or (set2.add(z) or False)}) # atualização da lista para garantir que apenas os documentos que correspondem a todos os tokens sejam mantidos.
        print(f"match_word_first: {match_word}")
        if len(input_text) > 1:
            match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}
            print(f"match_word_first: {match_word}")
            if len(match_word) == 0: #se nenhum documento satisfaz faz a query
                output_data = {}


            else: #se houver match, vamos calcular tf-idf e similaridade com o cos
                for j in list(match_word):
                    print(f"j: {j}")
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j]) #texto completo dos documentos sem stop words e com stem
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])
                        #print(f"temp_file: {temp_file}")
                temp_file = tfidf.fit_transform(temp_file)
                print(f"stem_word_file: {stem_word_file}")
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))
                print("cosine_output1: ", cosine_output)
                print(f"list(match_word):{list(match_word)}")
                for j in list(match_word):
                    output_data[j] = cosine_output[list(match_word).index(j)]
                    print(f"output_data: {output_data}")

        else:
            if len(pointer) == 0:
                output_data = {}
            else:
                for j in pointer:
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j])
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])

                if rank_by == "Sklearn function":
                    temp_file = tfidf.fit_transform(temp_file)
                    cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                    for j in pointer:
                        output_data[j] = cosine_output[pointer.index(j)]
                else:
                    i = 0
                    matrix = []
                    for elem in temp_file:
                        elem = tf_idf_vectorizer(elem.split())[0]
                        matrix.append(elem)
                        i += 1
                    mat_inv = tf_idf_vectorizer(stem_word_file)
                    cos_out = cos_sim(matrix, mat_inv)
                    for j in list(match_word):
                        output_data[j] = cos_out[list(match_word).index(j)]


    elif operator_val == 3:  # PESQUISA COM OPERADORES LOGICOS

        output_data = search_with_operators(input_text, search_type, stem_lema)

    return output_data


def search_data2(input_text, operator_val, search_type, stem_lema, rank_by="Sklearn function"):
    if stem_lema == 2:  # se lematizacao
        pub_index = pub_index_lemma
        pub_list_first = pub_list_first_lemma
        abstract_index = pub_abstract_index_lemma
        pub_abstract_list_first = pub_abstract_list_first_lemma
        author_index = author_index_lemma
        author_list_first = author_list_first_lemma
    else:
        pub_index = pub_index_stem
        pub_list_first = pub_list_first_stem
        abstract_index = pub_abstract_index_stem
        pub_abstract_list_first = pub_abstract_list_first_stem
        author_index = author_index_stem
        author_list_first = author_list_first_stem

    output_data = {}

    # Processa toda a query primeiro
    all_stem_words = []
    processed_terms = []

    # Pré-processamento de todos os termos da query
    for token in input_text.lower().split():
        word_list = word_tokenize(token)
        stem_temp = ""

        if stem_lema == 1:  # stemming
            for x in word_list:
                if x not in stop_words:
                    stem_temp += stemmer.stem(x) + " "
        else:  # lematização
            stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))

        stem_word = stem_temp.strip()
        if stem_word:  # só adiciona se não for string vazia
            all_stem_words.append(stem_word)
            processed_terms.append(stem_word)

    if operator_val == 2:  # operador OR
        pointer = []
        for term in processed_terms:
            if search_type == "publication" and pub_index.get(term):
                pointer.extend(pub_index.get(term))
            elif search_type == "author" and author_index.get(term):
                pointer.extend(author_index.get(term))
            elif search_type == "abstract" and abstract_index.get(term):
                pointer.extend(abstract_index.get(term))

        pointer = list(set(pointer))  # remove duplicados

        if len(pointer) == 0:
            return {}

        # Coletar textos dos documentos encontrados
        temp_file = []
        for j in pointer:
            if search_type == "publication":
                temp_file.append(pub_list_first[j])
            elif search_type == "author":
                temp_file.append(author_list_first[j])
            elif search_type == "abstract":
                temp_file.append(pub_abstract_list_first[j])

        if rank_by == "Sklearn function":
            tfidf_matrix = tfidf.fit_transform(temp_file)
            print(f"all_stem_words: {all_stem_words}")
            full_query = ' '.join(all_stem_words)
            print(f"fully_query: {full_query}")
            query_vector = tfidf.transform([full_query])
            cosine_scores = cosine_similarity(tfidf_matrix, query_vector)
            print(f"cosine_scores: {cosine_scores}")

            for idx, doc_id in enumerate(pointer):
                output_data[doc_id] = cosine_scores[idx][0]
        else:
            tokenized_docs = [doc.split() for doc in temp_file]
            word_set = list(set(sum(tokenized_docs, [])))
            word_to_index = {word: i for i, word in enumerate(word_set)}

            full_query = ' '.join(all_stem_words).split()
            query_vec = query_to_vector(full_query, word_to_index, tokenized_docs)
            doc_vectors = tf_idf_vectorizer(tokenized_docs)

            cosine_output = cosine_similarity(query_vec, doc_vectors)
            for idx, doc_id in enumerate(pointer):
                output_data[doc_id] = cosine_output[idx]

    elif operator_val == 1:  # operador AND
        pointer = None
        for term in processed_terms:
            term_docs = set()
            if search_type == "publication" and pub_index.get(term):
                term_docs = set(pub_index.get(term))
            elif search_type == "author" and author_index.get(term):
                term_docs = set(author_index.get(term))
            elif search_type == "abstract" and abstract_index.get(term):
                term_docs = set(abstract_index.get(term))

            if pointer is None:
                pointer = term_docs
                print(f"pointer: {pointer}")
            else:
                pointer.intersection_update(term_docs)
                print(f"pointer after intersection: {pointer}")
                if not pointer:  # early exit se conjunto vazio
                    break

        if not pointer:  # nenhum documento contém todos os termos
            return {}

        pointer = list(pointer)
        print(f"pointer after list: {pointer}")

        # Coletar textos dos documentos encontrados
        temp_file = []
        for j in pointer:
            if search_type == "publication":
                temp_file.append(pub_list_first[j])
            elif search_type == "author":
                temp_file.append(author_list_first[j])
            elif search_type == "abstract":
                temp_file.append(pub_abstract_list_first[j])

        if rank_by == "Sklearn function":
            tfidf_matrix = tfidf.fit_transform(temp_file)
            full_query = ' '.join(all_stem_words)
            query_vector = tfidf.transform([full_query])
            cosine_scores = cosine_similarity(tfidf_matrix, query_vector)

            for idx, doc_id in enumerate(pointer):
                output_data[doc_id] = cosine_scores[idx][0]
        else:
            tokenized_docs = [doc.split() for doc in temp_file]
            word_set = list(set(sum(tokenized_docs, [])))
            word_to_index = {word: i for i, word in enumerate(word_set)}

            full_query = ' '.join(all_stem_words).split()
            query_vec = query_to_vector(full_query, word_to_index, tokenized_docs)
            doc_vectors = tf_idf_vectorizer(tokenized_docs)

            cosine_output = cosine_similarity(query_vec, doc_vectors)
            for idx, doc_id in enumerate(pointer):
                output_data[doc_id] = cosine_output[idx]

    elif operator_val == 3:  # operadores lógicos (NOT, AND, OR)
        output_data = search_with_operators(input_text, search_type, stem_lema, rank_by)

    return output_data

#-----------------para indices pdf do grupo de pesquisa LIB
# Função para carregar os índices mapeados de dic_indices_pdfs.json
def load_pdf_index_mapping():
    with open('dic_indices_pdfs.json', 'r', encoding='utf-8') as f:
        map_pdf_to_lib = ujson.load(f)
    # Inverter o mapeamento: de índices globais para índices dos PDFs
    map_lib_to_pdf = {v: int(k) for k, v in map_pdf_to_lib.items()}
    return map_lib_to_pdf

# Função de pesquisa
def search_LIB_data(input_text, operator_val):

    output_data = {}

    # Carregar o mapeamento de índices (apenas uma vez)
    map_lib_to_pdf = load_pdf_index_mapping()

    # Para o operador OR (operator_val == 2)
    if operator_val == 2:  # Operador OR
        input_text = input_text.lower().split()
        pointer = []

        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break

            stem_temp = ""
            stem_word_file = []
            temp_file = []
            word_list = word_tokenize(token)

            for x in word_list:
                if x not in stop_words:
                    stem_temp += stemmer.stem(x) + " "
            stem_word_file.append(stem_temp)  # Palavras pré-processadas do input

            # Verificar se existe o índice para o token processado
            if lib_index.get(stem_word_file[0].strip()):
                pointer = lib_index.get(stem_word_file[0].strip())

        if len(pointer) == 0:
            return {}
        else:
            # Usar o mapeamento de índice para PDFs
            for j in pointer:
                pdf_index = map_lib_to_pdf.get(j)
                if pdf_index is not None:
                    temp_file.append(lib_texts[pdf_index])

            temp_file = tfidf.fit_transform(temp_file)
            cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

            # Atribuir os resultados ao output_data
            for j in pointer:
                pdf_index = map_lib_to_pdf.get(j)
                if pdf_index is not None:
                    output_data[j] = cosine_output[pointer.index(j)]

    # Para o operador AND (caso contrário)
    else:  # Operador AND
        input_text = input_text.lower().split()
        pointer = []
        match_word = []

        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break

            temp_file = []
            set2 = set()
            stem_word_file = []
            word_list = word_tokenize(token)
            stem_temp = ""
            for x in word_list:
                if x not in stop_words:
                    stem_temp += stemmer.stem(x) + " "
            stem_word_file.append(stem_temp)

            # Verificar se existe o índice para o token processado
            if lib_index.get(stem_word_file[0].strip()):
                set1 = set(lib_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))

            # Atualizar a lista match_word para documentos que atendem a todos os critérios
            if match_word == []:
                match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
            else:
                match_word.extend(list(set1))
                match_word = list({z for z in match_word if z in set2 or (set2.add(z) or False)})

        if len(input_text) > 1:
            match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}

            if len(match_word) == 0:
                output_data = {}
            else:
                # Usar o mapeamento de índice para PDFs
                for j in list(match_word):
                    pdf_index = map_lib_to_pdf.get(j)
                    if pdf_index is not None:
                        temp_file.append(lib_texts[pdf_index])

                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                # Atribuir os resultados ao output_data
                for j in list(match_word):
                    pdf_index = map_lib_to_pdf.get(j)
                    if pdf_index is not None:
                        output_data[j] = cosine_output[list(match_word).index(j)]

        else:
            if len(pointer) == 0:
                output_data = {}
            else:
                # Usar o mapeamento de índice para PDFs
                for j in pointer:
                    pdf_index = map_lib_to_pdf.get(j)
                    if pdf_index is not None:
                        temp_file.append(lib_texts[pdf_index])

                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                # Atribuir os resultados ao output_data
                for j in pointer:
                    pdf_index = map_lib_to_pdf.get(j)
                    if pdf_index is not None:
                        output_data[j] = cosine_output[pointer.index(j)]

    return output_data

#inicial
def app():  # interface Streamlit
    # Load the image and display it
    image = Image.open('cire.png')
    st.image(image)

    input_text = st.text_input("Search research:", key="query_input")
    operator_val = st.radio(
        "Search Filters",
        ["AND", "OR", "Logical operators"],
        index=1,
        key="operator_input",
        horizontal=True,
    )
    search_type = st.radio(
        "Search in:",
        ['Publications', 'Authors', 'Abstracts', 'LIB Mathematics Support Centre Publications Search'],
        index=0,
        key="search_type_input",
        horizontal=True,
    )

    stem_lema = st.radio(
        "Search with:",
        ["Stemming", "Lemmatization"],
        index=0,
        key="stem_lema_input",
        horizontal=True,
    )
    rank_by = st.radio(
        "Rank with:",
        ["Sklearn function", "Use Custom tf-idf"],
        index=0,
        key="rank_system",
        horizontal=True,
    )

    if st.button("SEARCH"):
        if search_type == "Publications":
            output_data = search_data(input_text, 1 if operator_val == 'AND' else (
                            2
                            if operator_val == "OR"
                            else 3
                        ), "publication",  1 if stem_lema == "Stemming" else 2,
                                      rank_by)

            show_results(output_data, search_type)
        elif search_type == "Authors":
            output_data = search_data(input_text, 1 if operator_val == 'AND' else (
                            2
                            if operator_val == "OR"
                            else 3
                        ), "author", 1 if stem_lema == "Stemming" else 2,
                                      rank_by)

            show_results(output_data, search_type)
        elif search_type == "Abstracts":
            output_data = search_data(input_text, 1 if operator_val == 'AND' else (
                            2
                            if operator_val == "OR"
                            else 3
                        ), "abstract", 1 if stem_lema == "Stemming" else 2,
                                      rank_by)

            show_results(output_data, search_type)
        elif search_type == "LIB Mathematics Support Centre Publications Search":
            output_data = search_LIB_data(input_text, 1 if operator_val == 'Exact' else 2)
            show_LIB_results(output_data)
        else:
            output_data = {}
            show_results(output_data, search_type)

    st.markdown("<p style='text-align: center;'> Brought to you with ❤ by <a href='https://github.com/maladeep'>Mala Deep</a> | Data © Coventry University </p>", unsafe_allow_html=True)



def show_LIB_results(output_data):
    # Carregar os dados completos
    aa = 0 #contador de resultados
    rank_sorting = sorted(output_data.items(), key=lambda z: z[1], reverse=True)

    # Show the total number of research results
    st.info(f"Showing results for: {len(rank_sorting)}")

    # Show the cards
    N_cards_per_row = 3
    for n_row, (id_val, ranking) in enumerate(rank_sorting): #id_val: índice do documento e ranking: score de similaridade
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # Draw the card
        with cols[n_row % N_cards_per_row]:
            st.caption(f"{pub_date[id_val].strip()}")
            st.markdown(f"**{pub_cu_author[id_val].strip()}**")
            st.markdown(f"*{pub_name[id_val].strip()}*")
            st.markdown(f"**{pub_url[id_val]}**")
            # Se tiver link para PDF
            pub = pub_group_links[id_val]
            if pub.get('link'):
                st.markdown(f"**[Download PDF]({pub.get('link', '')})**")

        aa += 1

    if aa == 0:
        st.info("No results found. Please try again.")
    else:
        st.info(f"Results shown for: {aa}")


def show_results(output_data, search_type):
    aa = 0
    rank_sorting = sorted(output_data.items(), key=lambda z: z[1], reverse=True) #Ordena os resultados pela pontuação de similaridade
    print(f"rank is {rank_sorting}")
    # Show the total number of research results
    st.info(f"Showing results for: {len(rank_sorting)}") #mostra resultados pela ordem decrescente da pontuação

    # Show the cards
    N_cards_per_row = 3
    for n_row, (id_val, ranking) in enumerate(rank_sorting):
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # Draw the card
        with cols[n_row % N_cards_per_row]:
            if search_type == "Publications":
                st.caption(f"{pub_date[id_val].strip()}")
                st.markdown(f"**{pub_cu_author[id_val].strip()}**")
                st.markdown(f"*{pub_name[id_val].strip()}*")
                st.markdown(f"**{pub_url[id_val]}**")
            elif search_type == "Authors":
                st.caption(f"{pub_date[id_val].strip()}")
                st.markdown(f"**{author_name[id_val].strip()}**")
                st.markdown(f"*{pub_name[id_val].strip()}*")
                st.markdown(f"**{pub_url[id_val]}**")
                st.markdown(f"Ranking: {ranking[0]:.2f}")
            elif search_type == "Abstracts":
                st.caption(f"{pub_date[id_val].strip()}")
                st.markdown(f"**{author_name[id_val].strip()}**")
                st.markdown(f"*{pub_name[id_val].strip()}*")
                st.markdown(f"**{pub_url[id_val]}**")
                st.markdown(f"**{pub_abstract[id_val]}**")

        aa += 1

    if aa == 0:
        st.info("No results found. Please try again.")
    else:
        st.info(f"Results shown for: {aa}")


if __name__ == '__main__':
    app()

