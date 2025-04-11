import streamlit as st #cria a interface web
from PIL import Image
import ujson
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
nltk.download('stopwords')
nltk.download('punkt')


# Set up the NLTK components
stemmer = PorterStemmer()
stop_words = stopwords.words('english')
tfidf = TfidfVectorizer()

# Load the data
with open('publication_list_stemmed.json', 'r') as f:
    pub_list_first_stem = ujson.load(f)
with open('publication_indexed_dictionary.json', 'r') as f: #{nome publicação:[0,1,5,..] publicações em que aparece}
    pub_index = ujson.load(f)
with open('author_list_stemmed.json', 'r') as f:
    author_list_first_stem = ujson.load(f)
with open('author_indexed_dictionary.json', 'r') as f:
    author_index = ujson.load(f)
with open('publication_abstract_list_stemmed_abstract.json', 'r') as f:
    pub_abstract_list_first_stem = ujson.load(f)
with open('publication_indexed_dictionary_abstract.json', 'r') as f:
    pub_abstract_index = ujson.load(f)


#with open("pdf_list_stemmed.json", "r") as f:
#    pdf_list_first_stem = ujson.load(f)
#with open("pdfs_indexed_dictionary.json", "r") as f:
#    pub_index = ujson.load(f)
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
###### ACRESCENTAR DOS LINKS E DOS RESEARCH GROUPS

with open('scraper_results_groups_links.json', 'r', encoding='utf-8') as f:
    pub_group_links = ujson.load(f)



def search_data(input_text, operator_val, search_type): #função de procura
    output_data = {}
    if operator_val == 2: #pesquisa or
        input_text = input_text.lower().split() #separa a frase por espaços
        pointer = []
        for token in input_text:
            if len(input_text) < 2:
                st.warning("Please enter at least 2 words to apply the operator.")
                break
            # if len(token) <= 3:
            #     st.warning("Please enter more than 4 characters.")
            #     break
            stem_temp = ""
            stem_word_file = []
            temp_file = []
            word_list = word_tokenize(token) #ex. machine-learning!, o tokenize faz ['machine','-','learning','!'] quando o split dava tudo junto

            for x in word_list: #divide a pesquisa em palavras
                if x not in stop_words: #remove stop words
                    stem_temp += stemmer.stem(x) + " " #aplica stemming
            stem_word_file.append(stem_temp)
            print(stem_word_file)
            #print(stem_temp)

            if search_type == "publication" and pub_index.get(stem_word_file[0].strip()): #Se for "publication", pesquisa no pub_index
                pointer = pub_index.get(stem_word_file[0].strip())
                print(pointer)
            elif search_type == "author" and author_index.get(stem_word_file[0].strip()): #Se for "author", pesquisa no author_index
                pointer = author_index.get(stem_word_file[0].strip())
            elif search_type == "abstract" and pub_abstract_index.get(stem_word_file[0].strip()): #Se for "author", pesquisa no author_index
                pointer = pub_abstract_index.get(stem_word_file[0].strip())

            #print(pointer)

            if len(pointer) == 0: #se nao encontrou nada no indice, sem resultados
                output_data = {}
            else:
                for j in pointer: #indice de cada documento que contem a palavra
                    if search_type == "publication":
                        temp_file.append(pub_list_first_stem[j]) #testo do ficheiro publication_list_stemmed.json que é o texto do documento sem stop words e com stem
                    elif search_type == "author":
                        temp_file.append(author_list_first_stem[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first_stem[j])

                temp_file = tfidf.fit_transform(temp_file) #Transforma os textos em vetores TF-IDF
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file)) #Calcula a similaridade do cosseno entre a pesquisa e os textos encontrados

                #print(pointer)
                for j in pointer:
                    output_data[j] = cosine_output[pointer.index(j)]
                print(output_data)

    else:  # operador and
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
            for x in word_list:
                if x not in stop_words:
                    stem_temp += stemmer.stem(x) + " "
            stem_word_file.append(stem_temp)

            if search_type == "publication" and pub_index.get(stem_word_file[0].strip()):
                set1 = set(pub_index.get(stem_word_file[0].strip())) #set 1 é o conjunto dos indices dos documentos onde a palavra processada aparece
                pointer.extend(list(set1)) #Adiciona o conjunto set1 ao pointer
            elif search_type == "author" and author_index.get(stem_word_file[0].strip()):
                set1 = set(author_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))
            elif search_type == "abstract" and pub_abstract_index.get(stem_word_file[0].strip()):
                set1 = set(pub_abstract_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1)) #adiciona os indices dos documentos onde aparece o token em questão

            if match_word == []: #se match_word estiver vazia - 1ºtoken, ela será preenchida com documentos que já aparecem em pointer
                match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
            else: #match_word vai conter no final os documentos onde aparecem TODOS os termos de pesquisa (interseção de set1 ao longo do loop)
                match_word.extend(list(set1))
                match_word = list({z for z in match_word if z in set2 or (set2.add(z) or False)}) # atualização da lista para garantir que apenas os documentos que correspondem a todos os tokens sejam mantidos.

        if len(input_text) > 1:
            match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}

            if len(match_word) == 0: #se nenhum documento satis faz a query
                output_data = {}
            else: #se houver match, vamos calcular tf-idf e similaridade com o cos
                for j in list(match_word):
                    if search_type == "publication":
                        temp_file.append(pub_list_first_stem[j]) #texto completo dos documentos sem stop words e com stem
                    elif search_type == "author":
                        temp_file.append(author_list_first_stem[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first_stem[j])

                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                for j in list(match_word):
                    output_data[j] = cosine_output[list(match_word).index(j)]
        else:   #se a query tiver só uma palavra
            if len(pointer) == 0:
                output_data = {}
            else:
                for j in pointer:
                    if search_type == "publication":
                        temp_file.append(pub_list_first_stem[j])
                    elif search_type == "author":
                        temp_file.append(author_list_first_stem[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first_stem[j])

                temp_file = tfidf.fit_transform(temp_file)
                cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))

                for j in pointer:
                    output_data[j] = cosine_output[pointer.index(j)]

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

    st.title("LIB Mathematics Support Centre Publications Search")

    # Campo e opções da primeira seção (LIB)
    search_input = st.text_input("Enter your search terms:")
    operator = st.radio("Search operator:", ["AND", "OR"], index=0)
    search_button = st.button("Search")

    if search_button and search_input:
        operator_val = 1 if operator == "AND" else 2
        results = search_LIB_data(search_input, operator_val)
        show_LIB_results(results)
    else:
        input_text = st.text_input("Search research:", key="query_input")
        operator_val = st.radio(
            "Search Filters",
            ['Exact', 'Relevant'],
            index=1,
            key="operator_input",
            horizontal=True,
        )
        search_type = st.radio(
            "Search in:",
            ['Publications', 'Authors', 'Abstracts'],
            index=0,
            key="search_type_input",
            horizontal=True,
        )

        if st.button("SEARCH"):
            if search_type == "Publications":
                output_data = search_data(input_text, 1 if operator_val == 'Exact' else 2, "publication")
            elif search_type == "Authors":
                output_data = search_data(input_text, 1 if operator_val == 'Exact' else 2, "author")
            elif search_type == "Abstracts":
                output_data = search_data(input_text, 1 if operator_val == 'Exact' else 2, "abstract")
            else:
                output_data = {}

            # Display the search results
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
    #print(f"rank is {rank_sorting}")
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

