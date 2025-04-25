
def operacoes_logicas(input_text, operator_val, search_type, stem_lema):
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
    if operator_val == 2:  # OPERADOR OR
        input_text = input_text.lower().split()
        pointer = []
        stem_word_file = []
        temp_file = []

        for token in input_text:
            stem_temp = ""
            stem_word_file1 = []  # query
            temp_file1 = []
            pointer1 = []
            word_list = word_tokenize(token)

            if stem_lema == 1:
                for x in word_list:
                    if x not in stop_words:
                        stem_temp += stemmer.stem(x) + " "
            elif stem_lema == 2:
                stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))

            stem_word_file.append(stem_temp.strip())
            print(f"stem_word {stem_word_file}")

            if search_type == "publication" and pub_index.get(stem_word_file1[0].strip()):
                pointer1 = pub_index.get(stem_word_file1[0].strip())

            elif search_type == "author" and author_index.get(
                    stem_word_file1[0].strip()
            ):
                pointer1 = author_index.get(stem_word_file1[0].strip())

            elif search_type == "abstract" and abstract_index.get(
                    stem_word_file1[0].strip()
            ):
                pointer1 = abstract_index.get(stem_word_file1[0].strip())


            if len(pointer1) == 0:
                output_data = {}
            else:
                for j in pointer1:
                    if search_type == "publication":
                        temp_file1.append(pub_list_first[j])
                    elif search_type == "author":
                        temp_file1.append(author_list_first[j])
                    elif search_type == "abstract":
                        temp_file1.append(pub_abstract_list_first[j])

            temp_file += temp_file1
            pointer += pointer1

        return temp_file, stem_word_file1, pointer

    elif operator_val == 1:  # AND
        input_text = input_text.lower().split()
        pointer = []
        match_word = []
        for token in input_text:
            temp_file = []
            set2 = set()
            stem_word_file = []
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
                set1 = set(pub_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))
            elif search_type == "author" and author_index.get(
                    stem_word_file[0].strip()
            ):
                set1 = set(author_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))
            elif search_type == "abstract" and abstract_index.get(
                    stem_word_file[0].strip()
            ):
                set1 = set(abstract_index.get(stem_word_file[0].strip()))
                pointer.extend(list(set1))

            if match_word == []:
                match_word = list(
                    {z for z in pointer if z in set2 or (set2.add(z) or False)}
                )
            else:
                match_word.extend(list(set1))
                match_word = list(
                    {z for z in match_word if z in set2 or (set2.add(z) or False)}
                )

        if len(input_text) > 1:
            match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}
            if len(match_word) == 0:
                output_data = {}

            else:
                for j in list(match_word):
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j])
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])

            return temp_file, stem_word_file, list(match_word)

        else:
            if len(pointer) == 0:
                output_data = {}
            else:
                for j in pointer:
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j])
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    if search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])

            return temp_file, stem_word_file, pointer

    elif operator_val == 3:  # OPERADOR NOT
        input_text = input_text.lower().split()
        pointer = []
        match_word = []
        complete_set = []
        to_remove = []
        temp_file = []
        stem_word_file = []
        if search_type == "publication":
            complete_set = list(
                set(value for values in pub_index.values() for value in values)
            )  # lista de todos os indices dos docs

        elif search_type == "author":
            complete_set = list(
                set(value for values in author_index.values() for value in values)
            )  # lista de todos os indices dos docs

        elif search_type == "abstract":
            complete_set = list(
                set(value for values in abstract_index.values() for value in values)
            )  # lista de todos os indices dos docs


        for token in input_text:

            set2 = set()

            word_list = word_tokenize(token)
            stem_temp = ""
            if stem_lema == 1:
                for x in word_list:
                    if x not in stop_words:
                        stem_temp += stemmer.stem(x) + " "
            else:
                stem_temp = enhanced_lemmatize(' '.join([w.lower() for w in word_list if w.lower() not in stop_words]))
            stem_word_file.append(stem_temp.strip())

            set1 = set()  # definir a variável
            if search_type == "publication" and pub_index.get(stem_word_file[0].strip()):
                set1 = set(
                    pub_index.get(stem_word_file[0].strip())
                )  # indices dos docs que fazem match

            elif search_type == "author" and author_index.get(
                    stem_word_file[0].strip()
            ):
                set1 = set(author_index.get(stem_word_file[0].strip()))

            elif search_type == "abstract" and abstract_index.get(
                    stem_word_file[0].strip()
            ):
                set1 = set(abstract_index.get(stem_word_file[0].strip()))

            to_remove += list(set1)

        complete_set = list(set(complete_set) - set(to_remove))

        if len(input_text) > 1:
            for j in list(complete_set):
                if search_type == "publication":
                    temp_file.append(pub_list_first[j])
                elif search_type == "author":
                    temp_file.append(author_list_first[j])
                elif search_type == "abstract":
                    temp_file.append(pub_abstract_list_first[j])

        else:
            if len(complete_set) == 0:
                output_data = {}
            else:
                for j in list(complete_set):
                    if search_type == "publication":
                        temp_file.append(pub_list_first[j])
                    elif search_type == "author":
                        temp_file.append(author_list_first[j])
                    if search_type == "abstract":
                        temp_file.append(pub_abstract_list_first[j])

        return temp_file, stem_word_file, list(complete_set)

def not_operador(word):
    operadores = ["not", "and", "or"]
    if word not in operadores:
        return True
    else:
        return False

def parse_query(query):
    operadores = ["not", "and", "or"]
    query = query.lower().strip()
    operacoes_and1 = []
    query1 = query
    while re.search(r"([^\s]+) and ([^\s]+)", query1):
        query1 = re.sub(r"([^\s]+) and ([^\s]+)", r"\1 \2", query1)

    operacoes_and = []
    for tuple in operacoes_and1:
        operacoes_and.append(f"{tuple[0]} {tuple[1]}")

    operacoes_or = re.split(r" or ", query)
    operacoes_not1 = re.findall(r"not ([^\s]+)", query)
    operacoes_not = []
    if operacoes_not1:
        for operacao in operacoes_not1:
            if operacao:
                operacao = operacao.split(" ")
                operacoes_not.append(operacao[0])
    or_queries = []
    for i in range(len(operacoes_or) - 1):
        t1 = operacoes_or[i].split()
        termo1 = t1[len(t1) - 1]
        t2 = operacoes_or[i + 1].split()
        termo2 = t2[0]
        or_queries.append(termo1 + " " + termo2)

    query2 = query1.split()
    for i in range(len(query2) - 1):
        if not_operador(query2[i]) and not_operador(query2[i + 1]):
            operacoes_and.append(f"{query2[i]} {query2[i + 1]}")
    print("look here ::: " + str(operacoes_and) + "----" + str(operacoes_not) + "-----" + str(or_queries))

    return operacoes_and, operacoes_not, or_queries

def search_data1(input_text, operator_val, search_type, stem_lema, rank_by="Sklearn function"): #função de procura

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
            print(f"stem_word {stem_word_file}")

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
                    print(f"stem_word_file_or: {stem_word_file}")
                    cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file)) #Calcula a similaridade do cosseno entre a pesquisa e os textos encontrados

                    print(f"pointer_or:{pointer}")
                    for j in pointer:
                        output_data[j] = cosine_output[pointer.index(j)] #primeira posição em que um indice de um documento aparece, para nao haver duplicados
                    print(f"output_data_or:{output_data}")

                else:
                    i = 0
                    matrix = []
                    for elem in temp_file:
                        #print(f"elem no temp_file: {elem}")
                        elem = tf_idf_vectorizer(elem.split())[0]
                        #print(f"elem_vector: {elem}, len(elem): {len(elem)}")
                        matrix.append(elem)
                        #print("matrix dentro do for:",matrix)
                        i+=1
                    print(f"matrix:{matrix}")
                    print(f"Tamanho da matriz:{len(matrix)}")

                    mat_inv = tf_idf_vectorizer(stem_word_file)
                    print(f"mat_inv(query)={mat_inv}") #mat_inv(query)=[[1.0]] vai dar sempre so um valor pq é uma apalvra e vai ser sempre 1 pq aparece na query de pesquisa obviamente
                    cos_out = cos_sim(matrix, mat_inv) #similaridade entre cada vetor do documento e a query
                    for j in pointer:
                        output_data[j] = cos_out[pointer.index(j)]
                        print(f"output_data {j}: {output_data[j]} ")

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
                if rank_by == "Sklearn function":
                    temp_file = tfidf.fit_transform(temp_file)
                    print(f"stem_word_file: {stem_word_file}")
                    cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))
                    print("cosine_output_and: ", cosine_output)
                    #print(f"list(match_word):{list(match_word)}")
                    for j in list(match_word):
                        output_data[j] = cosine_output[list(match_word).index(j)]
                        print(f"output_data_fim: {output_data}")
                else:
                    i = 0
                    matrix = []
                    for elem in temp_file:
                        elem = tf_idf_vectorizer(elem.split())[0] #lista de vetores tf-idf para cada documento, com as palabvras unicas de cada documento, logo as listas nao vao ter o mesmo tamanho
                        matrix.append(elem)
                        i += 1
                    mat_inv = tf_idf_vectorizer(stem_word_file) #vai ser sempre 1 pq a palavra vai estar lá? E aqui vai sempre comparar com a ultima palavra.
                    cos_out = cos_sim(matrix, mat_inv)
                    for j in list(match_word):
                        output_data[j] = cos_out[list(match_word).index(j)]


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
                        elem = tf_idf_vectorizer(elem.split())[0] #lista de vetores tf-idf para cada documento, com as palabvras unicas de cada documento, logo as listas nao vao ter o mesmo tamanho
                        matrix.append(elem)
                        i += 1
                    mat_inv = tf_idf_vectorizer(stem_word_file)
                    cos_out = cos_sim(matrix, mat_inv)
                    for j in list(match_word):
                        output_data[j] = cos_out[list(match_word).index(j)]

    elif operator_val == 3:  # PESQUISA COM OPERADORES LOGICOS
        temp_file = []
        stem_word_file = []
        output_data = {}
        pointer = []
        if len(input_text.split()) > 1:
            operacoes_and, operacoes_not, operacoes_or = parse_query(input_text)
            for ope in operacoes_not:
                temp_file1, stem_word_file1, pointer1 = operacoes_logicas(
                    ope, 3, search_type, stem_lema
                )
                temp_file += temp_file1
                stem_word_file = stem_word_file1
                print(f"stem_word_file_not: {stem_word_file}")
                pointer += pointer1

            for ope in operacoes_or:
                temp_file1, stem_word_file1, pointer1 = operacoes_logicas(
                    ope, 2, search_type, stem_lema
                )
                temp_file += temp_file1
                stem_word_file = stem_word_file1
                pointer += pointer1

            # EXECUTAR OPERACOES AND
            for ope in operacoes_and:
                temp_file1, stem_word_file1, pointer1 = operacoes_logicas(
                    ope, 1, search_type, stem_lema
                )
                temp_file += temp_file1
                stem_word_file = stem_word_file1
                if len(pointer) != 0:
                    pointer = list((set(pointer).intersection(set(pointer1))))
                else:
                    pointer = pointer1

        else:
            temp_file, stem_word_file, pointer = operacoes_logicas(
                input_text, 2, search_type, stem_lema
            )
        if rank_by == "Sklearn function":
            print(f"temp_file: {temp_file}")
            temp_file = tfidf.fit_transform(temp_file)
            print(f"stem_word_file: {stem_word_file}")
            cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))
            for j in pointer:
                print(f"j:{j},output_data: {output_data}")
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
            for j in pointer:
                output_data[j] = cos_out[pointer.index(j)]

    return output_data




# Para os títulos:
with open('pubname_bigram_index.json', 'r', encoding='utf-8') as f:
    pubname_bigram_index = ujson.load(f)
with open('pubname_trigram_index.json', 'r', encoding='utf-8') as f:
    pubname_trigram_index = ujson.load(f)

# Para os abstracts:
with open('abstract_bigram_index.json', 'r', encoding='utf-8') as f:
    abstract_bigram_index = ujson.load(f)
with open('abstract_trigram_index.json', 'r', encoding='utf-8') as f:
    abstract_trigram_index = ujson.load(f)

#PESQUISA EM BIGRAMAS E TRIGRAMAS

def search_ngrams_only(input_text, operator_val, search_type="publication", rank_by="Sklearn function"):
    """
    Pesquisa em bigramas/trigramas seguindo EXATAMENTE o raciocínio de search_data
    """
    # Configuração dos índices
    if search_type == "publication":
        bigram_index = pubname_bigram_index
        trigram_index = pubname_trigram_index
        with open('pub_name.json', 'r') as f:
            pub_texts = ujson.load(f)
    elif search_type == "abstract":
        bigram_index = abstract_bigram_index
        trigram_index = abstract_trigram_index
        with open('pub_abstract.json', 'r') as f:
            pub_texts = ujson.load(f)
    else:
        return {}

    # Extrair frases entre aspas (n-gramas)
    phrases = [p.lower() for p in re.findall(r'"(.*?)"', input_text)]
    if not phrases:
        return {}


    output_data = {}
    if operator_val == 2:  # OPERADOR OR (igual à search_data)
        pointer = []

        # 1. Coletar documentos para cada n-grama (como na search_data)
        for phrase in phrases:
            temp_file = []
            stem_word_file = []  # Armazena cada n-grama individualmente
            words = phrase.split()
            if len(words) not in [2, 3]:  # Apenas bigramas/trigramas
                continue

            stem_word_file.append(phrase)  # Cada n-grama é tratado como um "termo"

            # Verificar índices
            if len(words) == 2 and phrase in bigram_index:
                pointer = bigram_index[phrase]
            elif len(words) == 3 and phrase in trigram_index:
                pointer = trigram_index[phrase]

            if len(pointer) == 0: #se nao encontrou nada no indice, sem resultados
                output_data = {}

            else:
                for j in pointer:
                    if search_type == "publication":
                        temp_file.append(pub_texts[j])
                    elif search_type == "abstract":
                        temp_file.append(pub_texts[j])

            if rank_by == "Sklearn function":
                temp_file = tfidf.fit_transform(stem_word_file)
                print(f"stem_word_file_or: {stem_word_file}")
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

                print(f"matrix:{matrix}")
                print(f"Tamanho da matriz:{len(matrix)}")

                mat_inv = tf_idf_vectorizer(stem_word_file)
                print(
                    f"mat_inv(query)={mat_inv}")  # mat_inv(query)=[[1.0]] vai dar sempre so um valor pq é uma apalvra e vai ser sempre 1 pq aparece na query de pesquisa obviamente
                cos_out = cos_sim(matrix, mat_inv)  # similaridade entre cada vetor do documento e a query
                for j in pointer:
                    output_data[j] = cos_out[pointer.index(j)]
                    print(f"output_data {j}: {output_data[j]} ")


    elif operator_val == 1:  # OPERADOR AND (igual à search_data)
        match_word = []
        pointer = []

        # 1. Encontrar interseção de documentos (AND)
        for phrase in phrases:
            temp_file = []
            set2 = set()
            stem_word_file = []
            words = phrase.split()
            if len(words) not in [2, 3]:
                continue

            stem_word_file.append(phrase)

            # Verificar índices
            if len(words) == 2 and phrase in bigram_index:
                set1 = set(bigram_index[phrase])
                pointer.extend(list(set1))  # Adiciona o conjunto set1 ao pointer
            elif len(words) == 3 and phrase in trigram_index:
                set1 = set(trigram_index[phrase])
                pointer.extend(list(set1))

            if match_word == []:  # se match_word estiver vazia - 1ºtoken, ela será preenchida com documentos que já aparecem em pointer
                match_word = list({z for z in pointer if z in set2 or (set2.add(z) or False)})
            else:  # match_word vai conter no final os documentos onde aparecem TODOS os termos de pesquisa (interseção de set1 ao longo do loop)
                match_word.extend(list(set1))
                match_word = list({z for z in match_word if z in set2 or (set2.add(
                    z) or False)})  # atualização da lista para garantir que apenas os documentos que correspondem a todos os tokens sejam mantidos.
            print(f"match_word_first: {match_word}")

            if len(phrases) > 1:
                match_word = {z for z in match_word if z in set2 or (set2.add(z) or False)}
                print(f"match_word_first: {match_word}")
                if len(match_word) == 0:  # se nenhum documento satisfaz faz a query
                    output_data = {}

                else:  # se houver match, vamos calcular tf-idf e similaridade com o cos
                    for j in list(match_word):
                        print(f"j: {j}")
                        if search_type == "publication":
                            temp_file.append(pub_texts[j])
                        elif search_type == "abstract":
                            temp_file.append(pub_texts[j])


                    if rank_by == "Sklearn function":
                        temp_file = tfidf.fit_transform(temp_file)
                        print(f"stem_word_file: {stem_word_file}")
                        cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))
                        print("cosine_output_and: ", cosine_output)
                        # print(f"list(match_word):{list(match_word)}")
                        for j in list(match_word):
                            output_data[j] = cosine_output[list(match_word).index(j)]
                            print(f"output_data_fim: {output_data}")
                    else:
                        i = 0
                        matrix = []
                        for elem in temp_file:
                            elem = tf_idf_vectorizer(elem.split())[0]  # lista de vetores tf-idf para cada documento, com as palabvras unicas de cada documento, logo as listas nao vao ter o mesmo tamanho
                            matrix.append(elem)
                            i += 1
                        mat_inv = tf_idf_vectorizer(stem_word_file)  # vai ser sempre 1 pq a palavra vai estar lá? E aqui vai sempre comparar com a ultima palavra.
                        cos_out = cos_sim(matrix, mat_inv)
                        for j in list(match_word):
                            output_data[j] = cos_out[list(match_word).index(j)]

            else:
                if len(pointer) == 0:
                    output_data = {}
                else:
                    for j in pointer:
                        print(f"j: {j}")
                        if search_type == "publication":
                            temp_file.append(pub_texts[j])
                        elif search_type == "abstract":
                            temp_file.append(pub_texts[j])

                    if rank_by == "Sklearn function":
                        temp_file = tfidf.fit_transform(temp_file)
                        print(f"stem_word_file: {stem_word_file}")
                        cosine_output = cosine_similarity(temp_file, tfidf.transform(stem_word_file))
                        print("cosine_output_and: ", cosine_output)
                        # print(f"list(match_word):{list(match_word)}")
                        for j in list(match_word):
                            output_data[j] = cosine_output[list(match_word).index(j)]
                            print(f"output_data_fim: {output_data}")
                    else:
                        i = 0
                        matrix = []
                        for elem in temp_file:
                            elem = tf_idf_vectorizer(elem.split())[0]  # lista de vetores tf-idf para cada documento, com as palabvras unicas de cada documento, logo as listas nao vao ter o mesmo tamanho
                            matrix.append(elem)
                            i += 1
                        mat_inv = tf_idf_vectorizer(stem_word_file)  # vai ser sempre 1 pq a palavra vai estar lá? E aqui vai sempre comparar com a ultima palavra.
                        cos_out = cos_sim(matrix, mat_inv)
                        for j in list(match_word):
                            output_data[j] = cos_out[list(match_word).index(j)]

    return output_data