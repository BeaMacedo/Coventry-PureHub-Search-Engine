import os
import json
import time
import shutil
import re
import pdfplumber
import ujson
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict
import nltk


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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

def configurar_navegador():
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    temp_dir = os.path.abspath("temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)

    prefs = {
        "download.default_directory": temp_dir,
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True
    }
    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver, temp_dir

def sanitize_filename(filename, max_length=100):
    invalid_chars = '<>:"/\\|?*'
    filename = ''.join(char if char not in invalid_chars else '_' for char in filename)
    return filename[:max_length].strip()

def extrair_texto_pdf(caminho_pdf):
    try:
        texto = ""
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                texto_pagina = pagina.extract_text(
                    x_tolerance=1,
                    y_tolerance=1,
                    layout=False,
                    x_density=7.25,
                    y_density=13
                )
                if texto_pagina:
                    texto_pagina = re.sub(r'(?<!\n)\n(?!\n)', ' ', texto_pagina)
                    texto += texto_pagina + "\n"
        return texto.strip()
    except Exception as e:
        print(f"Erro ao processar o PDF {caminho_pdf}: {e}")
        return ""

def limpar_texto(texto):
    if not texto:
        return ""

    # Remove URLs (http, https, www)
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)
    # Remove endereços de email
    texto = re.sub(r'\S+@\S+', '', texto)
    # Remove informações de copyright, DOI e ISBN
    texto = re.sub(r'©.*|DOI:.*|ISBN:.*', '', texto)
    # Junta palavras que foram separadas por hífen no final da linha
    texto = re.sub(r'-\n(\w+)', r'\1', texto)
    # Remove todos os números
    texto = re.sub(r'\d+', ' ', texto)
    # Substitui múltiplos espaços/quebras de linha por um único espaço
    texto = re.sub(r'\s+', ' ', texto).strip()
    #texto = re.sub(r'[^\w\s]|_', ' ', texto)  # Remove caracteres especiais

    return texto

def processar_texto_stem(texto):
    palavras = word_tokenize(texto.lower())
    palavras_filtradas = [
        stemmer.stem(palavra)
        for palavra in palavras
        if palavra not in stop_words
    ]
    return ' '.join(palavras_filtradas)

def processar_texto_lemma(texto):
    return enhanced_lemmatize(texto.lower())

def remover_referencias(texto):
    # Encontra a palavra 'references' no texto, ignorando case
    pos_references = texto.lower().find('references')

    if pos_references != -1:
        # Se a palavra 'references' for encontrada, retorna o texto antes dela
        return texto[:pos_references]

    # Retorna o texto original se a palavra 'references' não for encontrada
    return texto

def descarregar_pdfs(limite=40):
    with open("scraper_results_groups_links.json", "r", encoding="utf-8") as f:
        publicacoes = json.load(f)

    driver, temp_dir = configurar_navegador()
    #pasta_destino = "pdfs_LIBMathematicsSupportCentre"
    pasta_destino = "pdfs_CentreforHealthcareandCommunities"
    os.makedirs(pasta_destino, exist_ok=True)

    #grupo_alvo = "LIB Mathematics Support Centre"
    grupo_alvo = "Centre for Healthcare and Communities"
    count = 0

    textos_limpos = []
    textos_stem = []
    textos_lemma = []
    dic_indices_pdfs = {}
    dic_titulospdf_indiceI = {} #vai armazenar os indices iniciais e o nome do artigo da publicação para identificação mais rapida
    index_invertido_stem = {}
    index_invertido_lemma = {}

    for idx, pub in enumerate(publicacoes):
        # Verifica se atingiu o limite
        if count >= limite:
            print(f"\n[⏹] Limite de {limite} PDFs atingido. Interrompendo downloads.")
            break

        grupos = pub.get("research_group", [])
        pdf_url = pub.get("link")
        titulo = pub.get("name", "sem_titulo")

        if grupo_alvo not in grupos or not pdf_url:
            continue

        titulo_sanitizado = sanitize_filename(titulo) + ".pdf"
        caminho_final = os.path.join(pasta_destino, titulo_sanitizado)

        if os.path.exists(caminho_final):
            print(f"[↪] Já existe: {titulo_sanitizado}")
            continue

        print(f"\n[📥] A baixar a publicação {count + 1} do {grupo_alvo}: {titulo}")

        try:
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass

            driver.get(pdf_url)
            time.sleep(8)

            downloaded_files = []
            start_time = time.time()
            while time.time() - start_time < 30:
                downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]
                if downloaded_files:
                    break
                time.sleep(1)

            if downloaded_files:
                arquivo_baixado = os.path.join(temp_dir, downloaded_files[0])
                try:
                    shutil.move(arquivo_baixado, caminho_final)
                    print(f"[✔] Salvo como: {titulo_sanitizado}")
                except Exception as e:
                    titulo_curto = sanitize_filename(titulo, 50) + ".pdf"
                    caminho_curto = os.path.join(pasta_destino, titulo_curto)
                    shutil.move(arquivo_baixado, caminho_curto)
                    print(f"[✔] Salvo (nome curto): {titulo_curto}")
                    caminho_final = caminho_curto


                # --- Processamento do texto diretamente aqui ---
                texto_extraido = extrair_texto_pdf(caminho_final)

                #texto_semrefs = remover_referencias(texto_extraido)
                #texto_semcar = re.sub(r'[^\w\s]|_', ' ', texto_semrefs)  # Remove caracteres especiais

                texto_semcar = re.sub(r'[^\w\s]|_', ' ', texto_extraido)  # Remove caracteres especiais
                texto_limpo = limpar_texto(texto_semcar)
                textos_limpos.append(texto_limpo) #texto limpo para depois mostrar na pesquisa da app

                texto_stem = processar_texto_stem(texto_limpo)
                textos_stem.append(texto_stem)

                texto_lemma = processar_texto_lemma(texto_limpo)
                textos_lemma.append(texto_lemma)

                dic_indices_pdfs[count] = idx
                dic_titulospdf_indiceI[count] = titulo

                for palavra in texto_stem.split():
                    if palavra not in index_invertido_stem:
                        index_invertido_stem[palavra] = [idx]
                    else:
                        index_invertido_stem[palavra].append(idx)

                for palavra in texto_lemma.split():
                    if palavra not in index_invertido_lemma:
                        index_invertido_lemma[palavra] = [idx]
                    else:
                        index_invertido_lemma[palavra].append(idx)


                print(f"✅ Texto processado e indexado. idx original: {idx}, índice PDF: {count}")
                count += 1

            else:
                print(f"[⚠] PDF não baixado: {pdf_url}")

        except Exception as e:
            print(f"[❌] Erro ao processar {titulo}: {str(e)}")

    driver.quit()
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    # Salvar resultados
    with open('pdf_texts.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_limpos, f, indent=2)

    with open('pdf_list_stemmed.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_stem, f, indent=2)

    with open('pdf_list_lemma.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_lemma, f, indent=2)

    with open('pdfs_indexed_dictionary_stemm.json', 'w', encoding='utf-8') as f:
        ujson.dump(index_invertido_stem, f, indent=2)

    with open('pdfs_indexed_dictionary_lema.json', 'w', encoding='utf-8') as f:
        ujson.dump(index_invertido_lemma, f, indent=2)

    with open("dic_indices_pdfs.json", "w", encoding="utf-8") as f:
        json.dump(dic_indices_pdfs, f, ensure_ascii=False, indent=4)

    with open("dic_titulospdf_indiceI.json", "w", encoding="utf-8") as f:
        json.dump(dic_titulospdf_indiceI, f, ensure_ascii=False, indent=4)

    print(f"\n[✅] Processo concluído! {count} PDFs do {grupo_alvo} foram baixados e processados.")


descarregar_pdfs()
