'''#Descarregar pdfs da area LIB Mathematics Support Centre
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

import re


def limpar_texto(texto):
    if not texto:
        return ""
    # Remove URLs (http, https, www)
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)
    # Remove endereços de email
    texto = re.sub(r'\S+@\S+', '', texto)
    # Remove informações de copyright, DOI e ISBN
    texto = re.sub(r'©.*|DOI:.*|ISBN:.*', '', texto)
    # Junta palavras que foram quebradas por hífen no final da linha
    texto = re.sub(r'-\n(\w+)', r'\1', texto)
    # Remove todos os números
    texto = re.sub(r'\d+', ' ', texto)
    # Substitui múltiplos espaços/quebras de linha por um único espaço
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


def processar_texto_stem(texto):
    palavras = word_tokenize(texto.lower())
    palavras_filtradas = [
        stemmer.stem(palavra)
        for palavra in palavras
        if (palavra not in stop_words and
            len(palavra) > 2 and
            palavra.isalpha() and
            not palavra.isnumeric())
    ]
    return ' '.join(palavras_filtradas)

def processar_texto_lemma(texto):
    texto_limpo = limpar_texto(texto)
    return enhanced_lemmatize(texto_limpo)


def descarregar_pdfs_LIB():
    with open("scraper_results_groups_links.json", "r", encoding="utf-8") as f:
        publicacoes = json.load(f)

    driver, temp_dir = configurar_navegador()
    pasta_destino = "pdfs_LIBMathematicsSupportCentre"
    os.makedirs(pasta_destino, exist_ok=True)

    grupo_alvo = "Centre for Healthcare and Communities"
    count = 0

    textos_stem = []
    textos_lemma = []
    textos_pdf = []
    dic_indices_pdfs = {}
    index_invertido_stem = defaultdict(list)
    index_invertido_lemma = defaultdict(list)

    for idx, pub in enumerate(publicacoes):
        grupos = pub.get("research_group", [])
        pdf_url = pub.get("link")
        titulo = pub.get("name", "sem_titulo")

        if grupo_alvo not in grupos or not pdf_url:
            continue

        titulo_sanitizado = sanitize_filename(titulo) + ".pdf"
        caminho_final = os.path.join(pasta_destino, titulo_sanitizado)

        texto_extraido = ""

        if os.path.exists(caminho_final):
            print(f"[↪] Já existe: {titulo_sanitizado}")
            texto_extraido = extrair_texto_pdf(caminho_final)
        else:
            print(f"\n[📥] Baixando publicação {count + 1} do {grupo_alvo}: {titulo}")
            try:
                for f in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass

                driver.get(pdf_url)
                time.sleep(5)

                downloaded_files = []
                start_time = time.time()
                while time.time() - start_time < 20:
                    downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.pdf')]
                    if downloaded_files:
                        break
                    time.sleep(0.5)

                if downloaded_files:
                    arquivo_baixado = os.path.join(temp_dir, downloaded_files[0])
                    try:
                        shutil.move(arquivo_baixado, caminho_final)
                        print(f"[✔] Salvo como: {titulo_sanitizado}")
                    except Exception:
                        titulo_curto = sanitize_filename(titulo, 50) + ".pdf"
                        caminho_curto = os.path.join(pasta_destino, titulo_curto)
                        shutil.move(arquivo_baixado, caminho_curto)
                        print(f"[✔] Salvo (nome curto): {titulo_curto}")
                        caminho_final = caminho_curto

                    texto_extraido = extrair_texto_pdf(caminho_final)
                else:
                    print(f"[⚠] PDF não baixado: {pdf_url}")
                    continue

            except Exception as e:
                print(f"[❌] Erro ao processar {titulo}: {str(e)}")
                continue

        # Processar texto com ambos os métodos
        texto_limpo = limpar_texto(texto_extraido)
        textos_pdf.append(texto_limpo)

        texto_stem = processar_texto_stem(texto_limpo)
        textos_stem.append(texto_stem)

        texto_lemma = processar_texto_lemma(texto_limpo)
        textos_lemma.append(texto_lemma)

        # Atualizar índices invertidos
        for termo in set(texto_stem.split()):
            index_invertido_stem[termo].append(count)

        for termo in set(texto_lemma.split()):
            index_invertido_lemma[termo].append(count)

        dic_indices_pdfs[count] = titulo
        count += 1

    driver.quit()
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    # Salvar resultados
    with open('pdf_texts.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_pdf, f, indent=2)

    with open('pdf_list_stemmed.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_stem, f, indent=2)

    with open('pdf_list_lemma.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_lemma, f, indent=2)

    with open('pdfs_indexed_dictionary_stem.json', 'w', encoding='utf-8') as f:
        ujson.dump(dict(index_invertido_stem), f, indent=2)

    with open('pdfs_indexed_dictionary_lemma.json', 'w', encoding='utf-8') as f:
        ujson.dump(dict(index_invertido_lemma), f, indent=2)

    with open("dic_indices_pdfs.json", "w", encoding="utf-8") as f:
        json.dump(dic_indices_pdfs, f, ensure_ascii=False, indent=4)

    print(f"\n[✅] Processo concluído! {count} PDFs do {grupo_alvo} foram processados.")
    print(f"- Versão com stemming: pdf_list_stemmed.json")
    print(f"- Versão com lematização: pdf_list_lemma.json")


#descarregar_pdfs_LIB()

import json


def carregar_arquivos():
    """Carrega todos os arquivos necessários"""
    with open('dic_indices_pdfs.json', 'r', encoding='utf-8') as f:
        dic_indices = json.load(f)

    with open('scraper_results.json', 'r', encoding='utf-8') as f:
        scrapper_results = json.load(f)

    with open('pdfs_indexed_dictionary_stem.json', 'r', encoding='utf-8') as f:
        stem_index = json.load(f)

    with open('pdfs_indexed_dictionary_lemma.json', 'r', encoding='utf-8') as f:
        lemma_index = json.load(f)

    return dic_indices, scrapper_results, stem_index, lemma_index


def criar_mapeamento_novos_indices(dic_indices, scrapper_results):
    """Cria o mapeamento de índices antigos para novos"""
    # Criar dicionário de nome para índice no scrapper
    nome_para_indice = {pub['name']: str(idx) for idx, pub in enumerate(scrapper_results)}

    # Criar novo dicionário de mapeamento
    dic_indices_pdfs2 = {}
    for old_idx, pub_name in dic_indices.items():
        new_idx = nome_para_indice.get(pub_name, "-1")  # Usa -1 se não encontrar
        dic_indices_pdfs2[old_idx] = new_idx

    return dic_indices_pdfs2


def atualizar_indices_invertidos(index_data, mapeamento):
    """Atualiza os índices em um índice invertido"""
    novo_index = {}
    for termo, indices in index_data.items():
        novos_indices = [mapeamento[str(idx)] for idx in indices if str(idx) in mapeamento]
        novo_index[termo] = novos_indices
    return novo_index


def salvar_arquivos(dic_indices_pdfs2, stem_index2, lemma_index2):
    """Salva todos os arquivos atualizados"""
    with open('dic_indices_pdfs2.json', 'w', encoding='utf-8') as f:
        json.dump(dic_indices_pdfs2, f, indent=4, ensure_ascii=False)

    with open('pdfs_indexed_dictionary_stem2.json', 'w', encoding='utf-8') as f:
        json.dump(stem_index2, f, indent=4, ensure_ascii=False)

    with open('pdfs_indexed_dictionary_lemma2.json', 'w', encoding='utf-8') as f:
        json.dump(lemma_index2, f, indent=4, ensure_ascii=False)


def processar_indices():
    """Função principal que orquestra todo o processamento"""
    print("Carregando arquivos...")
    dic_indices, scrapper_results, stem_index, lemma_index = carregar_arquivos()

    print("Criando mapeamento de novos índices...")
    dic_indices_pdfs2 = criar_mapeamento_novos_indices(dic_indices, scrapper_results)

    print("Atualizando índice invertido (stem)...")
    stem_index2 = atualizar_indices_invertidos(stem_index, dic_indices_pdfs2)

    print("Atualizando índice invertido (lemma)...")
    lemma_index2 = atualizar_indices_invertidos(lemma_index, dic_indices_pdfs2)

    print("Salvando arquivos atualizados...")
    salvar_arquivos(dic_indices_pdfs2, stem_index2, lemma_index2)

    print("\nProcesso concluído com sucesso!")
    print(f"- Total de publicações mapeadas: {len(dic_indices_pdfs2)}")
    print(f"- Termos no índice stem: {len(stem_index2)}")
    print(f"- Termos no índice lemma: {len(lemma_index2)}")


processar_indices()
'''