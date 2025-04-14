# Descarregar PDFs e processar texto imediatamente
import os
import json
import time
import shutil
import pdfplumber
import ujson
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# NLTK config
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


#Descarregar pdfs da area LIB Mathematics Support Centre
import os
import json
import time
import shutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


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
                texto += pagina.extract_text() or ""
        return texto
    except Exception as e:
        print(f"Erro ao processar o PDF {caminho_pdf}: {e}")
        return ""

def limpar_texto(texto):
    texto = texto.lower()
    palavras = word_tokenize(texto)
    return ' '.join(stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words)

def descarregar_pdfs_LIB():
    with open("scraper_results_groups_links.json", "r", encoding="utf-8") as f:
        publicacoes = json.load(f)

    driver, temp_dir = configurar_navegador()
    pasta_destino = "pdfs_LIBMathematicsSupportCentre"
    os.makedirs(pasta_destino, exist_ok=True)

    grupo_alvo = "Centre for Healthcare and Communities"
    count = 0

    textos_limpos = []
    data_dict = {}

    for idx, pub in enumerate(publicacoes):
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
                texto_semcar = ''.join(c for c in texto_extraido if c.isalnum() or c.isspace())
                texto_limpo = limpar_texto(texto_semcar)
                textos_limpos.append(texto_limpo)

                for palavra in texto_limpo.split():
                    data_dict.setdefault(palavra, []).append(idx)  # usa idx original diretamente

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
    with open('pdf_list_stemmed.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_limpos, f, indent=2)

    with open('pdfs_indexed_dictionary.json', 'w', encoding='utf-8') as f:
        ujson.dump(data_dict, f, indent=2)

    print(f"\n[✅] Processo concluído! {count} PDFs do {grupo_alvo} foram baixados e processados.")


# Executar
descarregar_pdfs_LIB()
