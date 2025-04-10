
#--------------------Índice invertido pdfs-------------------------------
import os
import PyPDF2
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ujson
import nltk

# Baixar stopwords e punkt do NLTK, caso não tenha feito antes
nltk.download('stopwords')
nltk.download('punkt')

# Definir o stemmer e as stopwords
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def limpar_texto(texto):
    # Converter o texto para minúsculas
    texto = texto.lower()

    # Tokenizar o texto em palavras
    palavras = word_tokenize(texto)

    # Remover stopwords e aplicar stemming
    texto_limpo = ' '.join(stemmer.stem(palavra) for palavra in palavras if palavra not in stop_words)

    return texto_limpo

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
    with open('dic_indices_pdfs.json', 'r', encoding='utf-8') as f:
        indices = ujson.load(f)

    # Diretório onde os PDFs estão localizados
    pasta_pdfs = "pdfs_LIBMathematicsSupportCentre"

    # Lista para armazenar textos extraídos e os índices dos PDFs
    textos_limpos = []
    textos_semcar_limpos = []
    index_documentos = []

    # Percorrer todos os PDFs na pasta
    for nome_arquivo in os.listdir(pasta_pdfs):
        caminho_pdf = os.path.join(pasta_pdfs, nome_arquivo)

        if nome_arquivo.endswith(".pdf"):
            print(f"A processar PDF: {nome_arquivo}")

            # Extrair o texto do PDF
            texto_extraido = extrair_texto_pdf(caminho_pdf)

            # Limpar o texto extraído (remover stop words e aplicar stemming)
            texto_limpo = limpar_texto(texto_extraido)

            # Remover caracteres especiais do texto original e aplicar limpar_texto
            texto_original_sem_caracteres = ''.join(e for e in texto_extraido if e.isalnum() or e.isspace())
            texto_semcar_limpo = limpar_texto(texto_original_sem_caracteres)

            if texto_limpo:
                textos_limpos.append(texto_limpo)
                textos_semcar_limpos.append(texto_semcar_limpo)
                index_documentos.append(nome_arquivo)  # Guardar o nome do arquivo como identificador

    # Criar o índice invertido usando os valores do arquivo JSON 'dic_indices_pdfs'
    data_dict = {}
    for i, texto in enumerate(textos_semcar_limpos):
        doc_index = indices[str(i)]  # Obter o índice original do documento a partir do arquivo JSON
        for palavra in texto.split():
            if palavra not in data_dict:
                data_dict[palavra] = [doc_index]
            else:
                data_dict[palavra].append(doc_index)

    # Salvar o índice invertido e os textos limpos
    with open('pdfs_indexed_dictionary.json', 'w', encoding='utf-8') as f:
        ujson.dump(data_dict, f, indent=2)

    with open('pdf_list_stemmed.json', 'w', encoding='utf-8') as f:
        ujson.dump(textos_limpos, f, indent=2)


    print(f"\n[✅] Processamento completo. {len(textos_limpos)} PDFs processados e indexados.")

# Chamar a função para processar os PDFs
processar_pdfs()
