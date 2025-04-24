# import os  # Module for interacting with the operating system
import json
import time  # Module for time-related operations
import ujson  # Module for working with JSON data
from random import randint  # Module for generating random numbers
from typing import Dict, List, Any  # Type hinting imports

import requests  # Library for making HTTP requests
from bs4 import BeautifulSoup  # Library for parsing HTML data
from selenium import webdriver  # Library for browser automation
from selenium.common.exceptions import NoSuchElementException  # Exception for missing elements
from webdriver_manager.chrome import ChromeDriverManager  # Driver manager for Chrome (We are using Chromium based )
from selenium.webdriver.common.by import By


# Delete files if present
# try:
#     os.remove('Authors_URL.txt')
#     os.remove('scraper_results.json')
# except OSError:
#     pass


# para salvar a lista de URLs dos autores
def write_authors(list1, file_name):
    # Function to write authors' URLs to a file
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(0, len(list1)):
            f.write(list1[i] + '\n')



#mudar o codigo que deu certo para ficar mais parecido ao do git
def initCrawlerScraper(seed, max_profiles=500):
    #MAX_NR_SEARCHES = 5 - coloquei quando foi para teste com um conjunto menor
    # Initialize driver for Chrome
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=webOpt)
    driver.get(seed)  # Start with the original link

    Links = []  # Array with pureportal profiles URL (dos autores)
    pub_data = []  # To store publication information for each pureportal profile

    nextLink = driver.find_element(By.CSS_SELECTOR, ".nextLink").is_enabled()  # Check if the next page link is enabled
    print("Crawler has begun...")

    # o loop seguinte é para conseguir os links dos autores
    while (nextLink):  # enquanto existirem páginas seguintes ativas
        page = driver.page_source  # Retorna o HTML completo da página como uma string
        # XML parser to parse each URL
        # Extrai os links dos perfis de pesquisadores do HTML da página usando BeautifulSoup
        bs = BeautifulSoup(page, "lxml")  # Parse the page source using BeautifulSoup

        # Extracting exact URL by spliting string into list
        for link in bs.findAll('a', class_='link person'):
            url = str(link)[str(link).find('https://pureportal.coventry.ac.uk/en/persons/'):].split('"')
            Links.append(
                url[0])  # Salva a URL do perfil do autor, vai acabar por ter todas as URLs dos perfis de pesquisadores

        # Click on Next button to visit next page
        try:
            if driver.find_element(By.CSS_SELECTOR, ".nextLink"):  # tenta encontrar o botao next
                element = driver.find_element(By.CSS_SELECTOR, ".nextLink")
                driver.execute_script("arguments[0].click();", element)  # clica no botao next
            else:
                nextLink = False
        except NoSuchElementException:
            break  # se o botao nao existir para o loop

        # Check if the maximum number of profiles is reached (se já chegamos aos 500 perfis)
        if len(Links) >= max_profiles:
            break

    #Links = Links[:MAX_NR_SEARCHES]

    print("Crawler has found ", len(Links), " pureportal profiles")
    print(f"Found {len(Links)} author profiles.")
    print(Links)
    write_authors(Links, 'Authors_URL.txt')  # Write the authors' URLs to a file

    print("Scraping publication data for ", len(Links), " pureportal profiles...")
    count = 0
    for link in Links:
        time.sleep(1)
        driver.get(link)
        time.sleep(2)

        try:
            # Verificar se existe um botão "View all research outputs"
            research_output_button = driver.find_elements(By.XPATH,
                                                          "//div[@class='view-all']/a[contains(@href, 'publications')]")
            if research_output_button:
                driver.execute_script("arguments[0].click();", research_output_button[0])
                time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'lxml')
            table = soup.find('ul', attrs={'class': 'list-results'})

            if table != None:
                author_name = soup.find('h1').text.strip()
                for row in table.findAll('div', class_='result-container'):
                    data = {}
                    data['name'] = row.h3.a.text.strip()
                    data['pub_url'] = row.h3.a['href']
                    data['cu_author'] = author_name
                    data['date'] = row.find("span", class_="date").text.strip() if row.find("span", class_="date") else "Unknown"
                    print("Publication Name :", row.h3.a.text.strip())
                    print("Publication URL :", row.h3.a['href'])
                    print("CU Author :", author_name)
                    print("Date :", row.find("span", class_="date").text.strip() if row.find("span", class_="date") else "Unknown")
                    print("\n")
                    pub_data.append(data)
        except NoSuchElementException:
            print("Research output button not found for this author.")

        #falta aqui o else

    print("Crawler has scrapped data for ", len(pub_data), " pureportal publications")
    driver.quit()

    with open('scraper_results.json', 'w') as f:
        ujson.dump(pub_data, f, indent=2)


initCrawlerScraper('https://pureportal.coventry.ac.uk/en/organisations/coventry-university/persons/', max_profiles=500)


def get_abstract():
    count = 0
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=webOpt)


    with open('scraper_results.json', 'r') as f:
        pub_data = ujson.load(f)
        for resultado in pub_data:
            #print("\n URL:", resultado["pub_url"])
            url = resultado["pub_url"] #url de cada publicação

            try:
                driver.get(url) #carrega a página da aplicação
                time.sleep(3)  # Aguarda a página carregar completamente

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Extrair abstract
                abstract_div = soup.select_one("div.rendering_researchoutput_abstractportal div.textblock")

                if abstract_div:
                    abstract_text = abstract_div.get_text(strip=True) #se abstract_div for encontrado, o código extrai o texto contido dentro dessa div
                    print(" Abstract encontrado")
                    resultado["abstract"] = abstract_text #abstract adicionado ao dicionario
                else:
                    print(" Abstract NÃO encontrado!")
                    count += 1

            except Exception as e:
                print("Erro ao processar URL:", url)
                print(str(e))
                count += 1

        print(f"\n Total de abstracts não encontrados: {count}")

        # Salvar os resultados atualizados
        with open('scraper_results_with_abstracts.json', 'w') as f:
            ujson.dump(pub_data, f, indent=2)

        driver.quit()

get_abstract()


import re


def get_groups():
    count = 0
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=webOpt)

    with open('scraper_results.json', 'r') as f:
        pub_data = ujson.load(f)
        for resultado in pub_data:
            # URL de cada publicação
            url = resultado["pub_url"]

            try:
                driver.get(url)  # Carrega a página da publicação
                time.sleep(3)  # Aguarda a página carregar completamente

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Encontrar o nome do grupo de pesquisa
                group_link = soup.find('a', rel='Organisation', class_='link')

                if group_link:
                    group_name = group_link.find(
                        'span').text.strip()  # Extrair o nome do grupo que está dentro do <span>
                    print(f"Grupo encontrado: {group_name}")

                    # Alterar o nome do grupo para uma lista
                    # Usar uma expressão regular para dividir a string em partes
                    group_list = re.split(r',\s*', group_name)  # Divide a string onde houver vírgula seguida de espaço

                    resultado["research_group"] = group_list  # Adiciona a lista de grupos à publicação
                else:
                    print("NENHUM GRUPO ENCONTRADO")
                    count += 1  # Contabiliza os casos sem grupo

            except Exception as e:
                print(f"Erro ao processar a URL {url}: {e}")
                count += 1  # Contabiliza falhas na extração

        print(f"Total de publicações sem grupo: {count}")

        # Salvar os resultados com os grupos associados
        with open('scraper_results_with_groups.json', 'w') as f:
            ujson.dump(pub_data, f, indent=2)

        driver.quit()  # Fechar o driver do Selenium


get_groups()

def adicionar_pdf_links(): #adicionar links dos pdfs nos resultados das pesquisas (odemos usar para isso, ou entao só colocar num grupo específico
    base_url = "https://pureportal.coventry.ac.uk"
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True

    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=webOpt)

    with open('scraper_results_with_groups.json', 'r', encoding='utf-8') as f:
        publicacoes = ujson.load(f)

    for pub in publicacoes:
        url = pub.get("pub_url")
        if not url:
            continue

        try:
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # Verifica se há link para o PDF diretamente
            pdf_tag = soup.find("a", class_="link document-link")
            if pdf_tag and pdf_tag.get("href"):
                pdf_url = pdf_tag["href"]
                if not pdf_url.startswith("http"):
                    pdf_url = base_url + pdf_url  # Corrigir URL relativo
                pub["link"] = pdf_url
                print(f"[✔] PDF encontrado para: {pub['name']}")
            else:
                print(f"[✘] Nenhum PDF encontrado para: {pub['name']}")

        except Exception as e:
            print(f"[Erro] {url}: {e}")

    # Guarda os dados com os links de PDF (se houver)
    with open('scraper_results_groups_links.json', 'w', encoding='utf-8') as f:
        ujson.dump(publicacoes, f, indent=2)

    driver.quit()

# Executa a função
adicionar_pdf_links()


'''
#função para ver se o numero de publicações de uma dado autor corresponde ao numero correto de aplicações

import json


def count_author_publications(author_name, file_path='scraper_results_with_abstracts.json'):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        count = sum(1 for pub in data if pub.get('cu_author') == author_name)

        print(f"O autor '{author_name}' tem {count} publicações.")
        return count

    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Erro: Problema ao ler o arquivo JSON '{file_path}'.")
        return None

#count_author_publications("Sally Abbott", file_path='scraper_results_with_abstracts.json')'''