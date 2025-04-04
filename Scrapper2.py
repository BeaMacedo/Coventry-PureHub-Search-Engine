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


"""# para salvar a lista de URLs dos autores
def write_authors(list1, file_name):
    # Function to write authors' URLs to a file
    with open(file_name, 'w', encoding='utf-8') as f:
        for i in range(0, len(list1)):
            f.write(list1[i] + '\n')



#mudar o codigo que deu certo para ficar mais parecido ao do git
def initCrawlerScraper(seed, max_profiles=500):
    MAX_NR_SEARCHES = 10
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

    Links = Links[:MAX_NR_SEARCHES]

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
"""

#este deu certo:
"""def initCrawlerScraper(seed: str, max_profiles=500):
    MAX_NR_SEARCHES = 20

    # Initialize driver for Chrome
    webOpt = webdriver.ChromeOptions()
    webOpt.add_experimental_option('excludeSwitches', ['enable-logging'])
    webOpt.add_argument('--ignore-certificate-errors')
    webOpt.add_argument('--incognito')
    webOpt.headless = True
    driver = webdriver.Chrome(service=webdriver.ChromeService(ChromeDriverManager().install()), options=webOpt)
    driver.get(seed)

    Links = []  # Array with pureportal profiles URL (dos autores)
    pub_data = []  # To store publication information for each pureportal profile

    print("Crawler has begun...")

    while True:
        page = driver.page_source
        soup = BeautifulSoup(page, "lxml")

        # Extrair links dos perfis de autores
        for link in soup.findAll('a', class_='link person'):
            url = link.get('href')
            if url and url.startswith('https://pureportal.coventry.ac.uk/en/persons/'):
                Links.append(url)

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, ".nextLink")
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(2)
        except NoSuchElementException:
            break

        if len(Links) >= max_profiles:
            break

    Links = Links[:MAX_NR_SEARCHES]
    print(f"Found {len(Links)} author profiles.")
    write_authors(Links, 'Authors_URL.txt')

    print("Scraping publication data...")
    for link in Links:
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

            if table:
                author_name = soup.find('h1').text.strip()
                for row in table.findAll('div', class_='result-container'):
                    data = {
                        'name': row.h3.a.text.strip(),
                        'pub_url': row.h3.a['href'],
                        'cu_author': author_name,
                        'date': row.find("span", class_="date").text.strip() if row.find("span",
                                                                                         class_="date") else "Unknown"
                    }
                    pub_data.append(data)
        except NoSuchElementException:
            print("Research output button not found for this author.")

    print(f"Scraped data for {len(pub_data)} publications.")
    driver.quit()

    with open('scraper_results.json', 'w') as f:
        ujson.dump(pub_data, f, indent=2)


initCrawlerScraper('https://pureportal.coventry.ac.uk/en/organisations/coventry-university/persons/', max_profiles=500)
"""
"""
def initCrawlerScraper(seed, max_profiles=500):
    MAX_NR_SEARCHES = 200
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

    Links = Links[:MAX_NR_SEARCHES]

    print("Crawler has found ", len(Links), " pureportal profiles")
    print(f"Found {len(Links)} author profiles.")
    print(Links)
    write_authors(Links, 'Authors_URL.txt')  # Write the authors' URLs to a file

    # -------------agora já temos o ficheiro "Authors_URL.txt" com os links de todos os perfis-----------------------

    # Agora vamos a cada perfil e conseguir as informações das publicações de cada autor
    print("Scraping publication data for ", len(Links), " pureportal profiles...")
    count = 0
    for link in Links:  # vai a cada link da lista Links com os links de todos os perfis
        # Visit each link to get data
        time.sleep(1)  # Delay of 1 second to hit next data
        driver.get(link)
    try:
        # Procurar pelo link "Research Output" dentro da lista de navegação
        research_output_button = driver.find_elements(By.XPATH, "//a[contains(@href, 'publications') and contains(., 'Research output')]")
        if research_output_button:
            research_output_button[0].click()  # Clica no botão "Research output"
            driver.get(driver.current_url)

            # Get name of Author
            name = driver.find_elements(By.CSS_SELECTOR, "div[class='header person-details']>h1")
            r = requests.get(driver.current_url)
            # Parse all the data via BeautifulSoup
            soup = BeautifulSoup(r.content, 'lxml')

            # Extracting publication name, publication url, date and CU Authors
            table = soup.find('ul', attrs={'class': 'list-results'})
            if table != None:
                for row in table.findAll('div', attrs={'class': 'result-container'}):
                    data = {}
                    data['name'] = row.h3.a.text
                    data['pub_url'] = row.h3.a['href']
                    date = row.find("span", class_="date")

                    rowitem = row.find_all(['div'])
                    span = row.find_all(['span'])
                    data['cu_author'] = name.text
                    data['date'] = date.text
                    print("Publication Name :", row.h3.a.text)
                    print("Publication URL :", row.h3.a['href'])
                    print("CU Author :", name.text)
                    print("Date :", date.text)
                    print("\n")
                    pub_data.append(data)  # Baixa os dados de cada publicação
        else:
            print("Research output button not found")
    except Exception as e:
        print(f"Error: {e}")

    print("Crawler has scrapped data for ", len(pub_data), " pureportal publications")
    driver.quit()


    # Writing all the scraped results in a file with JSON format
    with open('scraper_results.json', 'w') as f:
        ujson.dump(pub_data, f) # todas as informações das publicações são salvas no arquivo scraper_results.json

#O arquivo scraper_results.json contém uma lista de dicionários, onde cada dicionário representa uma publicação extraída do site, com as chaves name, pub_url, cu_author e date

initCrawlerScraper('https://pureportal.coventry.ac.uk/en/organisations/coventry-university/persons/', max_profiles=500)
"""


#resumo:
#1ºO código acede ao portal da Coventry University e coleta os perfis dos pesquisadores.
#2º Para cada perfil, ele acessa a seção "Research Output" e coleta as publicações.
#3º As informações são organizadas e salvas no arquivo scraper_results.json.


