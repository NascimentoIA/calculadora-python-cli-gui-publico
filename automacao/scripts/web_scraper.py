#!/usr/bin/env python3
"""
Web Scraper Automático
Coleta dados de sites de forma automatizada
"""

import requests
from bs4 import BeautifulSoup
import logging
import json
import csv
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/web_scraper.log'),
        logging.StreamHandler()
    ]
)

class WebScraper:
    def __init__(self, config_path=None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.config = self.carregar_configuracao(config_path)
        self.dados_coletados = []
    
    def carregar_configuracao(self, config_path):
        """Carrega configuração do scraper"""
        config_padrao = {
            'delay_requisicoes': 1,  # segundos entre requisições
            'timeout': 30,
            'max_retries': 3,
            'salvar_html': False,
            'diretorio_html': '../logs/html_pages',
            'formato_saida': 'json',  # json, csv, both
            'arquivo_saida': '../logs/dados_coletados'
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_usuario = json.load(f)
                    config_padrao.update(config_usuario)
                logging.info(f"Configuração carregada de {config_path}")
            except Exception as e:
                logging.error(f"Erro ao carregar configuração: {e}")
        
        return config_padrao
    
    def fazer_requisicao(self, url, retries=0):
        """Faz requisição HTTP com retry"""
        try:
            response = self.session.get(
                url, 
                timeout=self.config['timeout']
            )
            response.raise_for_status()
            
            logging.info(f"Sucesso ao acessar: {url}")
            return response
            
        except requests.exceptions.RequestException as e:
            if retries < self.config['max_retries']:
                logging.warning(f"Erro ao acessar {url}, tentativa {retries + 1}: {e}")
                time.sleep(2 ** retries)  # Backoff exponencial
                return self.fazer_requisicao(url, retries + 1)
            else:
                logging.error(f"Falha definitiva ao acessar {url}: {e}")
                return None
    
    def salvar_html(self, url, conteudo):
        """Salva o HTML da página"""
        if not self.config['salvar_html']:
            return
        
        diretorio = Path(self.config['diretorio_html'])
        diretorio.mkdir(parents=True, exist_ok=True)
        
        # Nome do arquivo baseado na URL
        nome_arquivo = urlparse(url).netloc.replace('.', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arquivo_html = diretorio / f"{nome_arquivo}_{timestamp}.html"
        
        try:
            with open(arquivo_html, 'w', encoding='utf-8') as f:
                f.write(conteudo)
            logging.debug(f"HTML salvo: {arquivo_html}")
        except Exception as e:
            logging.error(f"Erro ao salvar HTML: {e}")
    
    def extrair_dados_genericos(self, soup, seletores):
        """Extrai dados usando seletores CSS genéricos"""
        dados = {}
        
        for campo, seletor in seletores.items():
            try:
                if isinstance(seletor, dict):
                    # Seletor avançado com múltiplas opções
                    elementos = soup.select(seletor['selector'])
                    if elementos:
                        if seletor.get('multiple', False):
                            # Múltiplos elementos
                            dados[campo] = []
                            for elem in elementos:
                                if seletor.get('attribute'):
                                    dados[campo].append(elem.get(seletor['attribute']))
                                else:
                                    dados[campo].append(elem.get_text(strip=True))
                        else:
                            # Primeiro elemento
                            elem = elementos[0]
                            if seletor.get('attribute'):
                                dados[campo] = elem.get(seletor['attribute'])
                            else:
                                dados[campo] = elem.get_text(strip=True)
                else:
                    # Seletor simples
                    elemento = soup.select_one(seletor)
                    if elemento:
                        dados[campo] = elemento.get_text(strip=True)
                    
            except Exception as e:
                logging.error(f"Erro ao extrair campo {campo}: {e}")
                dados[campo] = None
        
        return dados
    
    def scrape_noticias_exemplo(self, url_base):
        """Exemplo: Extrai notícias de um site"""
        seletores = {
            'titulo': 'h1, h2.titulo, .headline',
            'resumo': '.resumo, .abstract, .lead',
            'autor': '.autor, .byline, .author',
            'data': '.data, .date, time',
            'categoria': '.categoria, .category, .tag'
        }
        
        response = self.fazer_requisicao(url_base)
        if not response:
            return []
        
        self.salvar_html(url_base, response.text)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Busca links de artigos
        links_artigos = soup.select('a[href*="noticia"], a[href*="artigo"], a[href*="/news/"]')
        
        noticias = []
        for i, link in enumerate(links_artigos[:10]):  # Limite de 10 notícias
            url_completa = urljoin(url_base, link.get('href'))
            
            logging.info(f"Processando notícia {i+1}: {url_completa}")
            
            response_artigo = self.fazer_requisicao(url_completa)
            if response_artigo:
                soup_artigo = BeautifulSoup(response_artigo.content, 'html.parser')
                dados_noticia = self.extrair_dados_genericos(soup_artigo, seletores)
                dados_noticia['url'] = url_completa
                dados_noticia['timestamp_coleta'] = datetime.now().isoformat()
                
                noticias.append(dados_noticia)
                
                # Delay entre requisições
                time.sleep(self.config['delay_requisicoes'])
        
        return noticias
    
    def scrape_produtos_exemplo(self, url_loja):
        """Exemplo: Extrai produtos de uma loja online"""
        seletores = {
            'nome': '.product-name, .title, h1',
            'preco': {
                'selector': '.price, .valor, .preco',
                'attribute': None
            },
            'descricao': '.description, .desc, .product-desc',
            'disponibilidade': '.stock, .availability, .disponivel',
            'imagens': {
                'selector': '.product-image img, .gallery img',
                'attribute': 'src',
                'multiple': True
            }
        }
        
        response = self.fazer_requisicao(url_loja)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Busca links de produtos
        links_produtos = soup.select('a[href*="produto"], a[href*="product"], .product-link')
        
        produtos = []
        for i, link in enumerate(links_produtos[:5]):  # Limite de 5 produtos
            url_produto = urljoin(url_loja, link.get('href'))
            
            logging.info(f"Processando produto {i+1}: {url_produto}")
            
            response_produto = self.fazer_requisicao(url_produto)
            if response_produto:
                soup_produto = BeautifulSoup(response_produto.content, 'html.parser')
                dados_produto = self.extrair_dados_genericos(soup_produto, seletores)
                dados_produto['url'] = url_produto
                dados_produto['timestamp_coleta'] = datetime.now().isoformat()
                
                produtos.append(dados_produto)
                time.sleep(self.config['delay_requisicoes'])
        
        return produtos
    
    def salvar_dados(self, dados, prefixo='dados'):
        """Salva os dados coletados"""
        if not dados:
            logging.warning("Nenhum dado para salvar")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = Path(self.config['arquivo_saida'])
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salva em JSON
        if self.config['formato_saida'] in ['json', 'both']:
            arquivo_json = f"{base_path}_{prefixo}_{timestamp}.json"
            try:
                with open(arquivo_json, 'w', encoding='utf-8') as f:
                    json.dump(dados, f, indent=2, ensure_ascii=False)
                logging.info(f"Dados salvos em JSON: {arquivo_json}")
            except Exception as e:
                logging.error(f"Erro ao salvar JSON: {e}")
        
        # Salva em CSV
        if self.config['formato_saida'] in ['csv', 'both']:
            arquivo_csv = f"{base_path}_{prefixo}_{timestamp}.csv"
            try:
                if dados:
                    with open(arquivo_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=dados[0].keys())
                        writer.writeheader()
                        writer.writerows(dados)
                    logging.info(f"Dados salvos em CSV: {arquivo_csv}")
            except Exception as e:
                logging.error(f"Erro ao salvar CSV: {e}")
    
    def executar_scraping_personalizado(self, config_scraping):
        """Executa scraping baseado em configuração"""
        for site_config in config_scraping.get('sites', []):
            url = site_config['url']
            tipo = site_config.get('tipo', 'generico')
            seletores = site_config.get('seletores', {})
            
            logging.info(f"Iniciando scraping de {url} (tipo: {tipo})")
            
            if tipo == 'noticias':
                dados = self.scrape_noticias_exemplo(url)
                self.salvar_dados(dados, 'noticias')
            elif tipo == 'produtos':
                dados = self.scrape_produtos_exemplo(url)
                self.salvar_dados(dados, 'produtos')
            else:
                # Scraping genérico
                response = self.fazer_requisicao(url)
                if response:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    dados = [self.extrair_dados_genericos(soup, seletores)]
                    dados[0]['url'] = url
                    dados[0]['timestamp_coleta'] = datetime.now().isoformat()
                    self.salvar_dados(dados, 'genericos')

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web Scraper Automático')
    parser.add_argument('--config', help='Arquivo de configuração JSON')
    parser.add_argument('--url', help='URL para fazer scraping')
    parser.add_argument('--tipo', choices=['noticias', 'produtos', 'generico'], 
                        default='generico', help='Tipo de scraping')
    
    args = parser.parse_args()
    
    scraper = WebScraper(args.config)
    
    if args.url:
        # Scraping de URL única
        if args.tipo == 'noticias':
            dados = scraper.scrape_noticias_exemplo(args.url)
            scraper.salvar_dados(dados, 'noticias')
        elif args.tipo == 'produtos':
            dados = scraper.scrape_produtos_exemplo(args.url)
            scraper.salvar_dados(dados, 'produtos')
        else:
            response = scraper.fazer_requisicao(args.url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extração genérica básica
                dados = [{
                    'titulo': soup.title.get_text() if soup.title else '',
                    'url': args.url,
                    'timestamp_coleta': datetime.now().isoformat()
                }]
                scraper.salvar_dados(dados, 'generico')
    
    elif args.config:
        # Scraping baseado em configuração
        with open(args.config, 'r', encoding='utf-8') as f:
            config_scraping = json.load(f)
        scraper.executar_scraping_personalizado(config_scraping)
    
    else:
        print("Forneça uma --url ou arquivo --config")

if __name__ == "__main__":
    main()