#!/usr/bin/env python3
"""
Organizador de Arquivos Automático
Organiza arquivos em pastas baseado na extensão
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime
import json

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/organizador_arquivos.log'),
        logging.StreamHandler()
    ]
)

class OrganizadorArquivos:
    def __init__(self, pasta_origem, pasta_destino=None):
        self.pasta_origem = Path(pasta_origem)
        self.pasta_destino = Path(pasta_destino) if pasta_destino else self.pasta_origem
        
        # Mapeamento de extensões para pastas
        self.mapeamento_extensoes = {
            'imagens': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'],
            'documentos': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
            'planilhas': ['.xls', '.xlsx', '.csv', '.ods'],
            'apresentacoes': ['.ppt', '.pptx', '.odp'],
            'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'audios': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
            'compactados': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
            'executaveis': ['.exe', '.msi', '.deb', '.rpm', '.dmg', '.app'],
            'codigo': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php']
        }
    
    def carregar_configuracao(self, arquivo_config):
        """Carrega configuração personalizada"""
        try:
            with open(arquivo_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'mapeamento_extensoes' in config:
                    self.mapeamento_extensoes.update(config['mapeamento_extensoes'])
            logging.info(f"Configuração carregada de {arquivo_config}")
        except FileNotFoundError:
            logging.warning(f"Arquivo de configuração {arquivo_config} não encontrado")
    
    def obter_pasta_destino(self, extensao):
        """Determina a pasta destino baseada na extensão"""
        extensao = extensao.lower()
        for pasta, extensoes in self.mapeamento_extensoes.items():
            if extensao in extensoes:
                return pasta
        return 'outros'
    
    def criar_pastas(self):
        """Cria as pastas necessárias"""
        pastas = list(self.mapeamento_extensoes.keys()) + ['outros']
        for pasta in pastas:
            pasta_path = self.pasta_destino / pasta
            pasta_path.mkdir(exist_ok=True)
            logging.debug(f"Pasta criada/verificada: {pasta_path}")
    
    def organizar_arquivos(self, modo_teste=False):
        """Organiza os arquivos"""
        arquivos_organizados = 0
        arquivos_ignorados = 0
        
        logging.info(f"Iniciando organização em: {self.pasta_origem}")
        
        if not modo_teste:
            self.criar_pastas()
        
        for arquivo in self.pasta_origem.iterdir():
            if arquivo.is_file():
                extensao = arquivo.suffix
                pasta_destino = self.obter_pasta_destino(extensao)
                caminho_destino = self.pasta_destino / pasta_destino / arquivo.name
                
                if modo_teste:
                    logging.info(f"[TESTE] {arquivo.name} -> {pasta_destino}/")
                else:
                    try:
                        # Verifica se arquivo já existe no destino
                        if caminho_destino.exists():
                            # Adiciona timestamp ao nome
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            nome_sem_ext = arquivo.stem
                            caminho_destino = self.pasta_destino / pasta_destino / f"{nome_sem_ext}_{timestamp}{extensao}"
                        
                        shutil.move(str(arquivo), str(caminho_destino))
                        logging.info(f"Movido: {arquivo.name} -> {pasta_destino}/")
                        arquivos_organizados += 1
                        
                    except Exception as e:
                        logging.error(f"Erro ao mover {arquivo.name}: {e}")
                        arquivos_ignorados += 1
            else:
                logging.debug(f"Ignorando diretório: {arquivo.name}")
                arquivos_ignorados += 1
        
        logging.info(f"Organização concluída! Arquivos organizados: {arquivos_organizados}, Ignorados: {arquivos_ignorados}")
        return arquivos_organizados, arquivos_ignorados

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Organizador de Arquivos Automático')
    parser.add_argument('pasta_origem', help='Pasta com arquivos para organizar')
    parser.add_argument('--destino', help='Pasta de destino (padrão: mesma pasta de origem)')
    parser.add_argument('--config', help='Arquivo de configuração JSON')
    parser.add_argument('--teste', action='store_true', help='Modo teste (não move arquivos)')
    
    args = parser.parse_args()
    
    organizador = OrganizadorArquivos(args.pasta_origem, args.destino)
    
    if args.config:
        organizador.carregar_configuracao(args.config)
    
    organizador.organizar_arquivos(modo_teste=args.teste)

if __name__ == "__main__":
    main()