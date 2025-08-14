#!/usr/bin/env python3
"""
Utilitário para executar scripts de automação
Facilita a execução e gerenciamento dos scripts
"""

import sys
import subprocess
import argparse
from pathlib import Path

class AutomationRunner:
    def __init__(self):
        self.scripts_dir = Path(__file__).parent / "scripts"
        self.config_dir = Path(__file__).parent / "config"
        
        self.scripts_disponiveis = {
            'organizador': {
                'arquivo': 'organizador_arquivos.py',
                'descricao': 'Organiza arquivos por extensão',
                'config_exemplo': 'organizador_exemplo.json'
            },
            'monitor': {
                'arquivo': 'monitor_sistema.py',
                'descricao': 'Monitora recursos do sistema',
                'config_exemplo': 'monitor_exemplo.json'
            },
            'scraper': {
                'arquivo': 'web_scraper.py',
                'descricao': 'Faz web scraping de sites',
                'config_exemplo': 'scraper_exemplo.json'
            }
        }
    
    def listar_scripts(self):
        """Lista todos os scripts disponíveis"""
        print("🤖 Scripts de Automação Disponíveis:\n")
        
        for nome, info in self.scripts_disponiveis.items():
            print(f"  {nome:12} - {info['descricao']}")
            arquivo_script = self.scripts_dir / info['arquivo']
            arquivo_config = self.config_dir / info['config_exemplo']
            
            status_script = "✅" if arquivo_script.exists() else "❌"
            status_config = "✅" if arquivo_config.exists() else "❌"
            
            print(f"               Script: {status_script} | Config: {status_config}")
            print()
    
    def executar_script(self, nome_script, argumentos_extras=None):
        """Executa um script específico"""
        if nome_script not in self.scripts_disponiveis:
            print(f"❌ Script '{nome_script}' não encontrado!")
            self.listar_scripts()
            return False
        
        info_script = self.scripts_disponiveis[nome_script]
        arquivo_script = self.scripts_dir / info_script['arquivo']
        
        if not arquivo_script.exists():
            print(f"❌ Arquivo do script não encontrado: {arquivo_script}")
            return False
        
        # Monta comando
        cmd = [sys.executable, str(arquivo_script)]
        
        # Adiciona argumentos extras
        if argumentos_extras:
            cmd.extend(argumentos_extras)
        
        print(f"🚀 Executando: {' '.join(cmd)}")
        print("-" * 50)
        
        try:
            # Executa o script
            resultado = subprocess.run(cmd, cwd=self.scripts_dir.parent)
            return resultado.returncode == 0
        except KeyboardInterrupt:
            print("\n⚠️ Execução interrompida pelo usuário")
            return False
        except Exception as e:
            print(f"❌ Erro ao executar script: {e}")
            return False
    
    def mostrar_help_script(self, nome_script):
        """Mostra ajuda de um script específico"""
        if nome_script not in self.scripts_disponiveis:
            print(f"❌ Script '{nome_script}' não encontrado!")
            return
        
        info_script = self.scripts_disponiveis[nome_script]
        arquivo_script = self.scripts_dir / info_script['arquivo']
        
        print(f"📖 Ajuda para: {nome_script}")
        print(f"Descrição: {info_script['descricao']}")
        print("-" * 50)
        
        # Executa script com --help
        cmd = [sys.executable, str(arquivo_script), '--help']
        subprocess.run(cmd, cwd=self.scripts_dir.parent)
    
    def mostrar_configuracao_exemplo(self, nome_script):
        """Mostra exemplo de configuração"""
        if nome_script not in self.scripts_disponiveis:
            print(f"❌ Script '{nome_script}' não encontrado!")
            return
        
        info_script = self.scripts_disponiveis[nome_script]
        arquivo_config = self.config_dir / info_script['config_exemplo']
        
        if not arquivo_config.exists():
            print(f"❌ Arquivo de configuração não encontrado: {arquivo_config}")
            return
        
        print(f"⚙️ Configuração exemplo para: {nome_script}")
        print(f"Arquivo: {arquivo_config}")
        print("-" * 50)
        
        try:
            with open(arquivo_config, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"❌ Erro ao ler configuração: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Utilitário para executar scripts de automação',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponíveis')
    
    # Lista scripts
    subparsers.add_parser('list', help='Lista todos os scripts disponíveis')
    
    # Executa script
    parser_run = subparsers.add_parser('run', help='Executa um script')
    parser_run.add_argument('script', choices=['organizador', 'monitor', 'scraper'],
                           help='Nome do script para executar')
    parser_run.add_argument('args', nargs='*', help='Argumentos para o script')
    
    # Ajuda de script
    parser_help = subparsers.add_parser('help', help='Mostra ajuda de um script')
    parser_help.add_argument('script', choices=['organizador', 'monitor', 'scraper'],
                            help='Nome do script')
    
    # Mostra configuração
    parser_config = subparsers.add_parser('config', help='Mostra configuração exemplo')
    parser_config.add_argument('script', choices=['organizador', 'monitor', 'scraper'],
                              help='Nome do script')
    
    args = parser.parse_args()
    
    runner = AutomationRunner()
    
    if not args.comando or args.comando == 'list':
        runner.listar_scripts()
    
    elif args.comando == 'run':
        runner.executar_script(args.script, args.args)
    
    elif args.comando == 'help':
        runner.mostrar_help_script(args.script)
    
    elif args.comando == 'config':
        runner.mostrar_configuracao_exemplo(args.script)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()