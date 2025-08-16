#!/usr/bin/env python3
"""
Monitor de Sistema Automático
Monitora CPU, memória, disco e processos
"""

import psutil
import logging
import time
import json
from datetime import datetime
from pathlib import Path

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/monitor_sistema.log'),
        logging.StreamHandler()
    ]
)

class MonitorSistema:
    def __init__(self, config_path=None):
        self.config = self.carregar_configuracao(config_path)
        self.alertas_enviados = set()
    
    def carregar_configuracao(self, config_path):
        """Carrega configuração do monitor"""
        config_padrao = {
            'limites': {
                'cpu_percent': 80.0,
                'memoria_percent': 85.0,
                'disco_percent': 90.0,
                'temperatura_cpu': 70.0
            },
            'intervalo_monitoramento': 30,  # segundos
            'salvar_historico': True,
            'arquivo_historico': '../logs/historico_sistema.json'
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
    
    def obter_info_cpu(self):
        """Obtém informações da CPU"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        info_cpu = {
            'percentual_uso': cpu_percent,
            'nucleos': cpu_count,
            'frequencia_atual': cpu_freq.current if cpu_freq else None,
            'frequencia_max': cpu_freq.max if cpu_freq else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Verifica temperatura se disponível
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp_cpu = max([sensor.current for sensor in temps['coretemp']])
                info_cpu['temperatura'] = temp_cpu
        except:
            pass
        
        return info_cpu
    
    def obter_info_memoria(self):
        """Obtém informações da memória"""
        memoria = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': round(memoria.total / (1024**3), 2),
            'disponivel_gb': round(memoria.available / (1024**3), 2),
            'usado_gb': round(memoria.used / (1024**3), 2),
            'percentual_uso': memoria.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_usado_gb': round(swap.used / (1024**3), 2),
            'swap_percentual': swap.percent,
            'timestamp': datetime.now().isoformat()
        }
    
    def obter_info_disco(self):
        """Obtém informações dos discos"""
        discos = []
        
        for particao in psutil.disk_partitions():
            try:
                uso = psutil.disk_usage(particao.mountpoint)
                discos.append({
                    'dispositivo': particao.device,
                    'ponto_montagem': particao.mountpoint,
                    'sistema_arquivos': particao.fstype,
                    'total_gb': round(uso.total / (1024**3), 2),
                    'usado_gb': round(uso.used / (1024**3), 2),
                    'livre_gb': round(uso.free / (1024**3), 2),
                    'percentual_uso': round((uso.used / uso.total) * 100, 1)
                })
            except PermissionError:
                continue
        
        return {
            'discos': discos,
            'timestamp': datetime.now().isoformat()
        }
    
    def obter_processos_top(self, limit=10):
        """Obtém os processos que mais consomem recursos"""
        processos = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processos.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Ordena por uso de CPU
        processos_cpu = sorted(processos, key=lambda x: x['cpu_percent'] or 0, reverse=True)[:limit]
        
        # Ordena por uso de memória
        processos_memoria = sorted(processos, key=lambda x: x['memory_percent'] or 0, reverse=True)[:limit]
        
        return {
            'top_cpu': processos_cpu,
            'top_memoria': processos_memoria,
            'total_processos': len(processos),
            'timestamp': datetime.now().isoformat()
        }
    
    def verificar_alertas(self, info_sistema):
        """Verifica se algum limite foi ultrapassado"""
        alertas = []
        limites = self.config['limites']
        
        # CPU
        if info_sistema['cpu']['percentual_uso'] > limites['cpu_percent']:
            alerta = f"CPU em {info_sistema['cpu']['percentual_uso']:.1f}% (limite: {limites['cpu_percent']}%)"
            if alerta not in self.alertas_enviados:
                alertas.append(alerta)
                self.alertas_enviados.add(alerta)
        
        # Memória
        if info_sistema['memoria']['percentual_uso'] > limites['memoria_percent']:
            alerta = f"Memória em {info_sistema['memoria']['percentual_uso']:.1f}% (limite: {limites['memoria_percent']}%)"
            if alerta not in self.alertas_enviados:
                alertas.append(alerta)
                self.alertas_enviados.add(alerta)
        
        # Disco
        for disco in info_sistema['disco']['discos']:
            if disco['percentual_uso'] > limites['disco_percent']:
                alerta = f"Disco {disco['dispositivo']} em {disco['percentual_uso']:.1f}% (limite: {limites['disco_percent']}%)"
                if alerta not in self.alertas_enviados:
                    alertas.append(alerta)
                    self.alertas_enviados.add(alerta)
        
        # Temperatura CPU
        if 'temperatura' in info_sistema['cpu'] and info_sistema['cpu']['temperatura'] > limites['temperatura_cpu']:
            alerta = f"Temperatura CPU em {info_sistema['cpu']['temperatura']:.1f}°C (limite: {limites['temperatura_cpu']}°C)"
            if alerta not in self.alertas_enviados:
                alertas.append(alerta)
                self.alertas_enviados.add(alerta)
        
        return alertas
    
    def salvar_historico(self, info_sistema):
        """Salva histórico em arquivo JSON"""
        if not self.config['salvar_historico']:
            return
        
        arquivo_historico = Path(self.config['arquivo_historico'])
        arquivo_historico.parent.mkdir(parents=True, exist_ok=True)
        
        # Carrega histórico existente ou cria novo
        historico = []
        if arquivo_historico.exists():
            try:
                with open(arquivo_historico, 'r', encoding='utf-8') as f:
                    historico = json.load(f)
            except:
                pass
        
        # Adiciona nova entrada
        historico.append(info_sistema)
        
        # Mantém apenas últimas 1000 entradas
        if len(historico) > 1000:
            historico = historico[-1000:]
        
        # Salva histórico
        try:
            with open(arquivo_historico, 'w', encoding='utf-8') as f:
                json.dump(historico, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Erro ao salvar histórico: {e}")
    
    def executar_monitoramento(self, duracao_minutos=None):
        """Executa o monitoramento do sistema"""
        logging.info("Iniciando monitoramento do sistema...")
        
        inicio = time.time()
        
        try:
            while True:
                # Coleta informações do sistema
                info_sistema = {
                    'cpu': self.obter_info_cpu(),
                    'memoria': self.obter_info_memoria(),
                    'disco': self.obter_info_disco(),
                    'processos': self.obter_processos_top(),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Verifica alertas
                alertas = self.verificar_alertas(info_sistema)
                if alertas:
                    for alerta in alertas:
                        logging.warning(f"ALERTA: {alerta}")
                
                # Salva histórico
                self.salvar_historico(info_sistema)
                
                # Log de status
                logging.info(f"CPU: {info_sistema['cpu']['percentual_uso']:.1f}% | "
                           f"MEM: {info_sistema['memoria']['percentual_uso']:.1f}% | "
                           f"Processos: {info_sistema['processos']['total_processos']}")
                
                # Verifica duração
                if duracao_minutos:
                    tempo_decorrido = (time.time() - inicio) / 60
                    if tempo_decorrido >= duracao_minutos:
                        break
                
                # Aguarda próximo ciclo
                time.sleep(self.config['intervalo_monitoramento'])
                
        except KeyboardInterrupt:
            logging.info("Monitoramento interrompido pelo usuário")
        except Exception as e:
            logging.error(f"Erro durante monitoramento: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor de Sistema Automático')
    parser.add_argument('--config', help='Arquivo de configuração JSON')
    parser.add_argument('--duracao', type=int, help='Duração em minutos (padrão: infinito)')
    
    args = parser.parse_args()
    
    monitor = MonitorSistema(args.config)
    monitor.executar_monitoramento(args.duracao)

if __name__ == "__main__":
    main()