# Projetos de Desenvolvimento

Este repositório contém diferentes projetos organizados por categoria e funcionalidade.

## 📁 Estrutura do Repositório

```
/
├── calculadora-python/     # Implementação em Python (CLI + GUI)
│   ├── calculator_core.py
│   ├── cli_calculator.py
│   ├── gui_calculator.py
│   ├── main.py
│   └── README.md
├── web-calculator/         # Implementação Web (HTML/CSS/JS)
│   ├── index.html
│   └── README.md
├── automacao/             # Scripts de Automação
│   ├── scripts/           # Scripts Python de automação
│   ├── config/           # Arquivos de configuração
│   ├── logs/             # Logs de execução
│   ├── docs/             # Documentação
│   ├── tests/            # Testes
│   ├── requirements.txt  # Dependências
│   └── README.md
└── README.md              # Este arquivo
```

## 🚀 Projetos Disponíveis

### 🐍 [Calculadora Python](./calculadora-python/)
- Interface CLI interativa e por expressão
- Interface GUI com Tkinter
- Operações avançadas: potência, raiz quadrada, parênteses
- Avaliação segura sem `eval()`

### 🌐 [Calculadora Web](./web-calculator/)
- Interface web moderna e responsiva
- Funcionalidades equivalentes à versão Python
- Executa diretamente no navegador

### 🤖 [Automação](./automacao/)
- **Organização de Arquivos:** Classifica arquivos por extensão automaticamente
- **Monitor de Sistema:** Monitora CPU, memória, disco e processos
- **Web Scraping:** Coleta dados de sites de forma automatizada
- **Configurável:** Todos os scripts usam arquivos de configuração JSON
- **Logging Completo:** Registra todas as atividades para debugging

## 💡 Funcionalidades Gerais

### Calculadoras
- Operações básicas: `+`, `-`, `*`, `/`
- Potência: `**` ou `^`
- Raiz quadrada: `sqrt(x)`
- Suporte a parênteses para precedência
- Avaliação matemática segura

### Automação
- **Arquivos:** Organização automática, backup, limpeza
- **Sistema:** Monitoramento de recursos e alertas
- **Web:** Scraping de notícias, produtos, dados genéricos
- **Configuração:** Flexível via arquivos JSON/YAML
- **Agendamento:** Integração com cron/schedule

## 🛠️ Tecnologias Utilizadas

- **Python 3.8+** - Backend e automação
- **HTML/CSS/JavaScript** - Interface web
- **Tkinter** - Interface gráfica desktop
- **BeautifulSoup** - Web scraping
- **Psutil** - Monitoramento de sistema
- **Requests** - HTTP requests
- **JSON/YAML** - Configuração

## 📦 Instalação Rápida

### Para Calculadoras:
```bash
# Não requer instalação adicional, usa bibliotecas padrão do Python
cd calculadora-python
python main.py
```

### Para Automação:
```bash
cd automacao
pip install -r requirements.txt
python scripts/organizador_arquivos.py --help
```

## 🎯 Como começar

1. **Escolha o projeto** que mais te interessa
2. **Navegue até a pasta** do projeto
3. **Leia o README específico** para instruções detalhadas
4. **Execute os exemplos** para familiarizar-se com as funcionalidades
5. **Personalize as configurações** conforme suas necessidades

## 📚 Documentação

Cada projeto possui sua própria documentação detalhada:
- [Calculadora Python](./calculadora-python/README.md) - Múltiplas interfaces
- [Calculadora Web](./web-calculator/README.md) - Interface responsiva
- [Automação](./automacao/README.md) - Scripts e configurações

## 🤝 Contribuição

Estes projetos são exemplos educacionais e pontos de partida para suas próprias implementações. Sinta-se livre para:
- Expandir funcionalidades
- Melhorar interfaces
- Adicionar novos tipos de automação
- Otimizar performance
- Criar novas configurações

---
**Estrutura organizada para facilitar desenvolvimento e manutenção de múltiplos projetos! 🎉**
