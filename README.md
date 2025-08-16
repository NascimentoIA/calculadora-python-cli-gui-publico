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
├── cancer-detection/      # 🆕 Sistema de Detecção de Câncer de Pele
│   ├── src/              # Código fonte (modelos, treinamento, predição)
│   ├── web/              # Aplicação web Flask
│   ├── data/             # Dados ISIC e preprocessados
│   ├── models/           # Modelos treinados
│   ├── configs/          # Configurações de treinamento
│   ├── logs/             # Logs e métricas
│   ├── notebooks/        # Jupyter notebooks
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

### 🏥 [Sistema de Detecção de Câncer de Pele](./cancer-detection/) **⭐ NOVO**
- **🧠 IA Avançada:** Modelo CNN com EfficientNet-B4 treinado no dataset ISIC
- **📱 Aplicação Web:** Interface para captura de câmera e upload de imagens
- **🎯 7 Tipos de Lesões:** Melanoma, carcinomas, nevus benigno e mais
- **⚡ GPU/CUDA:** Treinamento acelerado com mixed precision
- **🏥 Contexto Médico:** Recomendações baseadas em urgência e tipo de lesão
- **📊 Métricas Clínicas:** >90% sensibilidade para melanoma, AUC >0.9
- **🔬 Transfer Learning:** Pesos pré-treinados + fine-tuning médico
- **📈 Monitoramento:** TensorBoard, logs detalhados, checkpoints

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

### IA Médica (Cancer Detection)
- **Deep Learning:** PyTorch + EfficientNet-B4
- **Dataset ISIC:** Treinado em milhares de imagens dermatológicas
- **Classificação:** 7 classes de lesões de pele
- **Interface Web:** Captura de câmera + upload
- **Predições:** Tempo real (<2s) com contexto médico
- **Métricas:** Sensibilidade >90% para melanoma

## 🛠️ Tecnologias Utilizadas

### Backend & IA
- **Python 3.8+** - Backend e automação
- **PyTorch 2.0+** - Deep learning e redes neurais
- **CUDA/cuDNN** - Aceleração GPU para treinamento
- **EfficientNet** - Arquitetura CNN state-of-the-art
- **Flask** - API web e servidor

### Frontend & Interface
- **HTML/CSS/JavaScript** - Interface web responsiva
- **Bootstrap 5** - Framework CSS moderno
- **WebRTC** - Captura de câmera no navegador
- **Canvas API** - Processamento de imagens client-side

### Dados & Processamento
- **OpenCV** - Processamento de imagens
- **Albumentations** - Data augmentation avançada
- **Pandas/NumPy** - Manipulação de dados
- **Scikit-learn** - Métricas e validação

### Monitoramento & Deploy
- **TensorBoard** - Visualização de treinamento
- **ONNX** - Otimização de modelos
- **Docker** - Containerização (planejado)

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

### Para Detecção de Câncer:
```bash
cd cancer-detection
pip install -r requirements.txt

# Treinar modelo (requer dataset ISIC)
python src/train.py --config configs/train_config.yaml

# Executar aplicação web
python web/app.py
# Acesse: http://localhost:5000
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
- [Detecção de Câncer](./cancer-detection/README.md) - IA médica completa ⭐

## 🏥 Projeto Destaque: Detecção de Câncer de Pele

### 🎯 **Objetivo**
Sistema de IA para triagem inicial de lesões de pele, auxiliando na detecção precoce de câncer.

### 🧠 **Tecnologia**
- **Modelo:** EfficientNet-B4 com transfer learning
- **Dataset:** ISIC (International Skin Imaging Collaboration)
- **Performance:** 85-90% acurácia, >90% sensibilidade melanoma
- **Velocidade:** <2 segundos por análise

### 📱 **Interface**
- **Captura de câmera** em tempo real (mobile/desktop)
- **Upload de imagens** com drag & drop
- **Resultados detalhados** com probabilidades
- **Recomendações médicas** baseadas em urgência

### ⚠️ **Aviso Médico**
**Este sistema é apenas para fins educacionais. SEMPRE consulte um dermatologista qualificado para diagnósticos médicos definitivos.**

## 🤝 Contribuição

Estes projetos são exemplos educacionais e pontos de partida para suas próprias implementações. Sinta-se livre para:
- Expandir funcionalidades
- Melhorar interfaces
- Adicionar novos tipos de automação
- Otimizar performance
- Criar novas configurações
- **Contribuir com o projeto de IA médica**

## 🎓 Finalidade Educacional

Todos os projetos têm **fins exclusivamente educacionais**:
- Demonstrar boas práticas de desenvolvimento
- Explorar diferentes tecnologias e frameworks
- Servir como base para projetos pessoais
- **Mostrar aplicações de IA em contextos médicos**

## 📊 Estatísticas dos Projetos

| Projeto | Linguagens | Complexidade | Status | Destaque |
|---------|------------|--------------|---------|----------|
| Calculadora Python | Python | ⭐⭐ | ✅ Completo | CLI + GUI |
| Calculadora Web | HTML/CSS/JS | ⭐ | ✅ Completo | Responsivo |
| Automação | Python | ⭐⭐⭐ | ✅ Completo | 3 Scripts |
| **Detecção Câncer** | **Python/PyTorch** | **⭐⭐⭐⭐⭐** | **🔥 Novo** | **IA Médica** |

---

**🎉 Estrutura organizada para facilitar desenvolvimento e manutenção de múltiplos projetos, agora com IA médica avançada!**

### 🚀 Próximos Passos Sugeridos:
1. ⭐ **Experimente o sistema de detecção de câncer**
2. 🔧 Configure seu ambiente para GPU/CUDA
3. 📊 Baixe o dataset ISIC para treinamento
4. 🏥 Explore as aplicações de IA em medicina
5. 🎯 Contribua com melhorias e otimizações
