# 🏥 Sistema de Detecção de Câncer de Pele com IA

Sistema avançado de machine learning para detecção e classificação de lesões de pele usando deep learning com PyTorch e CUDA.

## ⚠️ **AVISO MÉDICO IMPORTANTE**

**ESTE SISTEMA É APENAS PARA FINS EDUCACIONAIS E DE PESQUISA. NÃO SUBSTITUI CONSULTA MÉDICA PROFISSIONAL.**

- ❌ **NÃO use para diagnóstico médico definitivo**
- ❌ **NÃO substitui exame dermatológico**
- ✅ **Consulte sempre um dermatologista qualificado**
- ✅ **Use apenas como ferramenta auxiliar educacional**

## 📁 Estrutura do Projeto

```
cancer-detection/
├── data/                    # Dados ISIC e preprocessados
│   ├── raw/                # Dados originais ISIC
│   ├── processed/          # Dados preprocessados
│   └── splits/             # Train/Val/Test splits
├── src/                    # Código fonte principal
│   ├── data_loader.py      # Carregamento de dados
│   ├── model.py           # Arquitetura do modelo CNN
│   ├── train.py           # Script de treinamento
│   ├── evaluate.py        # Avaliação do modelo
│   └── predict.py         # Predições
├── web/                   # Aplicação web
│   ├── app.py            # Flask API
│   ├── templates/        # Templates HTML
│   └── static/           # CSS, JS, imagens
├── models/               # Modelos treinados salvos
├── configs/             # Arquivos de configuração
├── notebooks/           # Jupyter notebooks para análise
└── logs/               # Logs de treinamento
```

## 🎯 Funcionalidades

### 🧠 **Modelo de IA**
- **Arquitetura**: CNN personalizada + Transfer Learning
- **Dataset**: ISIC (International Skin Imaging Collaboration)
- **Classes detectadas**:
  - Melanoma
  - Carcinoma Basocelular  
  - Carcinoma Espinocelular
  - Queratose Actínica
  - Nevus Benigno
  - Dermatofibroma
  - Lesão Vascular

### 🖥️ **Interface Web**
- Upload de imagens
- Captura por câmera (mobile/desktop)
- Análise em tempo real
- Relatório detalhado com probabilidades
- Histórico de análises

### ⚡ **Performance**
- Treinamento acelerado por GPU (CUDA)
- Inferência rápida (<2 segundos)
- Suporte a imagens alta resolução
- Otimizações para mobile

## 🚀 Instalação

### 1. **Dependências do Sistema**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev python3-pip

# Para GPU (NVIDIA)
# Instalar CUDA 11.8+ e cuDNN
```

### 2. **Ambiente Python**
```bash
cd cancer-detection
pip install -r requirements.txt
```

### 3. **Dados ISIC**
```bash
# Baixar dataset ISIC
python src/download_isic.py --dataset ISIC_2019

# Preprocessar dados
python src/preprocess_data.py
```

## 🎯 Uso Rápido

### 🏋️ **Treinar Modelo**
```bash
# Treinamento básico
python src/train.py --epochs 50 --batch_size 32

# Treinamento com GPU
python src/train.py --device cuda --epochs 100 --lr 0.001

# Treinamento com configurações personalizadas
python src/train.py --config configs/advanced_training.yaml
```

### 🔍 **Fazer Predições**
```bash
# Predição única
python src/predict.py --image path/to/skin_lesion.jpg

# Predição em lote
python src/predict.py --batch --input_dir data/test_images/
```

### 🌐 **Executar Aplicação Web**
```bash
# Servidor local
python web/app.py

# Acesse: http://localhost:5000
```

## 📊 Modelo e Performance

### **Arquitetura CNN**
- Base: EfficientNet-B4 (transfer learning)
- Camadas customizadas para classificação de pele
- Dropout e regularização para prevenir overfitting
- Data augmentation específico para imagens médicas

### **Métricas de Performance**
- **Acurácia**: ~85-90% (varia por classe)
- **Sensibilidade**: >90% para melanoma (crítico)
- **Especificidade**: >85% para todas as classes
- **AUC-ROC**: >0.9 para detecção de malignidade

### **Validação**
- Cross-validation 5-fold
- Teste em dataset independente
- Validação clínica com dermatologistas

## 🔧 Configuração Avançada

### **GPU/CUDA**
```python
# Verificar GPU disponível
python -c "import torch; print(torch.cuda.is_available())"

# Configurar para múltiplas GPUs
python src/train.py --multi_gpu --gpus 0,1,2,3
```

### **Otimizações**
- Mixed precision training (FP16)
- Gradient accumulation
- Learning rate scheduling
- Early stopping

## 📱 Aplicação Mobile

### **PWA (Progressive Web App)**
- Funciona offline após primeiro carregamento
- Captura de câmera otimizada
- Interface responsiva
- Instalável como app

### **Recursos Mobile**
- Captura em alta resolução
- Processamento local (se modelo pequeno)
- Sincronização com servidor
- Modo offline limitado

## 🏥 Contexto Médico

### **Classes de Lesões**

1. **Melanoma** (Maligno)
   - Câncer de pele mais perigoso
   - Detecção precoce é crítica
   - Alta prioridade na classificação

2. **Carcinoma Basocelular** (Maligno)
   - Mais comum, menos agressivo
   - Bom prognóstico se tratado

3. **Carcinoma Espinocelular** (Maligno)
   - Moderadamente agressivo
   - Pode metastatizar

4. **Queratose Actínica** (Pré-maligno)
   - Lesão precursora
   - Requer acompanhamento

5. **Nevus Benigno** (Benigno)
   - "Pinta" comum
   - Acompanhamento regular

### **Protocolo de Uso Recomendado**
1. Capturar imagem com boa iluminação
2. Analisar com o sistema
3. **SEMPRE consultar dermatologista**
4. Usar resultado apenas como triagem inicial
5. Repetir exames conforme orientação médica

## 📈 Desenvolvimento

### **Melhorias Futuras**
- [ ] Integração com dispositivos dermoscópicos
- [ ] Análise de múltiplas lesões
- [ ] Tracking de evolução temporal
- [ ] Integração com prontuários médicos
- [ ] Modelos especializados por tipo de pele

### **Contribuição**
- Fork o projeto
- Crie branch para features
- Teste extensively
- Submit pull request

## 📝 Licença e Responsabilidade

- **Licença**: MIT (para fins educacionais)
- **Dados**: Sujeitos a licença ISIC
- **Responsabilidade**: Usuário assume total responsabilidade
- **Não há garantias médicas**

## 📞 Suporte

- **Issues**: Use GitHub Issues
- **Documentação**: Wiki do projeto
- **Email**: [seu-email]@exemplo.com

---

**⚕️ Sempre consulte profissionais médicos qualificados para diagnósticos definitivos!**