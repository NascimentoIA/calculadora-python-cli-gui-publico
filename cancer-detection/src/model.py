#!/usr/bin/env python3
"""
Modelo CNN para Detecção de Câncer de Pele
Arquitetura baseada em transfer learning com EfficientNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Dict, Any
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinCancerClassifier(nn.Module):
    """
    Modelo principal para classificação de câncer de pele
    Baseado em EfficientNet com transfer learning
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        model_name: str = 'efficientnet_b4',
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(SkinCancerClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        
        # Carregar backbone pré-treinado
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classificador final
            global_pool='avg'  # Global average pooling
        )
        
        # Obter dimensões das features
        self.feature_dim = self.backbone.num_features
        
        # Congelar backbone se necessário
        if freeze_backbone:
            self.freeze_backbone_layers()
        
        # Cabeçalho de classificação personalizado
        self.classifier = self._build_classifier()
        
        # Inicializar pesos do classificador
        self._init_classifier_weights()
        
        logger.info(f"Modelo criado: {model_name}")
        logger.info(f"Feature dim: {self.feature_dim}")
        logger.info(f"Num classes: {num_classes}")
        logger.info(f"Dropout: {dropout_rate}")
    
    def _build_classifier(self) -> nn.Module:
        """Constrói o cabeçalho de classificação"""
        return nn.Sequential(
            # Primeira camada densa
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            
            # Segunda camada densa
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            
            # Camada final
            nn.Linear(256, self.num_classes)
        )
    
    def _init_classifier_weights(self):
        """Inicializa pesos do classificador"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def freeze_backbone_layers(self):
        """Congela camadas do backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone layers frozen")
    
    def unfreeze_backbone_layers(self):
        """Descongela camadas do backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone layers unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extrair features do backbone
        features = self.backbone(x)
        
        # Classificação
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai features sem classificação"""
        with torch.no_grad():
            features = self.backbone(x)
        return features

class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desbalanceamento de classes
    Especialmente importante para detecção de melanoma (classe rara)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcular cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Calcular probabilidades
        pt = torch.exp(-ce_loss)
        
        # Aplicar focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing para regularização
    Ajuda a prevenir overconfidence do modelo
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Criar labels suaves
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # Calcular loss
        loss = -torch.sum(smooth_targets * log_probs, dim=1)
        
        return loss.mean()

class ModelEMA:
    """
    Exponential Moving Average para os pesos do modelo
    Melhora estabilidade e performance
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Inicializar shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Atualiza EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Aplica shadow weights ao modelo"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restaura weights originais"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def create_model(
    num_classes: int = 7,
    model_name: str = 'efficientnet_b4',
    pretrained: bool = True,
    **kwargs
) -> SkinCancerClassifier:
    """Factory function para criar o modelo"""
    
    model = SkinCancerClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        **kwargs
    )
    
    return model

def calculate_model_size(model: nn.Module) -> Dict[str, Any]:
    """Calcula tamanho e parâmetros do modelo"""
    param_size = 0
    param_sum = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    
    buffer_size = 0
    buffer_sum = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    
    all_size = (param_size + buffer_size) / 1024 / 1024  # MB
    
    return {
        'param_count': param_sum,
        'buffer_count': buffer_sum,
        'model_size_mb': all_size,
        'param_size_mb': param_size / 1024 / 1024,
        'buffer_size_mb': buffer_size / 1024 / 1024
    }

def test_model():
    """Função de teste para o modelo"""
    
    # Configurações de teste
    batch_size = 4
    num_classes = 7
    image_size = 224
    
    # Criar modelo
    model = create_model(
        num_classes=num_classes,
        model_name='efficientnet_b4',
        pretrained=True,
        dropout_rate=0.3
    )
    
    # Calcular tamanho do modelo
    model_info = calculate_model_size(model)
    print(f"Modelo info: {model_info}")
    
    # Teste com entrada dummy
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        features = model.get_features(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Features shape: {features.shape}")
    
    # Teste de losses
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Cross Entropy
    ce_loss = F.cross_entropy(outputs, targets)
    print(f"CE Loss: {ce_loss.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    fl = focal_loss(outputs, targets)
    print(f"Focal Loss: {fl.item():.4f}")
    
    # Label Smoothing
    ls_loss = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
    ls = ls_loss(outputs, targets)
    print(f"Label Smoothing Loss: {ls.item():.4f}")
    
    # Teste EMA
    ema = ModelEMA(model, decay=0.9999)
    ema.update()
    print("EMA test passed")
    
    logger.info("Teste do modelo concluído com sucesso!")

if __name__ == "__main__":
    test_model()