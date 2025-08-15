#!/usr/bin/env python3
"""
Script de Treinamento para Detecção de Câncer de Pele
Treinamento com CUDA, mixed precision, e otimizações avançadas
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import json
import yaml
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Imports locais
from data_loader import ISICDataLoader
from model import create_model, FocalLoss, LabelSmoothingLoss, ModelEMA

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SkinCancerTrainer:
    """Classe principal para treinamento do modelo"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.start_epoch = 0
        
        # Paths
        self.model_dir = Path(config['paths']['model_dir'])
        self.log_dir = Path(config['paths']['log_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        logger.info(f"Dispositivo: {self.device}")
        logger.info(f"Mixed Precision: {self.use_amp}")
    
    def _setup_device(self) -> torch.device:
        """Configura dispositivo (CPU/GPU/Multi-GPU)"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"GPUs disponíveis: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            device = torch.device(f"cuda:{self.config.get('gpu_id', 0)}")
        else:
            logger.warning("CUDA não disponível, usando CPU")
            device = torch.device('cpu')
        
        return device
    
    def setup_model(self) -> nn.Module:
        """Configura o modelo"""
        model_config = self.config['model']
        
        model = create_model(
            num_classes=model_config['num_classes'],
            model_name=model_config['architecture'],
            pretrained=model_config['pretrained'],
            dropout_rate=model_config['dropout_rate'],
            freeze_backbone=model_config.get('freeze_backbone', False)
        )
        
        # Multi-GPU
        if torch.cuda.device_count() > 1 and self.config.get('multi_gpu', False):
            logger.info(f"Usando {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        model = model.to(self.device)
        
        # Carregar checkpoint se existir
        checkpoint_path = self.config.get('resume_from')
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(model, checkpoint_path)
        
        return model
    
    def setup_criterion(self) -> nn.Module:
        """Configura função de loss"""
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type']
        
        if loss_type == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_type == 'focal_loss':
            criterion = FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0)
            )
        elif loss_type == 'label_smoothing':
            criterion = LabelSmoothingLoss(
                num_classes=self.config['model']['num_classes'],
                smoothing=loss_config.get('smoothing', 0.1)
            )
        else:
            raise ValueError(f"Loss type não suportado: {loss_type}")
        
        return criterion.to(self.device)
    
    def setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Configura otimizador"""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type']
        
        if opt_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4),
                nesterov=opt_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Optimizer type não suportado: {opt_type}")
        
        return optimizer
    
    def setup_scheduler(self, optimizer: optim.Optimizer, train_loader) -> Any:
        """Configura learning rate scheduler"""
        sched_config = self.config['training']['scheduler']
        sched_type = sched_config['type']
        
        if sched_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        elif sched_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 5),
                verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(
        self, 
        model: nn.Module, 
        train_loader, 
        criterion: nn.Module, 
        optimizer: optim.Optimizer,
        epoch: int,
        ema: ModelEMA = None
    ) -> Dict[str, float]:
        """Treina uma época"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Atualizar EMA
            if ema:
                ema.update()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Log de progresso
            if batch_idx % 100 == 0:
                logger.info(
                    f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | '
                    f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%'
                )
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate_epoch(
        self, 
        model: nn.Module, 
        val_loader, 
        criterion: nn.Module,
        ema: ModelEMA = None
    ) -> Dict[str, float]:
        """Valida uma época"""
        
        # Aplicar EMA se disponível
        if ema:
            ema.apply_shadow()
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Para métricas detalhadas
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Restaurar pesos originais se EMA foi aplicado
        if ema:
            ema.restore()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        # Calcular AUC (One-vs-Rest para multiclasse)
        try:
            all_probs = np.array(all_probs)
            all_labels_onehot = np.eye(self.config['model']['num_classes'])[all_labels]
            val_auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            logger.warning(f"Erro ao calcular AUC: {e}")
            val_auc = 0.0
        
        return {
            'loss': val_loss, 
            'accuracy': val_acc, 
            'auc': val_auc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        epoch: int, 
        val_acc: float, 
        val_auc: float,
        is_best: bool = False
    ):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_auc': val_auc,
            'config': self.config
        }
        
        # Salvar checkpoint regular
        checkpoint_path = self.model_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Salvar melhor modelo
        if is_best:
            best_path = self.model_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Novo melhor modelo salvo: Acc={val_acc:.2f}%, AUC={val_auc:.4f}")
    
    def load_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Carrega checkpoint"""
        logger.info(f"Carregando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        self.best_val_auc = checkpoint.get('val_auc', 0.0)
        
        logger.info(f"Checkpoint carregado. Começando época: {self.start_epoch}")
    
    def train(self):
        """Função principal de treinamento"""
        logger.info("Iniciando treinamento...")
        
        # Setup dos dados
        data_config = self.config['data']
        data_loader = ISICDataLoader(
            data_dir=data_config['data_dir'],
            image_size=data_config['image_size'],
            batch_size=data_config['batch_size']
        )
        
        train_split, val_split, test_split = data_loader.create_data_splits(
            test_size=data_config.get('test_size', 0.2),
            val_size=data_config.get('val_size', 0.1)
        )
        
        train_loader, val_loader, _ = data_loader.create_dataloaders(
            train_split, val_split, test_split,
            num_workers=data_config.get('num_workers', 4)
        )
        
        # Setup do modelo e treinamento
        model = self.setup_model()
        criterion = self.setup_criterion()
        optimizer = self.setup_optimizer(model)
        scheduler = self.setup_scheduler(optimizer, train_loader)
        
        # EMA
        ema = None
        if self.config['training'].get('use_ema', False):
            ema = ModelEMA(model, decay=self.config['training'].get('ema_decay', 0.9999))
        
        # Loop de treinamento
        epochs = self.config['training']['epochs']
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            # Treinar
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch, ema)
            
            # Validar
            val_metrics = self.validate_epoch(model, val_loader, criterion, ema)
            
            # Scheduler step
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_metrics['accuracy'])
                else:
                    scheduler.step()
            
            # Log de métricas
            epoch_time = time.time() - start_time
            logger.info(
                f'Época {epoch}/{epochs-1} ({epoch_time:.1f}s) - '
                f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}% | '
                f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%, '
                f'Val AUC: {val_metrics["auc"]:.4f}'
            )
            
            # TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('AUC/Val', val_metrics['auc'], epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Salvar checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.best_val_auc = val_metrics['auc']
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(
                    model, optimizer, epoch, 
                    val_metrics['accuracy'], val_metrics['auc'], is_best
                )
        
        logger.info("Treinamento concluído!")
        logger.info(f"Melhor Val Acc: {self.best_val_acc:.2f}%")
        logger.info(f"Melhor Val AUC: {self.best_val_auc:.4f}")
        
        self.writer.close()

def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega arquivo de configuração"""
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Formato de config não suportado. Use .yaml ou .json")
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Treinamento de Detecção de Câncer de Pele')
    parser.add_argument('--config', type=str, default='../configs/train_config.yaml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                       help='Diretório dos dados')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Dispositivo (auto, cpu, cuda:0, etc.)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Caminho para checkpoint para continuar treinamento')
    
    args = parser.parse_args()
    
    # Carregar ou criar configuração
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        # Configuração padrão
        config = {
            'data': {
                'data_dir': args.data_dir,
                'image_size': 224,
                'batch_size': args.batch_size,
                'num_workers': 4
            },
            'model': {
                'num_classes': 7,
                'architecture': 'efficientnet_b4',
                'pretrained': True,
                'dropout_rate': 0.3
            },
            'training': {
                'epochs': args.epochs,
                'optimizer': {'type': 'adamw', 'lr': args.lr, 'weight_decay': 1e-4},
                'scheduler': {'type': 'cosine', 'min_lr': 1e-6},
                'loss': {'type': 'focal_loss', 'alpha': 1.0, 'gamma': 2.0},
                'use_ema': True,
                'ema_decay': 0.9999
            },
            'paths': {
                'model_dir': '../models',
                'log_dir': '../logs/tensorboard'
            },
            'use_amp': True,
            'multi_gpu': False
        }
    
    # Sobrescrever com argumentos da linha de comando
    if args.resume:
        config['resume_from'] = args.resume
    
    # Criar trainer e iniciar treinamento
    trainer = SkinCancerTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()