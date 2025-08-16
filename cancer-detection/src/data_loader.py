#!/usr/bin/env python3
"""
Data Loader para Dataset ISIC - Detecção de Câncer de Pele
Carregamento, preprocessamento e augmentação de dados
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, Dict, List, Optional

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ISICDataset(Dataset):
    """Dataset personalizado para dados ISIC"""
    
    def __init__(
        self, 
        image_paths: List[str],
        labels: List[str],
        transform=None,
        image_size: int = 224,
        is_training: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training
        
        # Encoder para labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Classes de lesões de pele
        self.class_names = [
            'melanoma',
            'basal_cell_carcinoma', 
            'squamous_cell_carcinoma',
            'actinic_keratosis',
            'benign_nevus',
            'dermatofibroma',
            'vascular_lesion'
        ]
        
        logger.info(f"Dataset criado com {len(self.image_paths)} imagens")
        logger.info(f"Classes: {self.label_encoder.classes_}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Carregar imagem
        image_path = self.image_paths[idx]
        image = self.load_image(image_path)
        
        # Label
        label = self.encoded_labels[idx]
        
        # Aplicar transformações
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # PyTorch transforms
                image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': image_path
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Carrega e preprocessa imagem"""
        try:
            # Carregar com OpenCV (BGR)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            # Converter BGR para RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar mantendo aspect ratio
            image = self.resize_with_padding(image, self.image_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {image_path}: {e}")
            # Retorna imagem preta como fallback
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Redimensiona imagem mantendo aspect ratio com padding"""
        h, w = image.shape[:2]
        
        # Calcular novo tamanho mantendo aspect ratio
        if h > w:
            new_h = target_size
            new_w = int(w * target_size / h)
        else:
            new_w = target_size
            new_h = int(h * target_size / w)
        
        # Redimensionar
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Criar imagem com padding
        padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calcular posição central
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        # Colocar imagem no centro
        padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image
        
        return padded_image

class ISICDataLoader:
    """Classe principal para gerenciar dados ISIC"""
    
    def __init__(self, data_dir: str, image_size: int = 224, batch_size: int = 32):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Caminhos
        self.images_dir = self.data_dir / "images"
        self.labels_file = self.data_dir / "labels.csv"
        
        # Verificar se dados existem
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {self.images_dir}")
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Arquivo de labels não encontrado: {self.labels_file}")
    
    def get_transforms(self, is_training: bool = True) -> A.Compose:
        """Retorna transformações de augmentação"""
        
        if is_training:
            # Augmentações para treinamento (específicas para imagens médicas)
            return A.Compose([
                # Augmentações geométricas suaves
                A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.1, 
                    rotate_limit=15, 
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                
                # Augmentações de cor e contraste (importantes para lesões)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=20, 
                    val_shift_limit=10, 
                    p=0.3
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                
                # Ruído e blur leves
                A.GaussNoise(var_limit=(0, 25), p=0.2),
                A.GaussianBlur(blur_limit=(1, 3), p=0.2),
                
                # Normalização
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],   # ImageNet stds
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            # Apenas normalização para validação/teste
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
    
    def load_data(self) -> Tuple[List[str], List[str]]:
        """Carrega paths das imagens e labels"""
        # Carregar CSV com labels
        df = pd.read_csv(self.labels_file)
        
        # Verificar colunas necessárias
        required_cols = ['image_name', 'diagnosis']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Coluna '{col}' não encontrada no CSV")
        
        image_paths = []
        labels = []
        
        for _, row in df.iterrows():
            image_name = row['image_name']
            diagnosis = row['diagnosis']
            
            # Construir caminho completo da imagem
            image_path = self.images_dir / f"{image_name}.jpg"
            
            # Verificar se imagem existe
            if image_path.exists():
                image_paths.append(str(image_path))
                labels.append(diagnosis)
            else:
                logger.warning(f"Imagem não encontrada: {image_path}")
        
        logger.info(f"Carregadas {len(image_paths)} imagens válidas")
        
        # Estatísticas das classes
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logger.info(f"Classe '{label}': {count} imagens")
        
        return image_paths, labels
    
    def create_data_splits(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[Dict, Dict, Dict]:
        """Cria splits de treino, validação e teste"""
        
        image_paths, labels = self.load_data()
        
        # Split inicial: treino+val vs teste
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels  # Manter proporção das classes
        )
        
        # Split treino vs validação
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size/(1-test_size),  # Ajustar proporção
            random_state=random_state,
            stratify=train_val_labels
        )
        
        # Criar dicionários com os splits
        train_split = {'paths': train_paths, 'labels': train_labels}
        val_split = {'paths': val_paths, 'labels': val_labels}
        test_split = {'paths': test_paths, 'labels': test_labels}
        
        logger.info(f"Split criado - Treino: {len(train_paths)}, "
                   f"Validação: {len(val_paths)}, Teste: {len(test_paths)}")
        
        return train_split, val_split, test_split
    
    def create_dataloaders(
        self, 
        train_split: Dict, 
        val_split: Dict, 
        test_split: Dict,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Cria DataLoaders para treino, validação e teste"""
        
        # Transformações
        train_transform = self.get_transforms(is_training=True)
        val_transform = self.get_transforms(is_training=False)
        
        # Datasets
        train_dataset = ISICDataset(
            train_split['paths'], 
            train_split['labels'], 
            transform=train_transform,
            image_size=self.image_size,
            is_training=True
        )
        
        val_dataset = ISICDataset(
            val_split['paths'], 
            val_split['labels'], 
            transform=val_transform,
            image_size=self.image_size,
            is_training=False
        )
        
        test_dataset = ISICDataset(
            test_split['paths'], 
            test_split['labels'], 
            transform=val_transform,
            image_size=self.image_size,
            is_training=False
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

def test_data_loader():
    """Função de teste para o data loader"""
    
    # Configurações de teste
    data_dir = "data/processed"
    batch_size = 8
    image_size = 224
    
    try:
        # Criar data loader
        data_loader = ISICDataLoader(
            data_dir=data_dir,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Criar splits
        train_split, val_split, test_split = data_loader.create_data_splits()
        
        # Criar dataloaders
        train_loader, val_loader, test_loader = data_loader.create_dataloaders(
            train_split, val_split, test_split
        )
        
        # Testar um batch
        for batch in train_loader:
            images = batch['image']
            labels = batch['label']
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            break
            
        logger.info("Teste do data loader concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro no teste do data loader: {e}")

if __name__ == "__main__":
    test_data_loader()