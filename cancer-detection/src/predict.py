#!/usr/bin/env python3
"""
Script de Predição para Detecção de Câncer de Pele
Realiza predições em imagens individuais ou em lote
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Imports locais
from model import create_model

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SkinCancerPredictor:
    """Classe para realizar predições de câncer de pele"""
    
    def __init__(self, model_path: str, device: str = 'auto', image_size: int = 224):
        self.device = self._setup_device(device)
        self.image_size = image_size
        
        # Classes de lesões
        self.class_names = [
            'Melanoma',
            'Carcinoma Basocelular', 
            'Carcinoma Espinocelular',
            'Queratose Actínica',
            'Nevus Benigno',
            'Dermatofibroma',
            'Lesão Vascular'
        ]
        
        # Informações médicas das classes
        self.class_info = {
            'Melanoma': {
                'type': 'Maligno',
                'severity': 'Alto',
                'description': 'Tipo mais perigoso de câncer de pele. Requer atenção médica imediata.',
                'color': '#FF0000'
            },
            'Carcinoma Basocelular': {
                'type': 'Maligno',
                'severity': 'Moderado',
                'description': 'Câncer de pele mais comum. Crescimento lento, bom prognóstico se tratado.',
                'color': '#FF8C00'
            },
            'Carcinoma Espinocelular': {
                'type': 'Maligno',
                'severity': 'Moderado',
                'description': 'Câncer moderadamente agressivo. Pode metastatizar se não tratado.',
                'color': '#FF6347'
            },
            'Queratose Actínica': {
                'type': 'Pré-maligno',
                'severity': 'Baixo',
                'description': 'Lesão precursora que pode evoluir para câncer. Requer acompanhamento.',
                'color': '#FFA500'
            },
            'Nevus Benigno': {
                'type': 'Benigno',
                'severity': 'Muito Baixo',
                'description': 'Pinta comum benigna. Acompanhamento regular recomendado.',
                'color': '#90EE90'
            },
            'Dermatofibroma': {
                'type': 'Benigno',
                'severity': 'Muito Baixo',
                'description': 'Tumor benigno da pele. Geralmente não requer tratamento.',
                'color': '#87CEEB'
            },
            'Lesão Vascular': {
                'type': 'Benigno',
                'severity': 'Muito Baixo',
                'description': 'Lesão vascular benigna. Raramente requer intervenção.',
                'color': '#DDA0DD'
            }
        }
        
        # Carregar modelo
        self.model = self.load_model(model_path)
        
        # Transformações para inferência
        self.transform = self.get_inference_transforms()
        
        logger.info(f"Predictor inicializado no dispositivo: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configura dispositivo"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        device = torch.device(device)
        logger.info(f"Usando dispositivo: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Carrega modelo treinado"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # Criar modelo
        model_config = config.get('model', {})
        model = create_model(
            num_classes=model_config.get('num_classes', 7),
            model_name=model_config.get('architecture', 'efficientnet_b4'),
            pretrained=False  # Não precisamos de pesos pré-treinados para inferência
        )
        
        # Carregar pesos
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Modelo carregado: {model_path}")
        logger.info(f"Acurácia de validação: {checkpoint.get('val_acc', 'N/A'):.2f}%")
        logger.info(f"AUC de validação: {checkpoint.get('val_auc', 'N/A'):.4f}")
        
        return model
    
    def get_inference_transforms(self) -> A.Compose:
        """Transformações para inferência"""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocessa imagem para inferência"""
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converter BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar mantendo aspect ratio
        image = self.resize_with_padding(image, self.image_size)
        
        # Aplicar transformações
        transformed = self.transform(image=image)
        image_tensor = transformed['image']
        
        # Adicionar dimensão de batch
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def resize_with_padding(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Redimensiona imagem mantendo aspect ratio"""
        h, w = image.shape[:2]
        
        # Calcular novo tamanho
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
    
    def predict_single(self, image_path: str, return_features: bool = False) -> Dict[str, Any]:
        """Predição em uma única imagem"""
        # Preprocessar imagem
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inferência
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            # Extrair features se solicitado
            features = None
            if return_features:
                features = self.model.get_features(image_tensor)
                features = features.cpu().numpy().flatten()
        
        # Converter para numpy
        probabilities = probabilities.cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Criar resultado estruturado
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_class_idx': int(predicted_class_idx),
            'confidence': float(confidence),
            'class_info': self.class_info[predicted_class],
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities)
            },
            'medical_recommendation': self.get_medical_recommendation(predicted_class, confidence),
            'features': features.tolist() if features is not None else None
        }
        
        return result
    
    def get_medical_recommendation(self, predicted_class: str, confidence: float) -> Dict[str, Any]:
        """Gera recomendação médica baseada na predição"""
        class_info = self.class_info[predicted_class]
        
        # Determinar urgência baseada na classe e confiança
        if predicted_class == 'Melanoma':
            if confidence > 0.7:
                urgency = "URGENTE"
                recommendation = "Consulte um dermatologista IMEDIATAMENTE. Melanoma requer diagnóstico e tratamento urgentes."
            else:
                urgency = "Alta"
                recommendation = "Consulte um dermatologista o mais rápido possível para avaliação."
        elif predicted_class in ['Carcinoma Basocelular', 'Carcinoma Espinocelular']:
            if confidence > 0.8:
                urgency = "Alta"
                recommendation = "Consulte um dermatologista em breve. Lesão suspeita de malignidade."
            else:
                urgency = "Moderada"
                recommendation = "Agende consulta dermatológica para avaliação da lesão."
        elif predicted_class == 'Queratose Actínica':
            urgency = "Moderada"
            recommendation = "Consulte dermatologista. Lesão pré-maligna requer acompanhamento."
        else:
            urgency = "Baixa"
            recommendation = "Acompanhamento dermatológico de rotina recomendado."
        
        # Ajustar com base na confiança
        if confidence < 0.6:
            urgency = "Moderada"
            recommendation += " NOTA: Confiança baixa na predição - avaliação médica é essencial."
        
        return {
            'urgency': urgency,
            'recommendation': recommendation,
            'confidence_level': 'Alta' if confidence > 0.8 else 'Moderada' if confidence > 0.6 else 'Baixa'
        }
    
    def predict_batch(self, image_paths: List[str], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predição em lote"""
        results = []
        
        logger.info(f"Processando {len(image_paths)} imagens...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path)
                results.append(result)
                
                logger.info(f"[{i+1}/{len(image_paths)}] {Path(image_path).name}: "
                           f"{result['predicted_class']} ({result['confidence']:.3f})")
                
            except Exception as e:
                logger.error(f"Erro ao processar {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # Salvar resultados se solicitado
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Salva resultados em arquivo JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar dados para serialização
        serializable_results = []
        for result in results:
            if 'error' not in result:
                # Remover features numpy para serialização
                result_copy = result.copy()
                if 'features' in result_copy:
                    del result_copy['features']
                serializable_results.append(result_copy)
            else:
                serializable_results.append(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados salvos em: {output_path}")
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Imprime resultado detalhado"""
        if 'error' in result:
            print(f"❌ Erro: {result['error']}")
            return
        
        print("\n" + "="*60)
        print(f"📸 Imagem: {Path(result['image_path']).name}")
        print("="*60)
        
        # Predição principal
        class_info = result['class_info']
        print(f"🏥 Diagnóstico Predito: {result['predicted_class']}")
        print(f"📊 Confiança: {result['confidence']:.1%}")
        print(f"🔬 Tipo: {class_info['type']}")
        print(f"⚠️  Severidade: {class_info['severity']}")
        print(f"📝 Descrição: {class_info['description']}")
        
        # Recomendação médica
        med_rec = result['medical_recommendation']
        urgency_emoji = {
            'URGENTE': '🚨',
            'Alta': '⚠️',
            'Moderada': '⚡',
            'Baixa': '💡'
        }
        
        print(f"\n{urgency_emoji.get(med_rec['urgency'], '❓')} RECOMENDAÇÃO MÉDICA:")
        print(f"Urgência: {med_rec['urgency']}")
        print(f"💊 {med_rec['recommendation']}")
        
        # Todas as probabilidades
        print(f"\n📊 PROBABILIDADES DETALHADAS:")
        sorted_probs = sorted(
            result['all_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for class_name, prob in sorted_probs:
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{class_name:20} {bar} {prob:.1%}")
        
        print("\n⚠️  IMPORTANTE: Esta é apenas uma análise preliminar.")
        print("   SEMPRE consulte um dermatologista qualificado!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Predição de Câncer de Pele')
    parser.add_argument('--model', type=str, required=True,
                       help='Caminho para o modelo treinado (.pth)')
    parser.add_argument('--image', type=str,
                       help='Caminho para imagem única')
    parser.add_argument('--batch', action='store_true',
                       help='Modo de processamento em lote')
    parser.add_argument('--input_dir', type=str,
                       help='Diretório com imagens para processamento em lote')
    parser.add_argument('--output', type=str,
                       help='Arquivo de saída para resultados JSON')
    parser.add_argument('--device', type=str, default='auto',
                       help='Dispositivo (auto, cpu, cuda)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Tamanho da imagem de entrada')
    parser.add_argument('--detailed', action='store_true',
                       help='Mostra resultados detalhados')
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.image and not args.batch:
        parser.error("Especifique --image para imagem única ou --batch para lote")
    
    if args.batch and not args.input_dir:
        parser.error("--input_dir é obrigatório no modo batch")
    
    # Criar predictor
    predictor = SkinCancerPredictor(
        model_path=args.model,
        device=args.device,
        image_size=args.image_size
    )
    
    if args.image:
        # Predição única
        result = predictor.predict_single(args.image, return_features=True)
        
        if args.detailed:
            predictor.print_detailed_result(result)
        else:
            print(f"Imagem: {Path(args.image).name}")
            print(f"Predição: {result['predicted_class']}")
            print(f"Confiança: {result['confidence']:.1%}")
            print(f"Urgência: {result['medical_recommendation']['urgency']}")
        
        if args.output:
            predictor.save_results([result], args.output)
    
    else:
        # Predição em lote
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
        
        # Buscar imagens
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in input_dir.rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            logger.error(f"Nenhuma imagem encontrada em: {input_dir}")
            return
        
        # Processar em lote
        results = predictor.predict_batch(
            image_paths, 
            output_file=args.output
        )
        
        # Estatísticas
        successful_predictions = [r for r in results if 'error' not in r]
        
        if successful_predictions:
            print(f"\n📊 ESTATÍSTICAS GERAIS:")
            print(f"Imagens processadas: {len(successful_predictions)}")
            
            # Contagem por classe
            class_counts = {}
            urgency_counts = {}
            
            for result in successful_predictions:
                pred_class = result['predicted_class']
                urgency = result['medical_recommendation']['urgency']
                
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            print("\n🏥 Distribuição por Diagnóstico:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name}: {count}")
            
            print("\n⚠️ Distribuição por Urgência:")
            for urgency, count in sorted(urgency_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {urgency}: {count}")

if __name__ == "__main__":
    main()