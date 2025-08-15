#!/usr/bin/env python3
"""
Aplicação Web para Detecção de Câncer de Pele
Flask app com captura de câmera e upload de imagens
"""

import os
import sys
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predict import SkinCancerPredictor

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações da aplicação
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
    RESULTS_FOLDER = Path(__file__).parent / 'results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    MODEL_PATH = os.environ.get('MODEL_PATH') or '../models/best_model.pth'

# Criar app Flask
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Criar diretórios necessários
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)

# Inicializar predictor (global)
predictor = None

def init_predictor():
    """Inicializa o predictor de câncer de pele"""
    global predictor
    
    model_path = app.config['MODEL_PATH']
    
    if not Path(model_path).exists():
        logger.error(f"Modelo não encontrado: {model_path}")
        return False
    
    try:
        predictor = SkinCancerPredictor(
            model_path=model_path,
            device='auto',
            image_size=224
        )
        logger.info("Predictor inicializado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Erro ao inicializar predictor: {e}")
        return False

def allowed_file(filename):
    """Verifica se arquivo tem extensão permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(image_path: str) -> bool:
    """Valida se é uma imagem válida"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def process_image(image_path: str) -> Dict[str, Any]:
    """Processa imagem e retorna resultado da predição"""
    global predictor
    
    if predictor is None:
        return {'error': 'Modelo não carregado'}
    
    try:
        # Fazer predição
        result = predictor.predict_single(image_path, return_features=False)
        
        # Adicionar timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        # Salvar resultado
        result_id = str(uuid.uuid4())
        result_file = app.config['RESULTS_FOLDER'] / f'{result_id}.json'
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        result['result_id'] = result_id
        
        return result
        
    except Exception as e:
        logger.error(f"Erro ao processar imagem {image_path}: {e}")
        return {'error': str(e)}

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/capture')
def capture():
    """Página de captura de câmera"""
    return render_template('capture.html')

@app.route('/upload')
def upload():
    """Página de upload de imagem"""
    return render_template('upload.html')

@app.route('/about')
def about():
    """Página sobre o projeto"""
    return render_template('about.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API para upload e análise de imagem"""
    
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Formato não suportado. Use: {", ".join(app.config["ALLOWED_EXTENSIONS"])}'
        }), 400
    
    try:
        # Salvar arquivo
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        
        file.save(str(file_path))
        
        # Validar imagem
        if not validate_image(str(file_path)):
            os.remove(file_path)
            return jsonify({'error': 'Arquivo não é uma imagem válida'}), 400
        
        # Processar imagem
        result = process_image(str(file_path))
        
        if 'error' in result:
            os.remove(file_path)
            return jsonify({'error': result['error']}), 500
        
        # Adicionar informações do arquivo
        result['filename'] = filename
        result['file_size'] = file_path.stat().st_size
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/api/capture', methods=['POST'])
def api_capture():
    """API para análise de imagem capturada da câmera"""
    
    try:
        # Receber dados da imagem em base64
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Dados da imagem não encontrados'}), 400
        
        image_data = data['image']
        
        # Decodificar base64
        import base64
        from io import BytesIO
        
        # Remover prefixo data:image/...;base64,
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decodificar
        image_bytes = base64.b64decode(image_data)
        
        # Criar arquivo temporário
        unique_filename = f"capture_{uuid.uuid4()}.jpg"
        file_path = app.config['UPLOAD_FOLDER'] / unique_filename
        
        # Salvar imagem
        with open(file_path, 'wb') as f:
            f.write(image_bytes)
        
        # Validar imagem
        if not validate_image(str(file_path)):
            os.remove(file_path)
            return jsonify({'error': 'Imagem capturada inválida'}), 400
        
        # Processar imagem
        result = process_image(str(file_path))
        
        if 'error' in result:
            os.remove(file_path)
            return jsonify({'error': result['error']}), 500
        
        # Adicionar informações
        result['filename'] = 'Imagem capturada'
        result['file_size'] = file_path.stat().st_size
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro na captura: {e}")
        return jsonify({'error': 'Erro ao processar imagem capturada'}), 500

@app.route('/api/result/<result_id>')
def api_get_result(result_id):
    """API para obter resultado por ID"""
    
    try:
        result_file = app.config['RESULTS_FOLDER'] / f'{result_id}.json'
        
        if not result_file.exists():
            return jsonify({'error': 'Resultado não encontrado'}), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro ao buscar resultado {result_id}: {e}")
        return jsonify({'error': 'Erro interno do servidor'}), 500

@app.route('/api/health')
def api_health():
    """API de saúde do sistema"""
    
    status = {
        'status': 'ok',
        'predictor_loaded': predictor is not None,
        'timestamp': datetime.now().isoformat()
    }
    
    if predictor:
        status['model_device'] = str(predictor.device)
        status['model_classes'] = len(predictor.class_names)
    
    return jsonify(status)

@app.route('/result/<result_id>')
def result_page(result_id):
    """Página de resultado detalhado"""
    return render_template('result.html', result_id=result_id)

@app.route('/history')
def history():
    """Página de histórico de análises"""
    
    try:
        # Listar arquivos de resultado
        result_files = list(app.config['RESULTS_FOLDER'].glob('*.json'))
        
        # Carregar metadados dos resultados
        results = []
        for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Extrair apenas metadados
                metadata = {
                    'result_id': result_file.stem,
                    'filename': result.get('filename', 'Desconhecido'),
                    'predicted_class': result.get('predicted_class', 'Erro'),
                    'confidence': result.get('confidence', 0),
                    'timestamp': result.get('timestamp', ''),
                    'urgency': result.get('medical_recommendation', {}).get('urgency', 'Desconhecida')
                }
                
                results.append(metadata)
                
            except Exception as e:
                logger.error(f"Erro ao carregar resultado {result_file}: {e}")
                continue
        
        return render_template('history.html', results=results[:50])  # Últimos 50
        
    except Exception as e:
        logger.error(f"Erro ao carregar histórico: {e}")
        flash('Erro ao carregar histórico', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(error):
    """Página de erro 404"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Página não encontrada"), 404

@app.errorhandler(500)
def internal_error(error):
    """Página de erro 500"""
    return render_template('error.html',
                         error_code=500,
                         error_message="Erro interno do servidor"), 500

@app.errorhandler(413)
def file_too_large(error):
    """Erro de arquivo muito grande"""
    return jsonify({'error': 'Arquivo muito grande. Máximo 16MB.'}), 413

# Comandos CLI personalizados
@app.cli.command()
def init_db():
    """Inicializa diretórios necessários"""
    app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)
    app.config['RESULTS_FOLDER'].mkdir(exist_ok=True)
    print("Diretórios inicializados!")

@app.cli.command()
def test_model():
    """Testa carregamento do modelo"""
    success = init_predictor()
    if success:
        print("✅ Modelo carregado com sucesso!")
    else:
        print("❌ Erro ao carregar modelo!")

@app.cli.command()
def clean_uploads():
    """Limpa arquivos de upload antigos"""
    import time
    
    cutoff_time = time.time() - (24 * 60 * 60)  # 24 horas
    cleaned = 0
    
    for file_path in app.config['UPLOAD_FOLDER'].glob('*'):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            cleaned += 1
    
    print(f"🧹 {cleaned} arquivos removidos!")

if __name__ == '__main__':
    # Inicializar predictor
    print("🏥 Inicializando Sistema de Detecção de Câncer de Pele...")
    
    model_loaded = init_predictor()
    
    if not model_loaded:
        print("⚠️ AVISO: Modelo não carregado. Algumas funcionalidades não estarão disponíveis.")
        print("   Certifique-se de que o modelo treinado existe em:", app.config['MODEL_PATH'])
    
    print("\n" + "="*60)
    print("🌐 SERVIDOR WEB INICIADO")
    print("="*60)
    print(f"📱 Acesse: http://localhost:5000")
    print(f"📊 Status: http://localhost:5000/api/health")
    print(f"🏥 Modelo: {'✅ Carregado' if model_loaded else '❌ Não carregado'}")
    print("="*60)
    print("\n⚠️ IMPORTANTE: Este sistema é apenas para fins educacionais.")
    print("   SEMPRE consulte um dermatologista qualificado!")
    print("\n🚀 Pressione Ctrl+C para parar o servidor")
    
    # Executar servidor
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )