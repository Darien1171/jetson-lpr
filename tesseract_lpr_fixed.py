"""
SISTEMA LPR CON TESSERACT - Reemplazo completo de EasyOCR
========================================================
Sistema optimizado que reemplaza EasyOCR con Tesseract
Sin dependencias GPU/PyTorch - Solo CPU
CON AUTO-LIMPIEZA Y PREVENCIÓN TOTAL DE BLOQUEO EN 11 DETECCIONES
SIN LOGGING (para debug del problema de bloqueo)

Características:
- Tesseract OCR optimizado para placas
- Preprocesamiento específico para LPR
- Cache inteligente con auto-limpieza
- Timeout estricto
- Configuración ultra-rápida
- MÚLTIPLES LÍNEAS DE DEFENSA ANTI-BLOQUEO

Autor: Sistema LPR con Tesseract
Fecha: 2025-06-30
Versión: 3.1 (Sin Logging)
"""

import cv2
import numpy as np
import time
import re
import subprocess
import threading
import queue
import hashlib
from datetime import datetime
from pathlib import Path
import json
import os

class TesseractLPR:
    """Sistema LPR usando Tesseract OCR con Auto-Limpieza y Prevención de Bloqueo"""
    
    def __init__(self, config_path="config/ptz_config.json"):
        self.config_path = Path(config_path)
        self.load_config()
        
        # Contadores y stats
        self.running = False
        self.capture_frame_count = 0
        self.display_frame_count = 0
        self.ai_processed_frames = 0
        self.detections_count = 0
        self.start_time = None
        
        # NUEVO: Contador para limpieza automática (ultra-agresivo)
        self.auto_cleanup_counter = 0
        self.auto_cleanup_every = 2  # Limpiar cada 2 detecciones (MÁS agresivo)
        
        # Threading
        self.capture_queue = queue.Queue(maxsize=2)
        self.display_queue = queue.Queue(maxsize=2)
        self.ai_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Cache OCR
        self.ocr_cache = {}
        self.detection_cooldown = {}
        self.recent_detections = []
        self.ocr_times = []
        
        # Verificar e instalar Tesseract
        self.setup_tesseract()
        
        # Configurar YOLO
        self.setup_yolo()
        
        # Configurar red
        self.setup_network()
        
        print("✅ Sistema LPR con Tesseract y Auto-Limpieza inicializado")
    
    def load_config(self):
        """Cargar configuración"""
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
            print(f"✅ Configuración cargada desde {self.config_path}")
        except FileNotFoundError:
            loaded_config = {}
        
        # Configuración específica para Tesseract
        self.config = {
            "camera": {
                "ip": "192.168.1.101",
                "rtsp_url": "rtsp://admin:admin@192.168.1.101/cam/realmonitor?channel=1&subtype=1"
            },
            "tesseract_optimization": {
                "capture_target_fps": 15,
                "display_target_fps": 12,
                "ai_process_every": 4,        # Menos frecuente
                "display_scale": 0.3,
                "ocr_timeout": 0.8,           # 800ms timeout
                "psm_mode": "8",              # Single word
                "oem_mode": "3",              # Default engine
                "whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "preprocessing_enabled": True,
                "cache_enabled": True
            },
            "processing": {
                "confidence_threshold": 0.35,
                "plate_confidence_min": 0.3,
                "detection_cooldown_sec": 10.0,  # Aumentado a 10s para placas estáticas
                "max_detections": 3,
                "roi_min_width": 40,
                "roi_min_height": 15
            },
            "output": {
                "save_results": True,
                "show_video": True,
                "window_title": "LPR con Tesseract 🔤"
            }
        }
        
        # Fusionar configuración cargada
        if loaded_config:
            for section, values in loaded_config.items():
                if section in self.config and isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
        
        self.save_config()
    
    def save_config(self):
        """Guardar configuración"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def setup_tesseract(self):
        """Verificar e instalar Tesseract"""
        print("🔤 Configurando Tesseract OCR...")
        
        # Verificar si Tesseract está instalado
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_info = result.stdout.split('\n')[0]
                print(f"✅ Tesseract ya instalado: {version_info}")
                self.tesseract_available = True
            else:
                print("❌ Tesseract no funciona correctamente")
                self.tesseract_available = False
        except FileNotFoundError:
            print("📦 Tesseract no encontrado - instalando...")
            self.install_tesseract()
        except subprocess.TimeoutExpired:
            print("⚠️ Tesseract timeout - puede estar instalado pero lento")
            self.tesseract_available = True
        
        if self.tesseract_available:
            self.test_tesseract()
    
    def install_tesseract(self):
        """Instalar Tesseract y pytesseract"""
        try:
            print("📦 Instalando Tesseract...")
            
            # Actualizar package list
            print("   Actualizando package list...")
            subprocess.run(['sudo', 'apt', 'update'], check=True, timeout=60)
            
            # Instalar Tesseract
            print("   Instalando tesseract-ocr...")
            subprocess.run(['sudo', 'apt', 'install', '-y', 'tesseract-ocr'], 
                          check=True, timeout=120)
            
            # Instalar idioma inglés (por si acaso)
            print("   Instalando tesseract-ocr-eng...")
            subprocess.run(['sudo', 'apt', 'install', '-y', 'tesseract-ocr-eng'], 
                          check=True, timeout=60)
            
            # Verificar instalación
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Tesseract instalado correctamente")
                self.tesseract_available = True
            else:
                print("❌ Error verificando instalación de Tesseract")
                self.tesseract_available = False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando Tesseract: {e}")
            self.tesseract_available = False
        except subprocess.TimeoutExpired:
            print("⚠️ Timeout instalando Tesseract")
            self.tesseract_available = False
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            self.tesseract_available = False
    
    def test_tesseract(self):
        """Test rápido de Tesseract"""
        try:
            print("🧪 Testing Tesseract...")
            
            # Crear imagen de test
            test_img = np.ones((50, 150, 3), dtype=np.uint8) * 255
            cv2.putText(test_img, "ABC123", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            
            # Test con configuración optimizada
            start_time = time.time()
            result = self.tesseract_ocr_optimized(test_img)
            test_time = time.time() - start_time
            
            print(f"   ⏱️ Tiempo test: {test_time:.3f}s")
            print(f"   📝 Resultado: {result}")
            
            if test_time < 1.0:
                print("✅ Tesseract funcionando rápido")
            elif test_time < 2.0:
                print("⚠️ Tesseract funcionando pero lento")
            else:
                print("🐌 Tesseract muy lento - verificar configuración")
                
        except Exception as e:
            print(f"❌ Error en test de Tesseract: {e}")
    
    def setup_yolo(self):
        """Configurar YOLO"""
        try:
            from ultralytics import YOLO
            
            # Buscar modelo
            model_files = ["license_plate_detector.pt", "yolo11n.pt", "yolov8n.pt"]
            model_path = None
            
            for model_file in model_files:
                if Path(model_file).exists():
                    model_path = model_file
                    break
            
            if not model_path:
                print("❌ No se encontró modelo YOLO")
                raise FileNotFoundError("Modelo YOLO no encontrado")
            
            print(f"📦 Cargando YOLO: {model_path}")
            self.yolo_model = YOLO(model_path)
            
            # Warm-up
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in range(2):
                self.yolo_model(dummy_frame, verbose=False)
            
            print("✅ YOLO configurado correctamente")
            
        except Exception as e:
            print(f"❌ Error configurando YOLO: {e}")
            raise
    
    def setup_network(self):
        """Configurar red PTZ"""
        try:
            commands = [
                "sudo ip addr add 192.168.1.100/24 dev enP8p1s0 2>/dev/null || true",
                "sudo ethtool -s enP8p1s0 speed 100 duplex full autoneg off 2>/dev/null || true"
            ]
            
            for cmd in commands:
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            
            print("✅ Red PTZ configurada")
            
        except Exception as e:
            print(f"⚠️ Error configuración de red: {e}")
    
    def preprocess_roi_for_tesseract(self, roi):
        """Preprocesamiento específico para Tesseract"""
        if not self.config["tesseract_optimization"]["preprocessing_enabled"]:
            return roi
        
        try:
            if roi.size == 0:
                return None
            
            h, w = roi.shape[:2]
            
            # 1. Validar tamaño mínimo
            min_w = self.config["processing"]["roi_min_width"]
            min_h = self.config["processing"]["roi_min_height"]
            
            if h < min_h or w < min_w:
                return None
            
            # 2. Resize a tamaño óptimo para Tesseract (altura 48-64px)
            target_height = 56
            if h != target_height:
                scale = target_height / h
                new_w = int(w * scale)
                roi = cv2.resize(roi, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            
            # 3. Conversión a escala de grises
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # 4. Mejora de contraste adaptativa
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 5. Filtrado de ruido
            denoised = cv2.medianBlur(enhanced, 3)
            
            # 6. Binarización
            binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 7. Operaciones morfológicas para limpiar
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return roi
    
    def tesseract_ocr_optimized(self, roi):
        """OCR optimizado con Tesseract"""
        if not self.tesseract_available:
            return []
        
        try:
            # Preprocesamiento
            processed_roi = self.preprocess_roi_for_tesseract(roi)
            if processed_roi is None:
                return []
            
            # Guardar imagen temporal
            temp_path = '/tmp/tesseract_ocr_temp.png'
            cv2.imwrite(temp_path, processed_roi)
            
            # Configurar comando Tesseract
            config = self.config["tesseract_optimization"]
            
            cmd = [
                'tesseract', temp_path, 'stdout',
                '--psm', config["psm_mode"],
                '--oem', config["oem_mode"],
                '-c', f'tessedit_char_whitelist={config["whitelist"]}',
                '-c', 'tessedit_do_invert=0'
            ]
            
            # Ejecutar con timeout
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=config["ocr_timeout"])
            
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
            
            if result.returncode == 0:
                # Procesar resultado
                text = result.stdout.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                
                # Limpiar caracteres no deseados
                cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                
                if len(cleaned_text) >= 3:
                    return [{
                        'text': cleaned_text,
                        'confidence': 0.85  # Tesseract asume buena confianza
                    }]
            
            return []
            
        except subprocess.TimeoutExpired:
            print("⚠️ Tesseract OCR timeout")
            return []
        except Exception as e:
            print(f"Error en Tesseract OCR: {e}")
            return []
    
    def get_roi_hash(self, roi):
        """Hash para cache"""
        sample = roi[::3, ::3]  # Sample cada 3 píxeles
        return hashlib.md5(sample.tobytes()).hexdigest()[:10]
    
    def read_plate_text_tesseract(self, roi):
        """Método principal de OCR con cache"""
        start_time = time.time()
        
        # Cache check
        if self.config["tesseract_optimization"]["cache_enabled"]:
            roi_hash = self.get_roi_hash(roi)
            if roi_hash in self.ocr_cache:
                cached = self.ocr_cache[roi_hash]
                return cached['texts'], cached['time'], 'tesseract_cached'
        
        # OCR real
        texts = self.tesseract_ocr_optimized(roi)
        ocr_time = time.time() - start_time
        
        # Cache resultado
        if texts and self.config["tesseract_optimization"]["cache_enabled"]:
            self.ocr_cache[roi_hash] = {
                'texts': texts,
                'time': ocr_time
            }
            
            # Limitar cache
            if len(self.ocr_cache) > 30:
                old_keys = list(self.ocr_cache.keys())[:10]
                for key in old_keys:
                    del self.ocr_cache[key]
        
        # Estadísticas
        self.ocr_times.append(ocr_time)
        if len(self.ocr_times) > 50:
            self.ocr_times = self.ocr_times[-50:]
        
        return texts, ocr_time, 'tesseract'
    
    def is_valid_license_plate(self, text):
        """Validar placa"""
        if not text or len(text.strip()) < 3:
            return False
        
        clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        
        if len(clean_text) < 4 or len(clean_text) > 8:
            return False
        
        has_letter = bool(re.search(r'[A-Z]', clean_text))
        has_number = bool(re.search(r'[0-9]', clean_text))
        
        return has_letter and has_number
    
    def auto_cleanup_check(self):
        """Verificar si es necesario hacer limpieza automática - VERSIÓN ANTI-BLOQUEO TOTAL"""
        self.auto_cleanup_counter += 1
        
        # CRÍTICO: MÚLTIPLES LÍNEAS DE DEFENSA PARA EVITAR LLEGAR A 11
        
        # LÍNEA DE DEFENSA 1: Reset absoluto en 10 detecciones (última oportunidad)
        if self.detections_count >= 10:
            print(f"🚨 EMERGENCIA: {self.detections_count} detecciones - RESET INMEDIATO ANTES DE BLOQUEO")
            old_detections = self.detections_count
            self.detections_count = 0
            self.ocr_cache.clear()
            self.detection_cooldown.clear()
            self.recent_detections.clear()
            self.auto_cleanup_counter = 0
            self.ocr_times = self.ocr_times[-2:]
            print(f"🔄 RESET EMERGENCIA: {old_detections} → 0 (evitando bloqueo fatal)")
            return
        
        # LÍNEA DE DEFENSA 2: Reset preventivo en 8 detecciones
        if self.detections_count >= 8:
            print(f"🚨 ALERTA: {self.detections_count} detecciones - RESET PREVENTIVO CRÍTICO")
            old_detections = self.detections_count
            self.detections_count = 0
            self.ocr_cache.clear()
            self.detection_cooldown.clear()
            self.recent_detections.clear()
            self.auto_cleanup_counter = 0
            self.ocr_times = self.ocr_times[-3:]
            print(f"🔄 RESET PREVENTIVO: {old_detections} → 0 (prevención bloqueo)")
            return
        
        # LÍNEA DE DEFENSA 3: Reset temprano en 6 detecciones
        if self.detections_count >= 6:
            print(f"⚠️ PRECAUCIÓN: {self.detections_count} detecciones - RESET TEMPRANO")
            old_detections = self.detections_count
            self.detections_count = 0
            self.ocr_cache.clear()
            self.detection_cooldown.clear()
            self.recent_detections.clear()
            self.auto_cleanup_counter = 0
            print(f"🔄 RESET TEMPRANO: {old_detections} → 0 (prevención proactiva)")
            return
        
        # Auto-limpieza normal cada 2 detecciones
        if self.auto_cleanup_counter >= self.auto_cleanup_every:
            print(f"🔄 AUTO-LIMPIEZA: {self.auto_cleanup_counter} detecciones alcanzadas")
            
            # Guardar estadísticas antes de limpiar
            cache_size = len(self.ocr_cache)
            cooldown_size = len(self.detection_cooldown)
            
            # Limpieza COMPLETA siempre
            self.ocr_cache.clear()
            print(f"   🗄️ Cache OCR: Limpiado completamente ({cache_size} entradas)")
            
            # Limpieza COMPLETA de cooldowns
            self.detection_cooldown.clear()
            print(f"   ⏱️ Cooldowns: Limpiados completamente ({cooldown_size} entradas)")
            
            # Limpiar detecciones recientes SIEMPRE
            self.recent_detections.clear()
            print(f"   📝 Detecciones recientes: Limpiadas completamente")
            
            # Reset contador de auto-limpieza
            self.auto_cleanup_counter = 0
            
            print(f"✅ AUTO-LIMPIEZA COMPLETADA - contador global: {self.detections_count}")
    
    def process_frame_ai_tesseract(self, frame, frame_time):
        """Procesamiento de frame con Tesseract - VERSIÓN CON VERIFICACIÓN CONSTANTE"""
        try:
            self.ai_processed_frames += 1
            
            # VERIFICACIÓN CRÍTICA: Si por alguna razón llegamos a 11, parar inmediatamente
            if self.detections_count >= 11:
                print(f"🚨 BLOQUEO DETECTADO: {self.detections_count} detecciones - RESETEO FORZADO")
                self.detections_count = 0
                self.ocr_cache.clear()
                self.detection_cooldown.clear()
                self.recent_detections.clear()
                self.auto_cleanup_counter = 0
                print("🔄 SISTEMA DESBLOQUEADO - Reiniciando desde 0")
                return []
            
            # YOLO detection
            results = self.yolo_model(frame, verbose=False, 
                                     conf=self.config["processing"]["confidence_threshold"])
            
            detections = []
            current_time = datetime.now()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        if confidence < self.config["processing"]["plate_confidence_min"]:
                            continue
                        
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            roi = frame[y1:y2, x1:x2]
                            
                            # OCR con Tesseract
                            plate_texts, ocr_time, method = self.read_plate_text_tesseract(roi)
                            
                            for plate_text in plate_texts:
                                text = plate_text['text']
                                
                                if self.is_valid_license_plate(text):
                                    # Cooldown con limpieza automática
                                    cooldown_sec = self.config["processing"]["detection_cooldown_sec"]
                                    if text in self.detection_cooldown:
                                        last_time = self.detection_cooldown[text]
                                        if (current_time - last_time).total_seconds() < cooldown_sec:
                                            continue
                                    
                                    # Limpiar cooldowns antiguos para evitar acumulación
                                    if len(self.detection_cooldown) > 50:
                                        old_entries = []
                                        for plate, last_detection in self.detection_cooldown.items():
                                            if (current_time - last_detection).total_seconds() > 30:
                                                old_entries.append(plate)
                                        for old_plate in old_entries:
                                            del self.detection_cooldown[old_plate]
                                    
                                    self.detection_cooldown[text] = current_time
                                    
                                    # VERIFICACIÓN ANTES DE INCREMENTAR
                                    if self.detections_count >= 10:
                                        print("🚨 Evitando incremento - contador en límite crítico")
                                        self.detections_count = 0
                                        self.ocr_cache.clear()
                                        self.detection_cooldown.clear()
                                        self.recent_detections.clear()
                                        self.auto_cleanup_counter = 0
                                        continue
                                    
                                    # Calcular latencia
                                    processing_latency = time.time() - frame_time
                                    
                                    detection = {
                                        'timestamp': current_time.isoformat(),
                                        'frame_number': self.capture_frame_count,
                                        'ai_frame_number': self.ai_processed_frames,
                                        'plate_text': text,
                                        'yolo_confidence': confidence,
                                        'ocr_confidence': plate_text['confidence'],
                                        'bbox': [x1, y1, x2, y2],
                                        'processing_latency_ms': int(processing_latency * 1000),
                                        'ocr_time_ms': int(ocr_time * 1000),
                                        'ocr_method': method,
                                        'valid': True
                                    }
                                    
                                    detections.append(detection)
                                    
                                    # INCREMENTAR CONTADOR DE FORMA SEGURA
                                    self.detections_count += 1
                                    
                                    # VERIFICACIÓN INMEDIATA DESPUÉS DE INCREMENTAR
                                    if self.detections_count >= 11:
                                        print(f"🚨 CONTADOR ALCANZÓ {self.detections_count} - RESET INMEDIATO")
                                        self.detections_count = 0
                                        self.ocr_cache.clear()
                                        self.detection_cooldown.clear()
                                        self.recent_detections.clear()
                                        self.auto_cleanup_counter = 0
                                    
                                    # Verificar limpieza automática cada detección
                                    self.auto_cleanup_check()
                                    
                                    # Log simplificado sin logging
                                    print(f"🎯 PLACA: {text} "
                                          f"(YOLO: {confidence:.2f}, OCR: {plate_text['confidence']:.2f}, "
                                          f"Latencia: {int(processing_latency * 1000)}ms, "
                                          f"OCR: {int(ocr_time * 1000)}ms-{method.upper()}) "
                                          f"[Contador: {self.detections_count}/10 MAX, Auto-cleanup: {self.auto_cleanup_counter}/{self.auto_cleanup_every}]")
                                    
                                    if self.config["output"]["save_results"]:
                                        self.save_detection(detection)
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Error en IA Tesseract: {e}")
            # En caso de error, verificar contador por seguridad
            if self.detections_count >= 10:
                print("🚨 Error detectado con contador alto - Reset preventivo")
                self.detections_count = 0
                self.auto_cleanup_counter = 0
            return []
    
    def save_detection(self, detection):
        """Guardar detección"""
        try:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"tesseract_detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(results_file, 'a') as f:
                json.dump(detection, f)
                f.write('\n')
                
        except Exception as e:
            print(f"❌ Error guardando: {e}")
    
    def capture_worker(self):
        """Worker de captura"""
        print("📹 Iniciando captura Tesseract...")
        
        rtsp_url = self.config["camera"]["rtsp_url"]
        
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, self.config["tesseract_optimization"]["capture_target_fps"])
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.capture_frame_count += 1
                
                # Display
                try:
                    if not self.display_queue.full():
                        self.display_queue.put(frame.copy(), block=False)
                except:
                    pass
                
                # IA
                ai_every = self.config["tesseract_optimization"]["ai_process_every"]
                if self.capture_frame_count % ai_every == 0:
                    try:
                        if not self.ai_queue.full():
                            self.ai_queue.put(frame.copy(), block=False)
                    except:
                        pass
                
                time.sleep(0.02)  # 50 FPS max
                
            cap.release()
            
        except Exception as e:
            print(f"❌ Error en captura: {e}")
    
    def display_worker(self):
        """Worker de display"""
        print("🖥️ Iniciando display Tesseract...")
        
        window_title = self.config["output"]["window_title"]
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                try:
                    frame = self.display_queue.get(timeout=0.1)
                    
                    # Overlay con información
                    display_frame = self.create_overlay(frame)
                    
                    cv2.imshow(window_title, display_frame)
                    self.display_frame_count += 1
                    
                    # Controls
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.running = False
                        break
                    elif key == ord('r'):
                        self.reset_stats()
                    elif key == ord('c'):
                        self.clear_cache()
                    elif key == ord('s'):
                        self.print_stats()
                    elif key == ord('d'):
                        self.debug_system_state()
                    elif key == ord('f'):
                        self.force_cleanup()
                    
                except queue.Empty:
                    continue
                
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Error en display: {e}")
    
    def ai_worker(self):
        """Worker de IA"""
        print("🧠 Iniciando IA Tesseract...")
        
        try:
            while self.running:
                try:
                    frame = self.ai_queue.get(timeout=0.5)
                    frame_time = time.time()
                    
                    detections = self.process_frame_ai_tesseract(frame, frame_time)
                    
                    if detections:
                        self.result_queue.put(detections)
                        self.recent_detections.extend(detections)
                        
                        if len(self.recent_detections) > 3:
                            self.recent_detections = self.recent_detections[-3:]
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            print(f"❌ Error en IA: {e}")
    
    def create_overlay(self, frame):
        """Crear overlay de información"""
        scale = self.config["tesseract_optimization"]["display_scale"]
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        display_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        if self.start_time:
            runtime = time.time() - self.start_time
            display_fps = self.display_frame_count / runtime if runtime > 0 else 0
            ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
            avg_ocr_time = sum(self.ocr_times[-10:]) / len(self.ocr_times[-10:]) if self.ocr_times else 0
            
            # Info overlay
            info1 = f"FPS: {display_fps:.1f} | IA: {ai_fps:.1f} | Tesseract"
            cv2.putText(display_frame, info1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            info2 = f"Detecciones: {self.detections_count}/10 MAX | OCR avg: {avg_ocr_time:.3f}s"
            cv2.putText(display_frame, info2, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # Info de auto-limpieza con alerta de color
            color = (0, 255, 255)  # Amarillo normal
            if self.detections_count >= 8:
                color = (0, 0, 255)  # Rojo crítico
            elif self.detections_count >= 6:
                color = (0, 165, 255)  # Naranja precaución
            elif self.detections_count >= 4:
                color = (0, 255, 255)  # Amarillo normal
            else:
                color = (0, 255, 0)  # Verde seguro
            
            info3 = f"Contador: {self.detections_count}/10 | Auto-cleanup: {self.auto_cleanup_counter}/{self.auto_cleanup_every}"
            cv2.putText(display_frame, info3, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Detecciones recientes
            for i, detection in enumerate(self.recent_detections[-2:]):
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord * scale) for coord in bbox]
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                plate_text = detection['plate_text']
                ocr_time = detection['ocr_time_ms']
                plate_info = f"{plate_text} ({ocr_time}ms)"
                cv2.putText(display_frame, plate_info, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        return display_frame
    
    def reset_stats(self):
        """Reset estadísticas"""
        self.capture_frame_count = 0
        self.display_frame_count = 0
        self.ai_processed_frames = 0
        self.detections_count = 0
        self.recent_detections = []
        self.ocr_times = []
        self.auto_cleanup_counter = 0
        self.start_time = time.time()
        print("🔄 Reset estadísticas y contador auto-limpieza")
    
    def clear_cache(self):
        """Limpiar cache y cooldowns"""
        cache_size = len(self.ocr_cache)
        cooldown_size = len(self.detection_cooldown)
        
        self.ocr_cache.clear()
        self.detection_cooldown.clear()
        self.recent_detections.clear()
        
        self.auto_cleanup_counter = 0
        
        print(f"🧹 LIMPIEZA MANUAL: {cache_size} OCR, {cooldown_size} cooldowns - Contador auto-limpieza reseteado")
    
    def debug_system_state(self):
        """Debug: mostrar estado interno del sistema con alertas de seguridad"""
        print("🔍 DEBUG ESTADO SISTEMA:")
        print(f"   📊 Contador detecciones: {self.detections_count}/10 (MÁXIMO PERMITIDO)")
        print(f"   🗄️ OCR Cache: {len(self.ocr_cache)} entradas")
        print(f"   ⏱️ Cooldowns: {len(self.detection_cooldown)} placas")
        print(f"   📝 Detecciones recientes: {len(self.recent_detections)}")
        print(f"   🔄 Auto-cleanup: {self.auto_cleanup_counter}/{self.auto_cleanup_every}")
        
        # ALERTAS DE SEGURIDAD
        if self.detections_count >= 8:
            print(f"🚨 ALERTA CRÍTICA: Contador en {self.detections_count} - MUY CERCA DEL BLOQUEO")
        elif self.detections_count >= 6:
            print(f"⚠️ PRECAUCIÓN: Contador en {self.detections_count} - Acercándose al límite")
        elif self.detections_count >= 4:
            print(f"ℹ️ NORMAL: Contador en {self.detections_count} - Funcionamiento normal")
        
        # Mostrar últimas placas en cooldown
        if self.detection_cooldown:
            recent_cooldowns = list(self.detection_cooldown.keys())[-5:]
            print(f"   🚫 Últimas en cooldown: {recent_cooldowns}")
        
        # Alertas por acumulación
        if len(self.ocr_cache) > 20:
            print(f"   ⚠️ Cache OCR grande: {len(self.ocr_cache)} entradas")
        
        if len(self.detection_cooldown) > 30:
            print(f"   ⚠️ Muchos cooldowns: {len(self.detection_cooldown)} placas")
        
        # Próxima auto-limpieza
        remaining = self.auto_cleanup_every - self.auto_cleanup_counter
        print(f"   🕐 Próxima auto-limpieza en: {remaining} detecciones")
        
        # BOTÓN DE PÁNICO
        if self.detections_count >= 9:
            print("🚨 EJECUTANDO RESET DE EMERGENCIA AUTOMÁTICO")
            old_count = self.detections_count
            self.detections_count = 0
            self.ocr_cache.clear()
            self.detection_cooldown.clear()
            self.recent_detections.clear()
            self.auto_cleanup_counter = 0
            print(f"🔄 RESET AUTOMÁTICO: {old_count} → 0")
    
    def force_cleanup(self):
        """Forzar limpieza completa del sistema y RESET TOTAL DE CONTADOR"""
        old_cache = len(self.ocr_cache)
        old_cooldowns = len(self.detection_cooldown)
        old_detections = self.detections_count
        
        # Limpieza TOTAL - incluyendo reset del contador de detecciones
        self.ocr_cache.clear()
        self.detection_cooldown.clear()
        self.recent_detections.clear()
        self.ocr_times = self.ocr_times[-3:]
        
        # RESET TOTAL del contador de detecciones
        self.detections_count = 0
        self.auto_cleanup_counter = 0
        
        print(f"🧹 LIMPIEZA TOTAL FORZADA:")
        print(f"   🗄️ Cache eliminado: {old_cache} entradas")
        print(f"   ⏱️ Cooldowns eliminados: {old_cooldowns} entradas")
        print(f"   📊 Contador detecciones: {old_detections} → 0")
        print(f"   🔄 Contador auto-limpieza: reseteado")
        print("✅ SISTEMA COMPLETAMENTE REINICIADO")
    
    def print_stats(self):
        """Mostrar estadísticas con información de auto-limpieza"""
        if self.start_time:
            runtime = time.time() - self.start_time
            capture_fps = self.capture_frame_count / runtime if runtime > 0 else 0
            ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
            avg_ocr_time = sum(self.ocr_times) / len(self.ocr_times) if self.ocr_times else 0
            
            print("📊 ESTADÍSTICAS TESSERACT:")
            print(f"   ⏱️ Tiempo: {runtime:.1f}s")
            print(f"   📹 FPS Captura: {capture_fps:.1f}")
            print(f"   🧠 FPS IA: {ai_fps:.1f}")
            print(f"   🎯 Detecciones: {self.detections_count}/10 MAX")
            print(f"   🔤 OCR promedio: {avg_ocr_time:.3f}s")
            print(f"   🗄️ Cache OCR: {len(self.ocr_cache)} entradas")
            print(f"   ⏱️ Cooldowns activos: {len(self.detection_cooldown)} placas")
            print(f"   🔄 Auto-limpieza: {self.auto_cleanup_counter}/{self.auto_cleanup_every} (cada {self.auto_cleanup_every} detecciones)")
            
            # Advertencias
            if len(self.detection_cooldown) > 15:
                print("   ⚠️ Muchos cooldowns activos - posible acumulación")
            
            if self.detections_count >= 8:
                print("   🚨 CONTADOR MUY ALTO - Sistema en riesgo de bloqueo")
            elif self.detections_count >= 6:
                print("   ⚠️ Contador elevado - Vigilar de cerca")
            
            # Próxima auto-limpieza
            remaining = self.auto_cleanup_every - self.auto_cleanup_counter
            print(f"   🕐 Próxima auto-limpieza en: {remaining} detecciones")
    
    def run(self):
        """Ejecutar sistema con monitoreo de seguridad"""
        print("⚡ Iniciando sistema LPR con Tesseract y Auto-Limpieza...")
        
        if not self.tesseract_available:
            print("❌ Tesseract no disponible - no se puede ejecutar")
            return
        
        try:
            self.running = True
            self.start_time = time.time()
            
            # Iniciar threads
            threads = []
            
            capture_thread = threading.Thread(target=self.capture_worker, daemon=True)
            display_thread = threading.Thread(target=self.display_worker, daemon=True)
            ai_thread = threading.Thread(target=self.ai_worker, daemon=True)
            
            threads = [capture_thread, display_thread, ai_thread]
            
            for thread in threads:
                thread.start()
            
            print("✅ Sistema Tesseract LPR con Auto-Limpieza iniciado")
            print("🎮 Controles: 'q'=salir, 'r'=reset, 'c'=cache, 's'=stats, 'd'=debug, 'f'=force_cleanup")
            print("🔤 Usando Tesseract OCR - Sin dependencias GPU")
            print(f"🔄 Auto-limpieza configurada cada {self.auto_cleanup_every} detecciones")
            
            # Esperar con monitoreo de seguridad
            while self.running:
                time.sleep(0.1)
                
                # MONITOREO DE SEGURIDAD CONTINUO
                if self.detections_count >= 11:
                    print("🚨 BLOQUEO DETECTADO EN BUCLE PRINCIPAL - RESET FORZADO")
                    self.detections_count = 0
                    self.ocr_cache.clear()
                    self.detection_cooldown.clear()
                    self.recent_detections.clear()
                    self.auto_cleanup_counter = 0
                    print("🔄 SISTEMA DESBLOQUEADO DESDE BUCLE PRINCIPAL")
                
                # Verificar threads
                for thread in threads:
                    if not thread.is_alive():
                        print(f"❌ Thread {thread.name} murió")
                        self.running = False
                        break
                        
        except KeyboardInterrupt:
            print("⏹️ Detenido por usuario")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            self.stop()
    
    def stop(self):
        """Detener sistema"""
        print("🛑 Deteniendo sistema Tesseract LPR...")
        
        self.running = False
        time.sleep(1)  # Dar tiempo a threads
        
        # Estadísticas finales
        if self.start_time:
            runtime = time.time() - self.start_time
            capture_fps = self.capture_frame_count / runtime if runtime > 0 else 0
            ai_fps = self.ai_processed_frames / runtime if runtime > 0 else 0
            avg_ocr_time = sum(self.ocr_times) / len(self.ocr_times) if self.ocr_times else 0
            detections_per_min = (self.detections_count / runtime) * 60 if runtime > 0 else 0
            auto_cleanups_performed = max(1, self.detections_count // self.auto_cleanup_every)
            
            print("📊 ESTADÍSTICAS FINALES TESSERACT:")
            print(f"   ⏱️ Tiempo total: {runtime:.1f}s")
            print(f"   📹 FPS Captura: {capture_fps:.1f}")
            print(f"   🧠 FPS IA: {ai_fps:.1f}")
            print(f"   🎯 Detecciones totales: {self.detections_count}")
            print(f"   ⚡ Detecciones/min: {detections_per_min:.1f}")
            print(f"   🔤 OCR tiempo promedio: {avg_ocr_time:.3f}s")
            print(f"   🗄️ Cache final: {len(self.ocr_cache)} entradas")
            print(f"   🔄 Auto-limpiezas estimadas: {auto_cleanups_performed}")
            print("🔤 Sistema Tesseract: SIN GPU/PyTorch + PREVENCIÓN TOTAL DE BLOQUEO + SIN LOGGING")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema LPR con Tesseract y Prevención de Bloqueo - SIN LOGGING")
    parser.add_argument("--config", default="config/ptz_config.json")
    parser.add_argument("--ai-every", type=int, default=5, help="IA cada N frames (default: 5)")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Cooldown detecciones (default: 10.0s)")
    parser.add_argument("--timeout", type=float, default=0.5, help="Timeout OCR (default: 0.5s)")
    parser.add_argument("--auto-cleanup-every", type=int, default=2, help="Auto-limpiar cache cada N detecciones (default: 2)")
    parser.add_argument("--responsive", action="store_true", help="Modo ultra-responsivo (IA cada 3 frames)")
    parser.add_argument("--conservative", action="store_true", help="Modo conservativo (IA cada 8 frames)")
    parser.add_argument("--install-tesseract", action="store_true", help="Forzar instalación Tesseract")
    
    args = parser.parse_args()
    
    print("🔤 SISTEMA LPR CON TESSERACT - SIN LOGGING (DEBUG BLOQUEO)")
    print("=" * 60)
    print("✅ Sin PyTorch")
    print("✅ Sin GPU")
    print("✅ Solo CPU")
    print("✅ Tesseract OCR Optimizado")
    print("❌ SIN LOGGING (para debug)")
    print("🔄 Auto-limpieza de cache cada 2 detecciones")
    print("🛡️ MÚLTIPLES LÍNEAS DE DEFENSA ANTI-BLOQUEO:")
    print("   • Reset temprano en 6 detecciones")
    print("   • Reset preventivo en 8 detecciones")
    print("   • Reset de emergencia en 10 detecciones")
    print("   • Reset absoluto si llega a 11+ detecciones")
    print("🚨 Sistema GARANTIZADO para NUNCA bloquearse")
    print()
    
    if args.install_tesseract:
        print("📦 Forzando instalación de Tesseract...")
        temp_system = TesseractLPR()
        if not temp_system.tesseract_available:
            temp_system.install_tesseract()
        return
    
    try:
        system = TesseractLPR(args.config)
        
        # Configurar frecuencia de auto-limpieza
        system.auto_cleanup_every = args.auto_cleanup_every
        
        # Aplicar modo especial si se solicita
        if args.responsive:
            system.config["tesseract_optimization"]["ai_process_every"] = 3
            system.config["processing"]["detection_cooldown_sec"] = 5.0
            system.config["processing"]["ocr_min_confidence"] = 0.6
            system.config["tesseract_optimization"]["ocr_timeout"] = 0.4
            system.auto_cleanup_every = 2
            mode_description = "ULTRA-RESPONSIVO (auto-limpieza cada 2)"
        elif args.conservative:
            system.config["tesseract_optimization"]["ai_process_every"] = 8
            system.config["processing"]["detection_cooldown_sec"] = 15.0
            system.config["processing"]["ocr_min_confidence"] = 0.8
            system.config["tesseract_optimization"]["ocr_timeout"] = 0.6
            system.auto_cleanup_every = 3
            mode_description = "CONSERVATIVO (auto-limpieza cada 3)"
        else:
            system.config["tesseract_optimization"]["ai_process_every"] = args.ai_every
            system.config["processing"]["detection_cooldown_sec"] = args.cooldown
            system.config["tesseract_optimization"]["ocr_timeout"] = args.timeout
            mode_description = f"BALANCEADO (auto-limpieza cada {system.auto_cleanup_every})"
        
        print(f"⚙️ Configuración {mode_description}:")
        print(f"   🧠 IA cada: {system.config['tesseract_optimization']['ai_process_every']} frames")
        print(f"   ⏱️ Cooldown: {system.config['processing']['detection_cooldown_sec']}s")
        print(f"   ⏰ OCR timeout: {system.config['tesseract_optimization']['ocr_timeout']}s")
        print(f"   🎯 Confianza OCR: {system.config['processing'].get('ocr_min_confidence', 0.7)}")
        print(f"   🔄 Auto-limpieza: cada {system.auto_cleanup_every} detecciones")
        print(f"   🛡️ Límite máximo: 10 detecciones (reset automático)")
        print("   ❌ LOGGING DESHABILITADO (para debug)")
        print()
        
        if not system.tesseract_available:
            print("❌ Tesseract no disponible")
            print("💡 Ejecuta: python3 tesseract_lpr_complete.py --install-tesseract")
            return 1
        
        print("🚀 SISTEMA TESSERACT LPR - SIN LOGGING")
        print("🔧 Esta versión NO genera archivos de log")
        print("📊 Solo salida por consola para debug del problema de bloqueo")
        print("🛡️ Todas las protecciones anti-bloqueo siguen activas")
        print()
        
        system.run()
        
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())