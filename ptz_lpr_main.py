#!/usr/bin/env python3
"""
🎥 SISTEMA LPR CON CÁMARA PTZ - JETSON ORIN NANO
==================================================
Sistema completo de reconocimiento de placas vehiculares
con cámara PTZ conectada por Ethernet directo.

Autor: Sistema LPR automatizado
Fecha: 2025-06-28
Uso: python ptz_lpr_main.py
"""

import cv2
import json
import time
import argparse
import logging
import subprocess
import threading
import re
from datetime import datetime
from pathlib import Path

# Importar módulos del proyecto
try:
    from ultralytics import YOLO
    import easyocr
    import numpy as np
    
    # Intentar importar módulos opcionales
    try:
        from util import get_car
        HAS_UTIL = True
    except ImportError:
        HAS_UTIL = False
        print("⚠️ util.py no disponible, usando funciones básicas")
    
    print("✅ Módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("💡 Asegúrate de estar en el entorno virtual: source jetson_env/bin/activate")
    exit(1)

def is_valid_license_plate(text):
    """
    Validar si el texto corresponde a una placa válida
    Acepta varios formatos comunes de placas
    """
    if not text or len(text.strip()) < 3:
        return False
    
    # Limpiar texto
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
    
    # Patrones comunes de placas
    patterns = [
        r'^[A-Z]{3}[0-9]{3}$',      # ABC123
        r'^[A-Z]{3}[0-9]{2}[A-Z]$', # ABC12D
        r'^[A-Z]{2}[0-9]{4}$',      # AB1234
        r'^[A-Z]{4}[0-9]{2}$',      # ABCD12
        r'^[0-9]{3}[A-Z]{3}$',      # 123ABC
        r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$', # A12BCD
        r'^[A-Z]{2}[0-9]{3}[A-Z]{1}$', # AB123C
    ]
    
    # Verificar longitud (entre 4 y 8 caracteres es razonable)
    if len(clean_text) < 4 or len(clean_text) > 8:
        return False
    
    # Debe tener al menos un número y una letra
    has_letter = bool(re.search(r'[A-Z]', clean_text))
    has_number = bool(re.search(r'[0-9]', clean_text))
    
    if not (has_letter and has_number):
        return False
    
    # Verificar patrones específicos
    for pattern in patterns:
        if re.match(pattern, clean_text):
            return True
    
    # Verificación más flexible: al menos 3 caracteres, letras y números
    if len(clean_text) >= 4 and has_letter and has_number:
        return True
    
    return False

class PTZLPRSystem:
    """Sistema completo LPR con cámara PTZ"""
    
    def __init__(self, config_path="config/ptz_config.json"):
        self.config_path = Path(config_path)
        
        # Configurar logging primero
        self.setup_logging()
        
        # Luego cargar configuración
        self.load_config()
        
        # Estado del sistema
        self.running = False
        self.frame_count = 0
        self.detections_count = 0
        self.start_time = None
        
        # Inicializar modelos
        self.setup_models()
        
        # Resultados
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.logger.info("🚀 Sistema PTZ-LPR inicializado correctamente")
    
    def load_config(self):
        """Cargar configuración del sistema"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuración cargada desde {self.config_path}")
        except FileNotFoundError:
            # Configuración por defecto
            self.config = {
                "camera": {
                    "ip": "192.168.1.101",
                    "user": "admin",
                    "password": "admin",
                    "rtsp_url": "rtsp://admin:admin@192.168.1.101/cam/realmonitor?channel=1&subtype=1"
                },
                "jetson": {
                    "ip": "192.168.1.100",
                    "interface": "enP8p1s0"
                },
                "processing": {
                    "input_resolution": [640, 480],
                    "confidence_threshold": 0.25,
                    "plate_confidence_min": 0.2,
                    "max_detections": 8,
                    "skip_frames": 2
                },
                "output": {
                    "save_results": True,
                    "save_images": False,
                    "show_video": True
                }
            }
            self.save_config()
            print("📄 Configuración por defecto creada")
    
    def save_config(self):
        """Guardar configuración"""
        self.config_path.parent.mkdir(exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def setup_logging(self):
        """Configurar sistema de logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"ptz_lpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - PTZ-LPR - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print(f"📝 Logs guardados en: {log_file}")
    
    def setup_network(self):
        """Configurar red para cámara PTZ"""
        self.logger.info("🔧 Configurando red PTZ...")
        
        interface = self.config["jetson"]["interface"]
        jetson_ip = self.config["jetson"]["ip"]
        
        try:
            # Verificar si la interfaz existe
            check_result = subprocess.run(f"ip link show {interface}", 
                                        shell=True, capture_output=True)
            if check_result.returncode != 0:
                self.logger.warning(f"⚠️ Interfaz {interface} no encontrada, saltando configuración")
                return
            
            # Configurar interfaz de red
            commands = [
                f"sudo ip addr flush dev {interface} 2>/dev/null || true",
                f"sudo ip addr add {jetson_ip}/24 dev {interface} 2>/dev/null || true",
                f"sudo ethtool -s {interface} speed 100 duplex full autoneg off 2>/dev/null || true"
            ]
            
            for cmd in commands:
                subprocess.run(cmd, shell=True, capture_output=True)
            
            self.logger.info(f"✅ Red configurada: {interface} -> {jetson_ip}")
            
            # Verificar conectividad (opcional)
            time.sleep(2)
            camera_ip = self.config["camera"]["ip"]
            ping_result = subprocess.run(f"ping -c 1 -W 2 {camera_ip}", 
                                       shell=True, capture_output=True)
            
            if ping_result.returncode == 0:
                self.logger.info(f"✅ Conectividad con cámara {camera_ip}")
            else:
                self.logger.info(f"📡 Cámara {camera_ip} no responde a ping (puede ser normal)")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Configuración de red opcional falló: {e}")
    
    def setup_models(self):
        """Inicializar modelos YOLO y EasyOCR"""
        self.logger.info("🤖 Cargando modelos de IA...")
        
        try:
            # Buscar modelos disponibles
            model_files = list(Path(".").glob("*.pt"))
            
            if not model_files:
                self.logger.error("❌ No se encontraron modelos .pt")
                raise FileNotFoundError("Modelos YOLO no encontrados")
            
            # Prioridad: modelo personalizado > yolo11n > yolov8n
            preferred_models = ["license_plate_detector.pt", "yolo11n.pt", "yolov8n.pt"]
            
            selected_model = None
            for model_name in preferred_models:
                if Path(model_name).exists():
                    selected_model = model_name
                    break
            
            if not selected_model:
                selected_model = str(model_files[0])
            
            self.logger.info(f"📦 Cargando modelo YOLO: {selected_model}")
            self.yolo_model = YOLO(selected_model)
            
            # Inicializar EasyOCR
            self.logger.info("📝 Inicializando EasyOCR...")
            # Usar CPU por defecto en Jetson para estabilidad
            use_gpu = False  # Cambiar a True si CUDA está bien configurado
            self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            
            self.logger.info("✅ Modelos cargados exitosamente")
            
        except Exception as e:
            self.logger.error(f"❌ Error cargando modelos: {e}")
            raise
    
    def connect_camera(self):
        """Conectar con cámara PTZ o local"""
        # Intentar cámara PTZ primero
        rtsp_url = self.config["camera"]["rtsp_url"]
        self.logger.info(f"📹 Intentando conexión PTZ: {rtsp_url}")
        
        # Configurar red PTZ
        self.setup_network()
        
        camera_sources = [
            # PTZ RTSP
            {
                'name': 'PTZ RTSP',
                'source': rtsp_url,
                'backend': cv2.CAP_FFMPEG
            },
            # Cámara USB/local
            {
                'name': 'Cámara Local',
                'source': 0,
                'backend': cv2.CAP_V4L2
            },
            # Fallback genérico
            {
                'name': 'Cámara Genérica',
                'source': 0,
                'backend': cv2.CAP_ANY
            }
        ]
        
        for camera in camera_sources:
            try:
                self.logger.info(f"🔄 Probando {camera['name']}...")
                
                cap = cv2.VideoCapture(camera['source'], camera['backend'])
                
                # Configurar propiedades para mejor rendimiento
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Probar captura
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    height, width = frame.shape[:2]
                    self.logger.info(f"✅ {camera['name']} conectada: {width}x{height}")
                    return cap
                else:
                    cap.release()
                    
            except Exception as e:
                self.logger.warning(f"⚠️ {camera['name']} falló: {e}")
        
        raise ConnectionError("No se pudo conectar a ninguna cámara")
    
    def detect_vehicles_and_plates(self, frame):
        """Detectar vehículos y placas en el frame"""
        try:
            results = self.yolo_model(frame, verbose=False, 
                                     conf=self.config["processing"]["confidence_threshold"])
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Obtener coordenadas y confianza
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filtrar por confianza mínima
                        if confidence < self.config["processing"]["plate_confidence_min"]:
                            continue
                        
                        # Asegurar coordenadas válidas
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        # Verificar que el área sea válida
                        if x2 > x1 and y2 > y1:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en detección YOLO: {e}")
            return []
    
    def read_plate_text(self, roi):
        """Leer texto de placa usando EasyOCR"""
        try:
            if roi.size == 0:
                return []
            
            # Preprocesamiento de imagen
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Mejorar contraste y calidad
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Resize si es muy pequeño
            h, w = enhanced.shape
            if h < 30 or w < 80:
                scale = max(30/h, 80/w)
                new_h, new_w = int(h*scale), int(w*scale)
                enhanced = cv2.resize(enhanced, (new_w, new_h))
            
            # OCR
            results = self.ocr_reader.readtext(enhanced)
            
            # Procesar resultados
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.3 and len(text.strip()) > 2:
                    # Limpiar texto
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    if len(cleaned_text) >= 3:
                        texts.append({
                            'text': cleaned_text,
                            'confidence': confidence,
                            'bbox': bbox
                        })
            
            return texts
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en OCR: {e}")
            return []
    
    def process_frame(self, frame):
        """Procesar un frame completo"""
        self.frame_count += 1
        
        # Saltar frames para mejorar rendimiento
        skip_frames = self.config.get("processing", {}).get("skip_frames", 2)
        if self.frame_count % (skip_frames + 1) != 0:
            return frame, []
        
        # Detectar objetos
        detections = self.detect_vehicles_and_plates(frame)
        
        processed_detections = []
        
        for detection in detections:
            try:
                # Extraer ROI
                x1, y1, x2, y2 = detection['bbox']
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Leer texto de placa
                    plate_texts = self.read_plate_text(roi)
                    
                    for plate_text in plate_texts:
                        text = plate_text['text']
                        
                        # Validar placa
                        if is_valid_license_plate(text):
                            result = {
                                'timestamp': datetime.now().isoformat(),
                                'frame_number': self.frame_count,
                                'plate_text': text,
                                'confidence': detection['confidence'],
                                'ocr_confidence': plate_text['confidence'],
                                'bbox': detection['bbox'],
                                'valid': True
                            }
                            
                            processed_detections.append(result)
                            self.detections_count += 1
                            
                            self.logger.info(f"🎯 PLACA DETECTADA: {text} "
                                           f"(YOLO: {detection['confidence']:.2f}, "
                                           f"OCR: {plate_text['confidence']:.2f})")
                            
                            # Guardar resultado
                            save_results = self.config.get("output", {}).get("save_results", True)
                            save_images = self.config.get("output", {}).get("save_images", False)
                            
                            if save_results:
                                self.save_detection(result, roi if save_images else None)
                            
            except Exception as e:
                self.logger.warning(f"⚠️ Error procesando detección: {e}")
        
        return frame, processed_detections
    
    def save_detection(self, detection, plate_image=None):
        """Guardar detección en archivo"""
        try:
            # Guardar en JSON Lines
            results_file = self.results_dir / f"detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(results_file, 'a') as f:
                json.dump(detection, f)
                f.write('\n')
            
            # Guardar imagen si se solicita
            if plate_image is not None:
                save_images = self.config.get("output", {}).get("save_images", False)
                if save_images:
                    image_dir = Path("detected_plates")
                    image_dir.mkdir(exist_ok=True)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    image_file = image_dir / f"plate_{detection['plate_text']}_{timestamp}.jpg"
                    cv2.imwrite(str(image_file), plate_image)
                
        except Exception as e:
            self.logger.error(f"❌ Error guardando detección: {e}")
    
    def draw_detections(self, frame, detections):
        """Dibujar detecciones en el frame"""
        display_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Dibujar rectángulo
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Dibujar texto
            text = f"{detection['plate_text']} ({detection['confidence']:.2f})"
            
            # Fondo para el texto
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, (x1, y1-25), (x1 + text_size[0], y1), (0, 255, 0), -1)
            cv2.putText(display_frame, text, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Información del sistema
        fps = self.frame_count / (time.time() - self.start_time) if self.start_time else 0
        info_text = f"FPS: {fps:.1f} | Frames: {self.frame_count} | Detecciones: {self.detections_count}"
        
        # Fondo para información
        cv2.rectangle(display_frame, (10, 5), (500, 35), (0, 0, 0), -1)
        cv2.putText(display_frame, info_text, (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_frame
    
    def run(self):
        """Ejecutar sistema principal"""
        self.logger.info("🚀 Iniciando sistema PTZ-LPR...")
        
        try:
            # Conectar cámara
            cap = self.connect_camera()
            self.running = True
            self.start_time = time.time()
            
            self.logger.info("✅ Sistema en funcionamiento")
            self.logger.info("🎮 Controles: 'q'=salir, 's'=captura, 'r'=reset stats")
            
            while self.running:
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("⚠️ No se pudo capturar frame")
                    time.sleep(0.1)
                    continue
                
                # Procesar frame
                processed_frame, detections = self.process_frame(frame)
                
                # Mostrar video si está habilitado
                show_video = self.config.get("output", {}).get("show_video", True)
                if show_video:
                    display_frame = self.draw_detections(processed_frame, detections)
                    cv2.imshow('PTZ-LPR System', display_frame)
                    
                    # Verificar teclas
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' o ESC
                        break
                    elif key == ord('s'):  # 's' para captura
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        cv2.imwrite(f"capture_{timestamp}.jpg", processed_frame)
                        self.logger.info(f"📸 Captura guardada: capture_{timestamp}.jpg")
                    elif key == ord('r'):  # 'r' para reset
                        self.frame_count = 0
                        self.detections_count = 0
                        self.start_time = time.time()
                        self.logger.info("🔄 Estadísticas reiniciadas")
                
                # Control de velocidad (máximo 30 FPS)
                time.sleep(0.033)
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Detenido por usuario")
        except Exception as e:
            self.logger.error(f"❌ Error en ejecución: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            # Limpieza
            self.running = False
            if 'cap' in locals():
                cap.release()
            
            # Solo destruir ventanas si la GUI está disponible
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # Ignorar errores de GUI
            
            # Estadísticas finales
            if self.start_time:
                runtime = time.time() - self.start_time
                avg_fps = self.frame_count / runtime if runtime > 0 else 0
                self.logger.info("📊 ESTADÍSTICAS FINALES:")
                self.logger.info(f"   ⏱️  Tiempo ejecución: {runtime:.1f}s")
                self.logger.info(f"   🎬 Frames procesados: {self.frame_count}")
                self.logger.info(f"   📈 FPS promedio: {avg_fps:.1f}")
                self.logger.info(f"   🎯 Placas detectadas: {self.detections_count}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Sistema PTZ-LPR para Jetson Orin Nano")
    parser.add_argument("--config", default="config/ptz_config.json",
                       help="Archivo de configuración")
    parser.add_argument("--camera-ip", help="IP de la cámara PTZ")
    parser.add_argument("--jetson-ip", help="IP del Jetson")
    parser.add_argument("--no-video", action="store_true", 
                       help="No mostrar video en pantalla")
    parser.add_argument("--save-images", action="store_true",
                       help="Guardar imágenes de placas detectadas")
    
    args = parser.parse_args()
    
    print("🎥 SISTEMA PTZ-LPR - JETSON ORIN NANO")
    print("=" * 50)
    print("⚡ Versión optimizada con auto-detección")
    print("🔧 Configuración automática de red PTZ")
    print("🎯 Detección inteligente de placas")
    print("=" * 50)
    
    try:
        # Crear sistema
        system = PTZLPRSystem(args.config)
        
        # Aplicar argumentos de línea de comandos
        if args.camera_ip:
            system.config["camera"]["ip"] = args.camera_ip
            # Actualizar URL RTSP
            user = system.config["camera"]["user"]
            password = system.config["camera"]["password"]
            system.config["camera"]["rtsp_url"] = f"rtsp://{user}:{password}@{args.camera_ip}/cam/realmonitor?channel=1&subtype=1"
        
        if args.jetson_ip:
            system.config["jetson"]["ip"] = args.jetson_ip
        
        if args.no_video:
            if "output" not in system.config:
                system.config["output"] = {}
            system.config["output"]["show_video"] = False
            
        if args.save_images:
            if "output" not in system.config:
                system.config["output"] = {}
            system.config["output"]["save_images"] = True
        
        # Ejecutar sistema
        system.run()
        
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
