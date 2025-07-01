#!/usr/bin/env python3
"""
LPR Stream Principal - Adaptado de main_final.py
Testing en PC, Deploy en Jetson
"""

import time
import threading
import queue
import logging
from datetime import datetime
import numpy as np
import sys
import os

# Agregar path del proyecto principal
sys.path.append(str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importar módulos LPR existentes
from main_final import get_all_cached_models
from plate_validator import ultrafast_ocr_optimized_with_validation
from stream.config.stream_config import StreamConfig
from stream.database.db_manager import DatabaseManager
from stream.utils.rtsp_handler import RTSPHandler

class LPRStream:
    def __init__(self, mode='development'):
        self.config = StreamConfig(mode)
        self.setup_logging()
        self.setup_components()
        
        # Control de ejecución
        self.running = False
        self.processing_queue = queue.Queue(maxsize=20)
        self.results_queue = queue.Queue()
        
        # Métricas de rendimiento
        self.fps_counter = 0
        self.detection_counter = 0
        self.start_time = time.time()
        self.last_stats_time = time.time()
        
    def setup_logging(self):
        """Configurar logging sin emojis para evitar problemas Unicode"""
        log_file = self.config.logs_dir / "lpr_stream.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        # Configurar encoding para handler de consola en Windows
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stderr>':
                try:
                    handler.stream.reconfigure(encoding='utf-8')
                except:
                    pass  # En caso de que reconfigure no esté disponible
    
    def setup_components(self):
        """Inicializar componentes del sistema"""
        try:
            # 1. Configurar optimizaciones CUDA
            self.config.setup_cuda_optimizations()
            
            # 2. Cargar modelos LPR (reutilizar sistema existente)
            self.logger.info("[LPR] Cargando modelos LPR...")
            models, device, easyocr_reader = get_all_cached_models()
            
            if models is None or easyocr_reader is None:
                raise Exception("Error cargando modelos LPR")
            
            self.coco_model, self.license_detector, self.model_name = models
            self.device = device
            self.easyocr_reader = easyocr_reader
            
            self.logger.info(f"[OK] Modelos cargados: {self.model_name} en {device}")
            
            # 3. Configurar manejador de stream
            self.stream_handler = RTSPHandler(self.config)
            
            # 4. Configurar base de datos (con manejo de errores)
            try:
                self.db_manager = DatabaseManager(self.config.db_config)
                self.logger.info("[OK] Base de datos conectada")
            except Exception as db_error:
                self.logger.warning(f"[WARN] BD no disponible: {db_error}")
                self.logger.warning("[WARN] Continuando sin base de datos")
                self.db_manager = None
            
            # 5. Warm-up del sistema
            self._warmup_models()
            
            self.logger.info("[OK] Todos los componentes inicializados")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error inicializando componentes: {e}")
            raise
    
    def _warmup_models(self):
        """Warm-up de modelos con imagen dummy"""
        self.logger.info("[WARMUP] Calentando modelos...")
        
        dummy_frame = np.zeros((*self.config.input_resolution[::-1], 3), dtype=np.uint8)
        
        # Warm-up YOLO
        self.coco_model(dummy_frame, imgsz=320, conf=0.5, verbose=False)
        self.license_detector(dummy_frame, imgsz=320, conf=0.3, verbose=False)
        
        # Warm-up EasyOCR
        dummy_crop = np.zeros((50, 150, 3), dtype=np.uint8)
        try:
            ultrafast_ocr_optimized_with_validation(dummy_crop, self.easyocr_reader)
        except:
            pass
        
        self.logger.info("[WARMUP] Warm-up completado")
    
    def process_frame_lpr(self, frame, timestamp):
        """Procesar frame con LPR (adaptado de main_final.py)"""
        try:
            detections = []
            
            # PASO 1: Detección de vehículos
            vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
            
            vehicle_results = self.coco_model(
                frame,
                imgsz=320,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                max_det=self.config.max_detections,
                device=self.device,
                verbose=False,
                half=self.config.use_half_precision
            )[0]
            
            vehicle_detections = []
            for detection in vehicle_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    vehicle_detections.append([x1, y1, x2, y2, score])
            
            # PASO 2: Detección de placas
            plate_results = self.license_detector(
                frame,
                imgsz=320,
                conf=self.config.plate_confidence_min,
                iou=0.4,
                max_det=5,
                device=self.device,
                verbose=False,
                half=self.config.use_half_precision
            )[0]
            
            # PASO 3: OCR y validación
            for plate_detection in plate_results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate_detection
                
                # Encontrar vehículo asociado
                assigned_vehicle = self._find_associated_vehicle(
                    [x1, y1, x2, y2], vehicle_detections
                )
                
                if assigned_vehicle is not None:
                    # Recortar placa con padding
                    padding = 20
                    y1_pad = max(0, int(y1) - padding)
                    y2_pad = min(frame.shape[0], int(y2) + padding)
                    x1_pad = max(0, int(x1) - padding)
                    x2_pad = min(frame.shape[1], int(x2) + padding)
                    
                    plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # OCR con validación colombiana
                    plate_text, confidence = ultrafast_ocr_optimized_with_validation(
                        plate_crop, self.easyocr_reader
                    )
                    
                    if plate_text and confidence > 0.3:
                        detection_data = {
                            'timestamp': datetime.fromtimestamp(timestamp),
                            'vehicle_bbox': assigned_vehicle,
                            'plate_bbox': [x1, y1, x2, y2],
                            'plate_text': plate_text,
                            'confidence': confidence,
                            'plate_score': score,
                            'camera_location': self.config.mode
                        }
                        detections.append(detection_data)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error procesando frame: {e}")
            return []
    
    def _find_associated_vehicle(self, plate_bbox, vehicle_detections):
        """Encontrar vehículo asociado a placa (del código original)"""
        px1, py1, px2, py2 = plate_bbox
        
        for vehicle in vehicle_detections:
            vx1, vy1, vx2, vy2, score = vehicle
            
            # Verificar si placa está dentro del vehículo
            if px1 >= vx1 and py1 >= vy1 and px2 <= vx2 and py2 <= vy2:
                return vehicle
        
        return None
    
    def frame_processing_thread(self):
        """Thread para procesamiento de frames"""
        while self.running:
            try:
                # Obtener frame del stream
                frame, timestamp = self.stream_handler.get_frame(timeout=1.0)
                
                if frame is None:
                    continue
                
                # Procesar LPR
                start_time = time.time()
                detections = self.process_frame_lpr(frame, timestamp)
                process_time = time.time() - start_time
                
                # Enviar resultados si hay detecciones
                if detections:
                    self.results_queue.put(detections)
                    self.detection_counter += len(detections)
                    
                    placas_detectadas = [d['plate_text'] for d in detections]
                    self.logger.info(f"[DETECT] Placas: {', '.join(placas_detectadas)} ({process_time:.3f}s)")
                
                # Actualizar métricas
                self.fps_counter += 1
                self._update_stats()
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error en procesamiento: {e}")
                time.sleep(0.1)
    
    def database_thread(self):
        """Thread para gestión de base de datos"""
        while self.running:
            try:
                detections = self.results_queue.get(timeout=1.0)
                
                # Solo procesar si hay BD disponible
                if self.db_manager is None:
                    # Sin BD, solo logear las detecciones
                    for detection in detections:
                        self.logger.info(f"[NO-DB] Placa detectada: {detection['plate_text']} (conf: {detection['confidence']:.3f})")
                    continue
                
                for detection in detections:
                    try:
                        # Insertar en BD
                        success = self.db_manager.insert_detection(detection)
                        
                        if success:
                            # Verificar autorización
                            auth_info = self.db_manager.check_authorized_vehicle(
                                detection['plate_text']
                            )
                            
                            status = "[AUTH] AUTORIZADO" if auth_info['authorized'] else "[NOAUTH] NO AUTORIZADO"
                            owner = f" ({auth_info['owner_name']})" if auth_info['owner_name'] else ""
                            
                            self.logger.info(f"[DB] {detection['plate_text']}: {status}{owner}")
                        
                    except Exception as e:
                        self.logger.error(f"[ERROR] Error BD para {detection['plate_text']}: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"[ERROR] Error en thread BD: {e}")
    
    def _update_stats(self):
        """Actualizar estadísticas de rendimiento"""
        current_time = time.time()
        
        if current_time - self.last_stats_time >= 10.0:  # Cada 10 segundos
            elapsed = current_time - self.last_stats_time
            fps = self.fps_counter / elapsed
            
            # Stats del stream
            stream_stats = self.stream_handler.get_stats()
            
            self.logger.info(
                f"[STATS] FPS: {fps:.1f} | Detecciones: {self.detection_counter} | "
                f"Queue: {stream_stats['queue_size']}/{stream_stats['max_queue_size']}"
            )
            
            # Reset contadores
            self.fps_counter = 0
            self.last_stats_time = current_time
    
    def start_stream(self):
        """Iniciar procesamiento de stream"""
        try:
            self.logger.info(f"[START] Iniciando LPR Stream ({self.config.mode})")
            
            # Iniciar captura de video
            if not self.stream_handler.start_capture():
                raise Exception("No se pudo iniciar captura de video")
            
            # Iniciar threads de procesamiento
            self.running = True
            
            self.processing_thread = threading.Thread(
                target=self.frame_processing_thread, daemon=True
            )
            self.db_thread = threading.Thread(
                target=self.database_thread, daemon=True
            )
            
            self.processing_thread.start()
            self.db_thread.start()
            
            self.logger.info("[OK] Todos los threads iniciados")
            
            # Loop principal
            try:
                while self.running:
                    time.sleep(1)
                    
                    # Verificar salud de threads
                    if not self.processing_thread.is_alive():
                        self.logger.error("[ERROR] Thread de procesamiento falló")
                        break
                    
                    if not self.db_thread.is_alive():
                        self.logger.error("[ERROR] Thread de BD falló")
                        break
                        
            except KeyboardInterrupt:
                self.logger.info("[STOP] Deteniendo stream por usuario...")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error iniciando stream: {e}")
        finally:
            self.stop_stream()
    
    def stop_stream(self):
        """Detener stream"""
        self.logger.info("[STOP] Deteniendo LPR Stream...")
        
        self.running = False
        
        # Detener captura
        self.stream_handler.stop_capture()
        
        # Cerrar BD (solo si existe)
        if self.db_manager:
            self.db_manager.close()
        
        # Estadísticas finales
        total_time = time.time() - self.start_time
        avg_fps = self.detection_counter / total_time if total_time > 0 else 0
        
        self.logger.info("[STATS] Estadísticas finales:")
        self.logger.info(f"   Tiempo total: {total_time:.1f}s")
        self.logger.info(f"   Detecciones: {self.detection_counter}")
        self.logger.info(f"   FPS promedio: {avg_fps:.2f}")
        
        self.logger.info("[OK] Stream detenido")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LPR Stream')
    parser.add_argument('--mode', choices=['development', 'production'], 
                       default='development', help='Modo de ejecución')
    
    args = parser.parse_args()
    
    print(f"[INIT] Iniciando LPR Stream en modo: {args.mode}")
    
    try:
        lpr_stream = LPRStream(mode=args.mode)
        lpr_stream.start_stream()
    except KeyboardInterrupt:
        print("\n[EXIT] Saliendo...")
    except Exception as e:
        print(f"[ERROR] Error fatal: {e}")

if __name__ == "__main__":
    main()
