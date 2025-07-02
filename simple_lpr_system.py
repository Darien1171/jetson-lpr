#!/usr/bin/env python3
"""
🚀 SISTEMA LPR SIMPLIFICADO
==========================
Todas las funcionalidades principales pero con código más simple y directo.
- Detección YOLO + OCR Tesseract
- Múltiples fuentes de video (RTSP, local)
- Auto-limpieza sin bloqueos
- GUI opcional
- Threading simplificado

Autor: Sistema LPR Simple
Fecha: 2025-07-01
Versión: 1.0 (Simple)
"""

import cv2
import numpy as np
import time
import re
import subprocess
import threading
import queue
import json
import os
import sys
import signal
from datetime import datetime
from pathlib import Path

class SimpleLPR:
    """Sistema LPR simplificado pero completo"""
    
    def __init__(self, config_file="config/simple_lpr.json"):
        # Estado básico
        self.running = False
        self.frame_count = 0
        self.detections_count = 0
        self.last_cleanup = time.time()
        
        # Control de bloqueos
        self.max_detections = 8  # Reset antes de problemas
        
        # Threading simple
        self.frame_queue = queue.Queue(maxsize=3)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Cache OCR
        self.ocr_cache = {}
        self.detection_cooldown = {}
        
        # Configuración
        self.config_file = Path(config_file)
        self.load_simple_config()
        
        # Inicializar componentes
        print("🚀 Inicializando Sistema LPR Simple...")
        self.init_tesseract()
        self.init_yolo()
        
        # Signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("✅ Sistema LPR Simple listo")
    
    def load_simple_config(self):
        """Configuración simple y directa"""
        default_config = {
            "camera": {
                "rtsp_urls": [
                    "rtsp://admin:admin@192.168.1.101/video2",
                    "rtsp://admin:admin@192.168.1.101/cam/realmonitor?channel=1&subtype=1",
                    "rtsp://admin:admin@192.168.1.101:554/stream1"
                ],
                "fallback_video": "./videos/video2.mp4"
            },
            "processing": {
                "ai_every_frames": 4,
                "confidence_threshold": 0.3,
                "cooldown_seconds": 8.0,
                "ocr_timeout": 0.8
            },
            "display": {
                "show_video": True,
                "scale": 0.4,
                "window_title": "LPR Simple 🎯"
            }
        }
        
        # Cargar o crear configuración
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Configuración cargada: {self.config_file}")
            else:
                self.config = default_config
                self.save_config()
                print(f"📄 Configuración creada: {self.config_file}")
        except Exception as e:
            print(f"⚠️ Error en config: {e}")
            self.config = default_config
        
        # Preparar URLs
        self.video_sources = self.config["camera"]["rtsp_urls"].copy()
        
        # Añadir video local si existe
        fallback = self.config["camera"]["fallback_video"]
        if os.path.exists(fallback):
            self.video_sources.append(fallback)
            print(f"✅ Video local añadido: {fallback}")
        
        self.current_source = 0
        print(f"📋 {len(self.video_sources)} fuentes de video configuradas")
    
    def save_config(self):
        """Guardar configuración"""
        try:
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error guardando config: {e}")
    
    def init_tesseract(self):
        """Inicializar Tesseract simple"""
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Tesseract disponible")
                self.tesseract_ok = True
            else:
                raise Exception("Tesseract no funciona")
        except Exception as e:
            print(f"❌ Tesseract error: {e}")
            self.tesseract_ok = False
    
    def init_yolo(self):
        """Inicializar YOLO simple"""
        try:
            from ultralytics import YOLO
            
            # Buscar modelo
            models = ["license_plate_detector.pt", "yolo11n.pt", "yolov8n.pt"]
            model_path = None
            
            for model in models:
                if Path(model).exists():
                    model_path = model
                    break
            
            if not model_path:
                raise Exception("No se encontró modelo YOLO")
            
            print(f"📦 Cargando {model_path}...")
            self.yolo = YOLO(model_path)
            
            # Warm-up simple
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.yolo(dummy, verbose=False)
            
            print("✅ YOLO listo")
            
        except Exception as e:
            print(f"❌ YOLO error: {e}")
            sys.exit(1)
    
    def create_video_capture(self, source_index=0):
        """Crear captura de video ultra-rápida y efectiva"""
        if source_index >= len(self.video_sources):
            return None
        
        source = self.video_sources[source_index]
        print(f"🚀 Conexión ultra-rápida a: {source}")
        
        try:
            # Video local - instantáneo
            if not source.startswith('rtsp://'):
                if os.path.exists(source):
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        print(f"✅ Video local: {source}")
                        return cap
                return None
            
            # RTSP ultra-optimizado con múltiples pipelines
            pipelines = [
                # Pipeline 1: Mínima latencia absoluta
                (
                    f"rtspsrc location={source} "
                    "latency=0 buffer-mode=1 drop-on-latency=true do-lost=true "
                    "timeout=2000000 tcp-timeout=2000000 "  # 2s timeout
                    "! rtph264depay ! h264parse ! "
                    "avdec_h264 skip-frame=nonref lowres=1 "  # Skip frames, baja resolución
                    "! videoscale ! video/x-raw,width=640,height=480 ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 max-buffers=1 sync=false emit-signals=false",
                    "Ultra-Rápido"
                ),
                
                # Pipeline 2: Conexión agresiva
                (
                    f"rtspsrc location={source} "
                    "latency=0 timeout=1000000 "  # 1s timeout
                    "! rtph264depay ! h264parse ! avdec_h264 "
                    "! videoconvert ! appsink drop=1 max-buffers=1 sync=false",
                    "Agresivo"
                ),
                
                # Pipeline 3: Fallback básico
                (
                    f"rtspsrc location={source} latency=0 "
                    "! rtph264depay ! avdec_h264 ! videoconvert ! "
                    "appsink drop=1 max-buffers=1",
                    "Básico GStreamer"
                )
            ]
            
            # Probar pipelines GStreamer en orden de velocidad
            for pipeline, name in pipelines:
                try:
                    start_time = time.time()
                    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    
                    if cap.isOpened():
                        # Test inmediato de frame
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            connect_time = time.time() - start_time
                            print(f"✅ {name}: {connect_time:.3f}s - {frame.shape}")
                            return cap
                    
                    cap.release()
                    
                except Exception as e:
                    print(f"⚠️ {name} falló: {e}")
                    continue
            
            # Fallback final a FFMPEG optimizado
            try:
                print("🔄 Probando FFMPEG optimizado...")
                start_time = time.time()
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                
                if cap.isOpened():
                    # Configuración ultra-rápida
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
                    cap.set(cv2.CAP_PROP_FPS, 25)        # FPS fijo
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Resolución fija baja
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Test de frame
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        connect_time = time.time() - start_time
                        print(f"✅ FFMPEG Optimizado: {connect_time:.3f}s - {frame.shape}")
                        return cap
                
                cap.release()
                
            except Exception as e:
                print(f"⚠️ FFMPEG optimizado falló: {e}")
            
            # Último recurso: FFMPEG básico
            try:
                print("🔄 Fallback FFMPEG básico...")
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print("✅ FFMPEG Básico conectado")
                        return cap
                cap.release()
            except:
                pass
            
            print("❌ Todos los métodos de conexión fallaron")
            return None
            
        except Exception as e:
            print(f"❌ Error general conectando: {e}")
            return None
    
    def simple_ocr(self, roi):
        """OCR simplificado con Tesseract"""
        if not self.tesseract_ok or roi.size == 0:
            return None, 0.0
        
        try:
            # Preprocesamiento básico
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi
            
            # Redimensionar si es muy pequeño
            h, w = gray.shape
            if h < 32 or w < 96:
                scale = max(32/h, 96/w, 1.0)
                gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
            
            # Mejora de contraste
            gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            
            # Binarización
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Archivo temporal
            temp_file = f'/tmp/lpr_ocr_{int(time.time()*1000)}.png'
            cv2.imwrite(temp_file, binary)
            
            # Tesseract
            cmd = [
                'tesseract', temp_file, 'stdout',
                '--psm', '8', '--oem', '3',
                '-c', 'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=self.config["processing"]["ocr_timeout"])
            
            # Limpiar
            try:
                os.unlink(temp_file)
            except:
                pass
            
            if result.returncode == 0:
                text = result.stdout.strip().upper()
                text = re.sub(r'[^A-Z0-9]', '', text)
                
                if len(text) >= 3 and len(text) <= 8:
                    # Validar formato básico
                    has_letters = any(c.isalpha() for c in text)
                    has_numbers = any(c.isdigit() for c in text)
                    
                    if has_letters and has_numbers:
                        return text, 0.8
            
            return None, 0.0
            
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return None, 0.0
    
    def is_in_cooldown(self, plate_text):
        """Verificar cooldown simple"""
        if plate_text not in self.detection_cooldown:
            return False
        
        last_time = self.detection_cooldown[plate_text]
        cooldown = self.config["processing"]["cooldown_seconds"]
        
        return (time.time() - last_time) < cooldown
    
    def process_frame(self, frame):
        """Procesamiento principal del frame"""
        detections = []
        
        try:
            # YOLO detection
            results = self.yolo(frame, conf=self.config["processing"]["confidence_threshold"], verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Extraer bbox
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Validar bbox
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # ROI
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # OCR
                    plate_text, ocr_conf = self.simple_ocr(roi)
                    
                    if plate_text and not self.is_in_cooldown(plate_text):
                        # Registrar detección
                        self.detection_cooldown[plate_text] = time.time()
                        
                        detection = {
                            'timestamp': datetime.now().isoformat(),
                            'frame': self.frame_count,
                            'plate_text': plate_text,
                            'yolo_conf': confidence,
                            'ocr_conf': ocr_conf,
                            'bbox': [x1, y1, x2, y2]
                        }
                        
                        detections.append(detection)
                        self.detections_count += 1
                        
                        print(f"🎯 PLACA: {plate_text} (YOLO: {confidence:.2f}, OCR: {ocr_conf:.2f}) [{self.detections_count}]")
            
            return detections
            
        except Exception as e:
            print(f"⚠️ Error procesando frame: {e}")
            return []
    
    def auto_cleanup(self):
        """Auto-limpieza simple"""
        current_time = time.time()
        
        # Reset si llegamos al límite
        if self.detections_count >= self.max_detections:
            print(f"🔄 RESET: {self.detections_count} detecciones alcanzadas")
            self.detections_count = 0
            self.ocr_cache.clear()
            self.detection_cooldown.clear()
            self.last_cleanup = current_time
            return
        
        # Limpieza periódica (cada 30 segundos)
        if current_time - self.last_cleanup > 30:
            # Limpiar cooldowns viejos
            old_cooldowns = []
            for plate, last_time in self.detection_cooldown.items():
                if current_time - last_time > 60:  # 1 minuto
                    old_cooldowns.append(plate)
            
            for plate in old_cooldowns:
                del self.detection_cooldown[plate]
            
            # Limpiar cache
            if len(self.ocr_cache) > 20:
                self.ocr_cache.clear()
            
            self.last_cleanup = current_time
            
            if old_cooldowns:
                print(f"🧹 Limpieza: {len(old_cooldowns)} cooldowns viejos")
    
    def capture_thread(self):
        """Thread de captura simple"""
        cap = None
        reconnect_attempts = 0
        max_attempts = 3
        
        while self.running:
            try:
                # Conectar si es necesario
                if cap is None or not cap.isOpened():
                    if reconnect_attempts >= max_attempts:
                        self.current_source = (self.current_source + 1) % len(self.video_sources)
                        reconnect_attempts = 0
                        print(f"🔄 Cambiando a fuente {self.current_source}")
                    
                    if cap:
                        cap.release()
                    
                    cap = self.create_video_capture(self.current_source)
                    reconnect_attempts += 1
                    
                    if cap is None:
                        time.sleep(1)
                        continue
                
                # Capturar frame
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("⚠️ Frame inválido")
                    time.sleep(0.1)
                    continue
                
                # Video local: reiniciar para loop
                source = self.video_sources[self.current_source]
                if not source.startswith('rtsp://'):
                    frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_pos >= total_frames - 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                self.frame_count += 1
                
                # Enviar a procesamiento
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Control FPS
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"❌ Error captura: {e}")
                if cap:
                    cap.release()
                    cap = None
                time.sleep(1)
        
        if cap:
            cap.release()
        print("📹 Captura terminada")
    
    def process_thread(self):
        """Thread de procesamiento simple"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                
                # Procesar solo cada N frames
                ai_every = self.config["processing"]["ai_every_frames"]
                if self.frame_count % ai_every != 0:
                    continue
                
                # Procesar frame
                detections = self.process_frame(frame)
                
                if detections:
                    self.result_queue.put((frame, detections))
                
                # Auto-limpieza
                self.auto_cleanup()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error procesamiento: {e}")
        
        print("🧠 Procesamiento terminado")
    
    def display_thread(self):
        """Thread de display simple"""
        if not self.config["display"]["show_video"]:
            return
        
        window_title = self.config["display"]["window_title"]
        scale = self.config["display"]["scale"]
        
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"⚠️ Display no disponible: {e}")
            return
        
        last_frame = None
        last_detections = []
        
        while self.running:
            try:
                # Obtener frame con detecciones
                try:
                    frame, detections = self.result_queue.get(timeout=0.1)
                    last_frame = frame
                    last_detections = detections
                except queue.Empty:
                    pass
                
                # Obtener frame básico para display
                try:
                    display_frame = self.frame_queue.get(timeout=0.1)
                    if last_detections:
                        # Dibujar últimas detecciones
                        for det in last_detections:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = bbox
                            
                            # Dibujar rectángulo
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Texto
                            text = f"{det['plate_text']} ({det['yolo_conf']:.2f})"
                            cv2.putText(display_frame, text, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                except queue.Empty:
                    if last_frame is not None:
                        display_frame = last_frame.copy()
                    else:
                        continue
                
                # Info overlay
                info = f"Frame: {self.frame_count} | Detecciones: {self.detections_count}"
                cv2.putText(display_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Redimensionar
                h, w = display_frame.shape[:2]
                new_w, new_h = int(w * scale), int(h * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h))
                
                # Mostrar
                cv2.imshow(window_title, display_frame)
                
                # Controles
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("🛑 Salida solicitada")
                    self.stop()
                    break
                elif key == ord('r'):
                    print("🔄 Reset manual")
                    self.detections_count = 0
                    self.ocr_cache.clear()
                    self.detection_cooldown.clear()
                elif key == ord('s'):
                    print(f"📊 Stats: {self.frame_count} frames, {self.detections_count} detecciones")
                
            except Exception as e:
                print(f"❌ Error display: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
        print("🖥️ Display terminado")
    
    def signal_handler(self, signum, frame):
        """Manejador de señales"""
        print(f"\n🛑 Señal {signum} recibida")
        self.stop()
    
    def run(self):
        """Ejecutar sistema principal"""
        if not self.tesseract_ok:
            print("❌ Tesseract no disponible")
            return False
        
        if len(self.video_sources) == 0:
            print("❌ No hay fuentes de video configuradas")
            return False
        
        print("🚀 Iniciando Sistema LPR Simple...")
        print(f"📹 Fuentes: {len(self.video_sources)}")
        print(f"🎯 Reset automático cada {self.max_detections} detecciones")
        print("🎮 Controles: q=salir, r=reset, s=stats")
        
        self.running = True
        
        # Iniciar threads
        threads = [
            threading.Thread(target=self.capture_thread, daemon=True),
            threading.Thread(target=self.process_thread, daemon=True),
            threading.Thread(target=self.display_thread, daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        try:
            # Loop principal simple
            start_time = time.time()
            
            while self.running:
                time.sleep(1)
                
                # Stats cada 30 segundos
                if int(time.time() - start_time) % 30 == 0:
                    runtime = time.time() - start_time
                    fps = self.frame_count / runtime if runtime > 0 else 0
                    print(f"📊 Runtime: {runtime:.0f}s, FPS: {fps:.1f}, Detecciones: {self.detections_count}")
            
        except KeyboardInterrupt:
            print("⏹️ Interrupción de usuario")
        
        finally:
            self.stop()
            
            # Esperar threads
            for t in threads:
                if t.is_alive():
                    t.join(timeout=2)
            
            # Stats finales
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("📊 ESTADÍSTICAS FINALES:")
            print(f"   ⏱️ Tiempo: {total_time:.1f}s")
            print(f"   📹 Frames: {self.frame_count}")
            print(f"   📈 FPS promedio: {avg_fps:.1f}")
            print(f"   🎯 Detecciones: {self.detections_count}")
            print("✅ Sistema LPR Simple terminado")
        
        return True
    
    def stop(self):
        """Detener sistema"""
        self.running = False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema LPR Simple")
    parser.add_argument("--no-display", action="store_true", help="Sin interfaz gráfica")
    parser.add_argument("--ai-every", type=int, default=4, help="IA cada N frames")
    parser.add_argument("--cooldown", type=float, default=8.0, help="Cooldown en segundos")
    parser.add_argument("--max-detections", type=int, default=8, help="Reset cada N detecciones")
    
    args = parser.parse_args()
    
    print("🎯 SISTEMA LPR SIMPLE")
    print("=" * 30)
    print("✅ Código simplificado")
    print("✅ Funcionalidades completas")
    print("✅ Threading básico")
    print("✅ Auto-limpieza")
    print("✅ Múltiples fuentes")
    print("=" * 30)
    
    try:
        # Crear sistema
        lpr = SimpleLPR()
        
        # Aplicar configuraciones
        if args.no_display:
            lpr.config["display"]["show_video"] = False
        
        lpr.config["processing"]["ai_every_frames"] = args.ai_every
        lpr.config["processing"]["cooldown_seconds"] = args.cooldown
        lpr.max_detections = args.max_detections
        
        print(f"⚙️ Configuración:")
        print(f"   🧠 IA cada: {args.ai_every} frames")
        print(f"   ⏱️ Cooldown: {args.cooldown}s")
        print(f"   🔄 Reset cada: {args.max_detections} detecciones")
        print(f"   🖥️ Display: {'No' if args.no_display else 'Sí'}")
        print()
        
        # Ejecutar
        success = lpr.run()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
