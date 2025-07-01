#!/usr/bin/env python3
"""
📹 Manejador RTSP/Video adaptable para desarrollo y producción
"""

import cv2
import time
import threading
import queue
from pathlib import Path

class RTSPHandler:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=config.frame_buffer_size)
        self.reconnect_count = 0
        
    def connect(self):
        """Conectar a source (RTSP o archivo)"""
        source = self.config.rtsp_url
        
        try:
            if self.config.use_file_stream:
                # Modo desarrollo - usar archivo de video
                if not Path(source).exists():
                    raise FileNotFoundError(f"Video no encontrado: {source}")
                
                print(f"📁 Conectando a archivo: {Path(source).name}")
                self.cap = cv2.VideoCapture(source)
                
            else:
                # Modo producción - RTSP real
                print(f"📡 Conectando a RTSP: {source}")
                
                # Probar GStreamer primero (mejor para Jetson)
                gst_pipeline = self._get_gstreamer_pipeline()
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if not self.cap.isOpened():
                    # Fallback a OpenCV directo
                    print("⚠️ GStreamer falló, usando OpenCV directo...")
                    self.cap = cv2.VideoCapture(source)
            
            if self.cap.isOpened():
                # Configurar propiedades
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test de lectura
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Conectado exitosamente - Resolución: {test_frame.shape}")
                    return True
                    
            raise Exception("No se pudo leer frame de prueba")
            
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def _get_gstreamer_pipeline(self):
        """Pipeline GStreamer para RTSP (optimizado para Jetson)"""
        return (
            f"rtspsrc location={self.config.rtsp_url} latency=0 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink"
        )
    
    def start_capture(self):
        """Iniciar captura en thread separado"""
        if not self.connect():
            return False
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("🎬 Captura iniciada")
        return True
    
    def _capture_loop(self):
        """Loop principal de captura"""
        frame_count = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.config.use_file_stream and self.config.loop_video:
                        # Reiniciar video para loop continuo
                        print("🔄 Reiniciando video para loop...")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("⚠️ Frame perdido, reintentando conexión...")
                        if self._reconnect():
                            continue
                        else:
                            break
                
                # Redimensionar si es necesario
                target_size = self.config.input_resolution
                if frame.shape[:2] != target_size[::-1]:
                    frame = cv2.resize(frame, target_size)
                
                # Agregar timestamp
                timestamp = time.time()
                
                # Gestionar queue (no bloqueante)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, timestamp))
                else:
                    # Descartar frame más viejo
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put((frame, timestamp))
                    except queue.Empty:
                        pass
                
                frame_count += 1
                
                # Control de velocidad para archivos de video
                if self.config.use_file_stream:
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                print(f"❌ Error en captura: {e}")
                time.sleep(0.1)
        
        print(f"📹 Captura detenida. Frames procesados: {frame_count}")
    
    def _reconnect(self):
        """Intentar reconexión"""
        if self.reconnect_count >= self.config.reconnect_attempts:
            print(f"❌ Máximo de reconexiones alcanzado ({self.config.reconnect_attempts})")
            return False
        
        self.reconnect_count += 1
        print(f"🔄 Reconectando... (intento {self.reconnect_count})")
        
        if self.cap:
            self.cap.release()
        
        time.sleep(2)
        return self.connect()
    
    def get_frame(self, timeout=1.0):
        """Obtener siguiente frame disponible"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
    
    def stop_capture(self):
        """Detener captura"""
        self.running = False
        
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        print("⏹️ Captura detenida")
    
    def get_stats(self):
        """Obtener estadísticas de captura"""
        return {
            'queue_size': self.frame_queue.qsize(),
            'max_queue_size': self.config.frame_buffer_size,
            'reconnect_count': self.reconnect_count,
            'running': self.running
        }
