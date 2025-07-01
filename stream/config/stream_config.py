#!/usr/bin/env python3
"""
📹 Configuración para usar cámara RTSP real
Cambia entre video local y cámara IP fácilmente
"""

import os
import torch
from pathlib import Path

class StreamConfig:
    def __init__(self, mode='development', use_camera=False, camera_url=None):
        self.mode = mode
        self.use_camera = use_camera  # 🆕 Nuevo parámetro
        self.camera_url = camera_url  # 🆕 URL de cámara
        
        self.setup_paths()
        self.setup_hardware()
        self.setup_stream()
        self.setup_processing()
        self.setup_database()
    
    def setup_paths(self):
        """Configurar rutas del proyecto"""
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "stream" / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def setup_hardware(self):
        """Configuración de hardware (adaptable PC/Jetson)"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.mode == 'development':
            # Configuración para PC de desarrollo
            self.gpu_memory_fraction = 0.6
            self.max_batch_size = 2
            self.use_half_precision = False
            self.threads = 6
        else:
            # Configuración para Jetson (será usado en Fase 2)
            self.gpu_memory_fraction = 0.8
            self.max_batch_size = 1
            self.use_half_precision = True
            self.threads = 4
    
    def setup_stream(self):
        """Configuración de streaming - NUEVA LÓGICA PARA CÁMARA"""
        
        # 🎯 NUEVA LÓGICA: Priorizar cámara si está configurada
        if self.use_camera and self.camera_url:
            # MODO CÁMARA REAL
            print(f"📹 Configurado para cámara IP: {self.camera_url}")
            self.rtsp_url = self.camera_url
            self.use_file_stream = False
            self.loop_video = False
            
        elif self.mode == 'development':
            # MODO VIDEO LOCAL (como antes)
            print(f"📁 Configurado para video local")
            self.rtsp_url = str(self.project_root / "videos" / "video2.mp4")
            self.use_file_stream = True
            self.loop_video = True
            
        else:
            # MODO JETSON PRODUCCIÓN
            self.camera_ip = "192.168.1.50"
            self.rtsp_url = f"rtsp://admin:password@{self.camera_ip}:554/stream1"
            self.use_file_stream = False
            self.loop_video = False
        
        self.rtsp_timeout = 30
        self.reconnect_attempts = 5
        self.frame_buffer_size = 10
        
    def setup_processing(self):
        """Configuración de procesamiento LPR"""
        # Resolución optimizada
        if self.mode == 'development':
            self.input_resolution = (640, 480)  # PC puede manejar más
        else:
            self.input_resolution = (416, 320)  # Jetson optimizado
        
        # Parámetros de detección
        self.confidence_threshold = 0.3
        self.nms_threshold = 0.5
        self.max_detections = 10
        self.plate_confidence_min = 0.2
    
    def setup_database(self):
        """Configuración de base de datos - MySQL en ambos entornos"""
        if self.mode == 'development':
            # Para PC de desarrollo - MySQL local
            self.db_config = {
                'type': 'mysql',
                'host': 'localhost',
                'port': 3306,
                'database': 'lpr_development',
                'user': 'lpr_dev_user',
                'password': 'lpr_dev_pass',
                'charset': 'utf8mb4'
            }
        else:
            # Para Jetson - MySQL producción
            self.db_config = {
                'type': 'mysql',
                'host': 'localhost',
                'port': 3306,
                'database': 'parqueadero_jetson',
                'user': 'lpr_user',
                'password': 'lpr_password',
                'charset': 'utf8mb4'
            }
    
    def get_models_paths(self):
        """Obtener rutas de modelos"""
        return {
            'yolo11': self.models_dir / 'yolo11n.pt',
            'yolo8': self.models_dir / 'yolov8n.pt',
            'license_detector': self.models_dir / 'license_plate_detector.pt'
        }
    
    def setup_cuda_optimizations(self):
        """Configurar optimizaciones CUDA"""
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
                print(f"🎮 CUDA optimizado: {self.device} ({self.gpu_memory_fraction*100}% memoria)")
