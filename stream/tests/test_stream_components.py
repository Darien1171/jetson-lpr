#!/usr/bin/env python3
"""
🧪 Tests para componentes de Stream LPR
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Agregar paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_config():
    """Test configuración"""
    print("🔧 Testing configuración...")
    
    try:
        from stream.config.stream_config import StreamConfig
        
        config = StreamConfig('development')
        
        assert config.mode == 'development'
        assert config.use_file_stream == True
        assert config.db_config['type'] == 'sqlite'
        
        print("✅ Configuración OK")
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_database():
    """Test base de datos MySQL"""
    print("💾 Testing base de datos MySQL...")
    
    try:
        from stream.config.stream_config import StreamConfig
        from stream.database.db_manager import DatabaseManager
        
        config = StreamConfig('development')
        
        # Verificar que la configuración sea MySQL
        if config.db_config['type'] != 'mysql':
            print("⚠️ Configuración no es MySQL")
            return False
        
        print(f"   Conectando a: {config.db_config['host']}:{config.db_config['port']}")
        print(f"   Base de datos: {config.db_config['database']}")
        print(f"   Usuario: {config.db_config['user']}")
        
        try:
            db_manager = DatabaseManager(config.db_config)
        except Exception as e:
            print(f"❌ Error conectando a MySQL: {e}")
            print("💡 Ejecutar: python stream/database/setup_mysql_dev.py")
            return False
        
        # Test inserción
        test_detection = {
            'timestamp': '2024-01-01 12:00:00',
            'plate_text': 'TEST123',
            'confidence': 0.95,
            'plate_score': 0.8,
            'vehicle_bbox': [100, 100, 200, 200],
            'plate_bbox': [120, 150, 180, 170],
            'camera_location': 'test'
        }
        
        success = db_manager.insert_detection(test_detection)
        assert success, "Error insertando detección"
        
        # Test verificación de autorización
        auth_info = db_manager.check_authorized_vehicle('ABC123')
        if not auth_info['registered']:
            print("⚠️ No hay vehículos de prueba - ejecutar setup MySQL")
        
        # Test detecciones recientes
        recent = db_manager.get_recent_detections(24)
        print(f"   Detecciones recientes: {len(recent)}")
        
        db_manager.close()
        print("✅ Base de datos MySQL OK")
        return True
        
    except Exception as e:
        print(f"❌ Error en base de datos MySQL: {e}")
        print("💡 Verificar:")
        print("   1. MySQL está corriendo")
        print("   2. Ejecutar: python stream/database/setup_mysql_dev.py")
        return False

def test_video_handler():
    """Test manejador de video"""
    print("📹 Testing video handler...")
    
    try:
        from stream.config.stream_config import StreamConfig
        from stream.utils.rtsp_handler import RTSPHandler
        
        config = StreamConfig('development')
        
        # Verificar que existe video de prueba
        video_path = Path(config.rtsp_url)
        if not video_path.exists():
            print(f"⚠️ Video de prueba no encontrado: {video_path}")
            print("   Usa cualquier video .mp4 en /videos/")
            return False
        
        handler = RTSPHandler(config)
        
        # Test conexión
        success = handler.start_capture()
        assert success, "Error iniciando captura"
        
        # Test obtener frames
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 10 and (time.time() - start_time) < 5:
            frame, timestamp = handler.get_frame(timeout=1.0)
            
            if frame is not None:
                frame_count += 1
                assert frame.shape == (*config.input_resolution[::-1], 3)
        
        handler.stop_capture()
        
        assert frame_count > 0, "No se obtuvieron frames"
        print(f"✅ Video handler OK ({frame_count} frames)")
        return True
        
    except Exception as e:
        print(f"❌ Error en video handler: {e}")
        return False

def test_lpr_processing():
    """Test procesamiento LPR básico"""
    print("🤖 Testing procesamiento LPR...")
    
    try:
        from main_final import get_all_cached_models
        from plate_validator import ultrafast_ocr_optimized_with_validation
        from stream.config.stream_config import StreamConfig
        
        # Cargar modelos
        models, device, easyocr_reader = get_all_cached_models()
        assert models is not None, "Error cargando modelos YOLO"
        assert easyocr_reader is not None, "Error cargando EasyOCR"
        
        # Test con imagen dummy
        config = StreamConfig('development')
        dummy_frame = np.zeros((*config.input_resolution[::-1], 3), dtype=np.uint8)
        
        coco_model, license_detector, model_name = models
        
        # Test YOLO
        results = coco_model(dummy_frame, imgsz=320, conf=0.5, verbose=False)
        assert results is not None
        
        # Test detector de placas
        plate_results = license_detector(dummy_frame, imgsz=320, conf=0.3, verbose=False)
        assert plate_results is not None
        
        # Test OCR
        dummy_crop = np.ones((50, 150, 3), dtype=np.uint8) * 255
        cv2.putText(dummy_crop, 'ABC123', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        plate_text, confidence = ultrafast_ocr_optimized_with_validation(dummy_crop, easyocr_reader)
        
        print(f"   Modelos: {model_name} en {device}")
        print(f"   OCR test: '{plate_text}' (conf: {confidence})")
        
        print("✅ Procesamiento LPR OK")
        return True
        
    except Exception as e:
        print(f"❌ Error en LPR: {e}")
        return False

def test_stream_integration():
    """Test integración completa del stream"""
    print("🔄 Testing integración stream...")
    
    try:
        from stream.config.stream_config import StreamConfig
        from stream.lpr_stream_main import LPRStream
        
        # Crear instancia de stream
        lpr_stream = LPRStream(mode='development')
        
        # Verificar que se inicializó correctamente
        assert lpr_stream.config is not None
        assert lpr_stream.db_manager is not None
        assert lpr_stream.stream_handler is not None
        
        print("✅ Integración stream OK")
        return True
        
    except Exception as e:
        print(f"❌ Error en integración: {e}")
        return False

def test_performance_monitor():
    """Test del monitor de rendimiento"""
    print("📈 Testing performance monitor...")
    
    try:
        from stream.utils.performance_monitor import PerformanceMonitor
        
        # Crear instancia
        monitor = PerformanceMonitor(log_interval=1)
        
        # Test inicialización
        assert monitor.frame_count == 0
        assert monitor.detection_count == 0
        assert monitor.error_count == 0
        
        # Test contadores
        monitor.increment_frame_count()
        monitor.increment_detection_count()
        monitor.increment_error_count()
        
        assert monitor.frame_count == 1
        assert monitor.detection_count == 1
        assert monitor.error_count == 1
        
        # Test métricas
        metrics = monitor.get_current_metrics()
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'lpr' in metrics
        
        # Test reset
        monitor.reset_counters()
        assert monitor.frame_count == 0
        
        print("   ✅ Performance Monitor OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en performance monitor: {e}")
        return False

def test_database_config():
    """Test de configuración de base de datos"""
    print("🔧 Testing database config...")
    
    try:
        from stream.config.database_config import DatabaseConfig, TableSchemas, SampleData
        
        # Test configuraciones
        dev_config = DatabaseConfig.get_config('development')
        prod_config = DatabaseConfig.get_config('production')
        
        assert dev_config['database'] == 'lpr_development'
        assert prod_config['database'] == 'parqueadero_jetson'
        
        # Test esquemas
        lpr_schema = TableSchemas.get_create_statement('lpr_detections', 'mysql')
        assert 'CREATE TABLE' in lpr_schema
        
        # Test datos de muestra
        dev_vehicles = SampleData.get_sample_vehicles('development')
        assert len(dev_vehicles) > 0
        
        print("   ✅ Database Config OK")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en database config: {e}")
        return False

def run_all_tests():
    """Ejecutar todos los tests"""
    print("🧪 TESTING COMPONENTES LPR STREAM")
    print("=" * 50)
    
    tests = [
        ("Configuración", test_config),
        ("Base de Datos", test_database),
        ("Video Handler", test_video_handler),
        ("Procesamiento LPR", test_lpr_processing),
        ("Integración Stream", test_stream_integration),
        ("Performance Monitor", test_performance_monitor),
        ("Database Config", test_database_config)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error inesperado en {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 RESULTADOS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 TODOS LOS TESTS PASARON")
        print("✅ Sistema listo para testing completo")
        print("\nSiguiente paso:")
        print("   python stream/lpr_stream_main.py --mode development")
    else:
        print("⚠️ ALGUNOS TESTS FALLARON")
        print("🔧 Revisar configuración antes de continuar")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
