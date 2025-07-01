#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO Y OPTIMIZACIÓN DE CONEXIÓN DE CÁMARA
===================================================
Encuentra la mejor configuración para minimizar:
- Retraso de conexión
- Pérdida de paquetes H.264
- Errores de decodificación

Autor: Diagnóstico de Cámara
Fecha: 2025-07-01
Versión: 1.0
"""

import cv2
import time
import threading
import queue
import subprocess
import re
import json
import os
from datetime import datetime
import numpy as np

class CameraConnectionTester:
    """Tester completo de conexión de cámara"""
    
    def __init__(self, camera_ip="192.168.1.101", user="admin", password="admin"):
        self.camera_ip = camera_ip
        self.user = user
        self.password = password
        
        # Generar todas las URLs posibles
        self.generate_test_urls()
        
        # Resultados
        self.test_results = []
        
    def generate_test_urls(self):
        """Generar todas las URLs posibles para probar"""
        base_url = f"rtsp://{self.user}:{self.password}@{self.camera_ip}"
        
        self.test_urls = [
            # URLs con diferentes calidades y protocolos
            f"{base_url}/cam/realmonitor?channel=1&subtype=0",  # Alta calidad
            f"{base_url}/cam/realmonitor?channel=1&subtype=1",  # Media calidad
            f"{base_url}/cam/realmonitor?channel=1&subtype=2",  # Baja calidad
            f"{base_url}/cam/realmonitor?channel=1&subtype=0&proto=tcp",  # TCP alta
            f"{base_url}/cam/realmonitor?channel=1&subtype=1&proto=tcp",  # TCP media
            f"{base_url}/cam/realmonitor?channel=1&subtype=2&proto=tcp",  # TCP baja
            f"{base_url}/cam/realmonitor?channel=1&subtype=0&proto=udp",  # UDP alta
            f"{base_url}/cam/realmonitor?channel=1&subtype=1&proto=udp",  # UDP media
            f"{base_url}/cam/realmonitor?channel=1&subtype=2&proto=udp",  # UDP baja
            
            # URLs alternativas comunes
            f"{base_url}:554/stream1",
            f"{base_url}:554/stream2", 
            f"{base_url}:554/main",
            f"{base_url}:554/sub",
            f"{base_url}/h264Preview_01_main",
            f"{base_url}/h264Preview_01_sub",
            f"{base_url}/live/0/MAIN",
            f"{base_url}/live/0/SUB",
            f"{base_url}/cam/1/h264",
            f"{base_url}/video1",
            f"{base_url}/video2",
            
            # URLs con parámetros adicionales
            f"{base_url}/cam/realmonitor?channel=1&subtype=1&authbasic=YWRtaW46YWRtaW4=",
            f"{base_url}/videoMain",
            f"{base_url}/videoSub",
        ]
        
        print(f"📋 {len(self.test_urls)} URLs generadas para testing")
    
    def test_network_connectivity(self):
        """Test básico de conectividad de red"""
        print("🌐 Testing conectividad de red...")
        
        # Ping test
        try:
            result = subprocess.run(['ping', '-c', '3', self.camera_ip], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Extraer estadísticas de ping
                output = result.stdout
                avg_time = None
                packet_loss = None
                
                # Buscar tiempo promedio
                avg_match = re.search(r'avg = ([\d.]+)', output)
                if avg_match:
                    avg_time = float(avg_match.group(1))
                
                # Buscar pérdida de paquetes
                loss_match = re.search(r'(\d+)% packet loss', output)
                if loss_match:
                    packet_loss = int(loss_match.group(1))
                
                print(f"✅ Ping exitoso - Tiempo promedio: {avg_time}ms, Pérdida: {packet_loss}%")
                return True, avg_time, packet_loss
            else:
                print("❌ Ping falló")
                return False, None, None
                
        except Exception as e:
            print(f"❌ Error en ping: {e}")
            return False, None, None
    
    def test_port_connectivity(self):
        """Test de puertos específicos"""
        print("🔍 Testing puertos de cámara...")
        
        common_ports = [80, 554, 8000, 8080, 8899, 37777]
        open_ports = []
        
        for port in common_ports:
            try:
                result = subprocess.run(['nc', '-z', '-w', '2', self.camera_ip, str(port)], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    open_ports.append(port)
                    print(f"   ✅ Puerto {port} abierto")
                else:
                    print(f"   ❌ Puerto {port} cerrado")
            except:
                print(f"   ⚠️ Puerto {port} - error en test")
        
        return open_ports
    
    def create_optimized_pipelines(self, url):
        """Crear pipelines optimizados para diferentes escenarios"""
        pipelines = [
            # Pipeline 1: Ultra baja latencia
            {
                'name': 'Ultra-Low Latency',
                'pipeline': (
                    f"rtspsrc location={url} "
                    "latency=0 buffer-mode=1 drop-on-latency=true do-lost=true "
                    "timeout=1000000 tcp-timeout=1000000 "
                    "! rtph264depay ! h264parse ! "
                    "avdec_h264 skip-frame=nonref max-threads=1 "
                    "! videoscale ! video/x-raw,width=640,height=480 ! "
                    "videoconvert ! video/x-raw,format=BGR ! "
                    "appsink drop=1 max-buffers=1 sync=false emit-signals=false"
                ),
                'backend': cv2.CAP_GSTREAMER
            },
            
            # Pipeline 2: Anti-pérdida de paquetes
            {
                'name': 'Anti-Packet-Loss',
                'pipeline': (
                    f"rtspsrc location={url} "
                    "latency=200 buffer-mode=4 "
                    "do-retransmission=true retransmission-time=100 "
                    "! rtph264depay ! h264parse ! "
                    "avdec_h264 "
                    "! videoconvert ! "
                    "appsink drop=0 max-buffers=3 sync=false"
                ),
                'backend': cv2.CAP_GSTREAMER
            },
            
            # Pipeline 3: Balanced
            {
                'name': 'Balanced',
                'pipeline': (
                    f"rtspsrc location={url} "
                    "latency=100 buffer-mode=2 drop-on-latency=false "
                    "! rtph264depay ! h264parse ! "
                    "avdec_h264 "
                    "! videoconvert ! "
                    "appsink drop=1 max-buffers=2 sync=false"
                ),
                'backend': cv2.CAP_GSTREAMER
            },
            
            # Pipeline 4: FFMPEG optimizado
            {
                'name': 'FFMPEG-Optimized',
                'pipeline': url,
                'backend': cv2.CAP_FFMPEG,
                'options': {
                    cv2.CAP_PROP_BUFFERSIZE: 1,
                    cv2.CAP_PROP_FPS: 25,
                    cv2.CAP_PROP_FRAME_WIDTH: 640,
                    cv2.CAP_PROP_FRAME_HEIGHT: 480
                }
            },
            
            # Pipeline 5: FFMPEG básico
            {
                'name': 'FFMPEG-Basic',
                'pipeline': url,
                'backend': cv2.CAP_FFMPEG,
                'options': {
                    cv2.CAP_PROP_BUFFERSIZE: 1
                }
            }
        ]
        
        return pipelines
    
    def test_single_connection(self, url, pipeline_config, test_duration=10):
        """Test detallado de una conexión específica"""
        print(f"🔄 Testing: {pipeline_config['name']} con {url}")
        
        result = {
            'url': url,
            'pipeline_name': pipeline_config['name'],
            'connect_time': None,
            'fps': 0,
            'frame_count': 0,
            'h264_errors': 0,
            'connection_successful': False,
            'avg_frame_time': 0,
            'resolution': None,
            'stability_score': 0
        }
        
        try:
            # Medir tiempo de conexión
            start_connect = time.time()
            
            cap = cv2.VideoCapture(pipeline_config['pipeline'], pipeline_config['backend'])
            
            # Aplicar opciones si existen
            if 'options' in pipeline_config:
                for prop, value in pipeline_config['options'].items():
                    cap.set(prop, value)
            
            if not cap.isOpened():
                cap.release()
                return result
            
            # Test inicial de frame
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                cap.release()
                return result
            
            connect_time = time.time() - start_connect
            result['connect_time'] = connect_time
            result['connection_successful'] = True
            result['resolution'] = frame.shape
            
            print(f"   ✅ Conectado en {connect_time:.3f}s - Resolución: {frame.shape}")
            
            # Test de estabilidad y rendimiento
            frame_times = []
            error_count = 0
            successful_frames = 0
            
            # Capturar stderr para contar errores H.264
            error_queue = queue.Queue()
            
            start_test = time.time()
            while time.time() - start_test < test_duration:
                frame_start = time.time()
                
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    frame_time = time.time() - frame_start
                    frame_times.append(frame_time)
                    successful_frames += 1
                else:
                    error_count += 1
                
                # Pequeña pausa para no saturar
                time.sleep(0.01)
            
            cap.release()
            
            # Calcular métricas
            test_duration_actual = time.time() - start_test
            result['fps'] = successful_frames / test_duration_actual
            result['frame_count'] = successful_frames
            result['h264_errors'] = error_count
            result['avg_frame_time'] = sum(frame_times) / len(frame_times) if frame_times else 0
            
            # Score de estabilidad (0-100)
            fps_score = min(result['fps'] / 25 * 50, 50)  # Max 50 puntos por FPS
            error_score = max(50 - (error_count / successful_frames * 100), 0) if successful_frames > 0 else 0
            result['stability_score'] = fps_score + error_score
            
            print(f"   📊 FPS: {result['fps']:.1f}, Errores: {error_count}, Score: {result['stability_score']:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if 'cap' in locals():
                cap.release()
        
        return result
    
    def find_best_connection(self):
        """Encontrar la mejor conexión probando todas las combinaciones"""
        print("🚀 BUSCANDO LA MEJOR CONEXIÓN DE CÁMARA")
        print("=" * 60)
        
        # Test de conectividad básica
        network_ok, ping_time, packet_loss = self.test_network_connectivity()
        if not network_ok:
            print("❌ Problema de conectividad básica")
            return None
        
        # Test de puertos
        open_ports = self.test_port_connectivity()
        print(f"📡 Puertos abiertos: {open_ports}")
        print()
        
        # Test todas las URLs con todos los pipelines
        print("🔍 Testing todas las combinaciones URL + Pipeline...")
        print()
        
        best_results = []
        total_tests = 0
        
        for i, url in enumerate(self.test_urls):
            print(f"📡 URL {i+1}/{len(self.test_urls)}: {url.split('@')[1] if '@' in url else url}")
            
            pipelines = self.create_optimized_pipelines(url)
            url_best = None
            
            for pipeline in pipelines:
                total_tests += 1
                result = self.test_single_connection(url, pipeline, test_duration=5)
                
                if result['connection_successful']:
                    self.test_results.append(result)
                    
                    if url_best is None or result['stability_score'] > url_best['stability_score']:
                        url_best = result
            
            if url_best:
                best_results.append(url_best)
                print(f"   🏆 Mejor para esta URL: {url_best['pipeline_name']} "
                      f"(Score: {url_best['stability_score']:.1f})")
            else:
                print("   ❌ Ningún pipeline funcionó para esta URL")
            
            print()
        
        # Encontrar la mejor configuración global
        if best_results:
            global_best = max(best_results, key=lambda x: x['stability_score'])
            
            print("🏆 MEJOR CONFIGURACIÓN ENCONTRADA:")
            print("=" * 50)
            print(f"📡 URL: {global_best['url']}")
            print(f"🔧 Pipeline: {global_best['pipeline_name']}")
            print(f"⏱️ Tiempo conexión: {global_best['connect_time']:.3f}s")
            print(f"📈 FPS: {global_best['fps']:.1f}")
            print(f"📊 Score estabilidad: {global_best['stability_score']:.1f}/100")
            print(f"❌ Errores H.264: {global_best['h264_errors']}")
            print(f"📺 Resolución: {global_best['resolution']}")
            
            return global_best
        else:
            print("❌ No se encontró ninguna configuración funcional")
            return None
    
    def generate_optimized_config(self, best_config):
        """Generar configuración optimizada para simple_lpr.py"""
        if not best_config:
            return None
        
        # Determinar el tipo de configuración basado en el pipeline
        url = best_config['url']
        pipeline_name = best_config['pipeline_name']
        
        if pipeline_name == 'Ultra-Low Latency':
            config_type = "ultra_low_latency"
        elif pipeline_name == 'Anti-Packet-Loss':
            config_type = "anti_packet_loss"
        elif pipeline_name == 'Balanced':
            config_type = "balanced"
        else:
            config_type = "ffmpeg_optimized"
        
        optimized_config = {
            "camera": {
                "primary_url": url,
                "rtsp_urls": [url],  # Solo la mejor URL
                "fallback_video": "./videos/video2.mp4",
                "config_type": config_type,
                "pipeline_name": pipeline_name
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
                "window_title": f"LPR Simple 🎯 ({pipeline_name})"
            },
            "optimization_results": {
                "connect_time": best_config['connect_time'],
                "expected_fps": best_config['fps'],
                "stability_score": best_config['stability_score'],
                "h264_errors": best_config['h264_errors'],
                "resolution": best_config['resolution'],
                "test_date": datetime.now().isoformat()
            }
        }
        
        return optimized_config
    
    def save_results(self, best_config):
        """Guardar resultados completos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar resultados completos
        results_file = f"camera_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'test_summary': {
                    'total_urls_tested': len(self.test_urls),
                    'total_pipelines_tested': len(self.test_results),
                    'successful_connections': len([r for r in self.test_results if r['connection_successful']]),
                    'best_config': best_config,
                    'test_date': datetime.now().isoformat()
                },
                'all_results': self.test_results
            }, f, indent=2)
        
        print(f"📄 Resultados completos guardados: {results_file}")
        
        # Guardar configuración optimizada
        if best_config:
            optimized_config = self.generate_optimized_config(best_config)
            config_file = "config/optimized_simple_lpr.json"
            
            os.makedirs("config", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(optimized_config, f, indent=2)
            
            print(f"⚙️ Configuración optimizada guardada: {config_file}")
            print(f"💡 Usar con: python simple_lpr.py --config {config_file}")
        
        return results_file
    
    def run_full_test(self):
        """Ejecutar test completo"""
        print("🔍 DIAGNÓSTICO COMPLETO DE CÁMARA")
        print("=" * 50)
        print(f"📡 Cámara: {self.camera_ip}")
        print(f"👤 Usuario: {self.user}")
        print()
        
        start_time = time.time()
        
        # Ejecutar tests
        best_config = self.find_best_connection()
        
        # Guardar resultados
        results_file = self.save_results(best_config)
        
        total_time = time.time() - start_time
        
        print()
        print("📊 RESUMEN FINAL:")
        print("=" * 30)
        print(f"⏱️ Tiempo total de test: {total_time:.1f}s")
        print(f"📊 Configuraciones probadas: {len(self.test_results)}")
        print(f"✅ Conexiones exitosas: {len([r for r in self.test_results if r['connection_successful']])}")
        
        if best_config:
            print(f"🏆 Mejor configuración: {best_config['pipeline_name']}")
            print(f"⚡ Conexión en: {best_config['connect_time']:.3f}s")
            print(f"📈 FPS esperado: {best_config['fps']:.1f}")
            print()
            print("🚀 PRÓXIMOS PASOS:")
            print("1. Usar la configuración optimizada generada")
            print("2. Si aún hay problemas, verificar configuración de cámara")
            print("3. Considerar cambiar calidad/bitrate en la cámara")
        else:
            print("❌ No se encontró configuración funcional")
            print("🔧 SOLUCIONES SUGERIDAS:")
            print("1. Verificar IP y credenciales de cámara")
            print("2. Verificar que la cámara esté encendida")
            print("3. Probar desde navegador: http://192.168.1.101")
            print("4. Verificar configuración de red")
        
        return best_config

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnóstico de conexión de cámara")
    parser.add_argument("--ip", default="192.168.1.101", help="IP de la cámara")
    parser.add_argument("--user", default="admin", help="Usuario")
    parser.add_argument("--password", default="admin", help="Contraseña")
    parser.add_argument("--quick", action="store_true", help="Test rápido (menos URLs)")
    
    args = parser.parse_args()
    
    print("🎯 CAMERA CONNECTION TESTER")
    print("=" * 40)
    print("Encuentra la configuración óptima para:")
    print("✅ Mínimo retraso de conexión")
    print("✅ Máximo FPS estable")
    print("✅ Mínimos errores H.264")
    print("=" * 40)
    print()
    
    try:
        tester = CameraConnectionTester(args.ip, args.user, args.password)
        
        if args.quick:
            # Test rápido: solo las URLs más comunes
            tester.test_urls = tester.test_urls[:6]
            print("⚡ Modo rápido: probando solo 6 URLs principales")
        
        best_config = tester.run_full_test()
        
        return 0 if best_config else 1
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrumpido por usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
