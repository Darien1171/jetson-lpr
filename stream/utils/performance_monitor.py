#!/usr/bin/env python3
"""
📊 Performance Monitor para LPR Stream
Monitoreo de rendimiento y métricas del sistema
"""

import time
import psutil
import threading
import queue
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.running = False
        self.metrics = {}
        self.metrics_queue = queue.Queue()
        
        # Contadores
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        
        # Tiempos
        self.start_time = time.time()
        self.last_reset = time.time()
        
    def start_monitoring(self):
        """Iniciar monitoreo en thread separado"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance Monitor iniciado")
    
    def stop_monitoring(self):
        """Detener monitoreo"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        print("📊 Performance Monitor detenido")
    
    def _monitor_loop(self):
        """Loop principal de monitoreo"""
        while self.running:
            try:
                # Recopilar métricas
                metrics = self._collect_metrics()
                
                # Guardar en queue
                self.metrics_queue.put(metrics)
                
                # Log cada intervalo
                self._log_metrics(metrics)
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"❌ Error en monitor: {e}")
                time.sleep(1)
    
    def _collect_metrics(self):
        """Recopilar métricas del sistema"""
        current_time = time.time()
        elapsed = current_time - self.last_reset
        
        # Métricas del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Métricas de red
        try:
            net_io = psutil.net_io_counters()
            network_sent = net_io.bytes_sent
            network_recv = net_io.bytes_recv
        except:
            network_sent = 0
            network_recv = 0
        
        # Métricas específicas LPR
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        detection_rate = self.detection_count / elapsed if elapsed > 0 else 0
        error_rate = self.error_count / elapsed if elapsed > 0 else 0
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'uptime': current_time - self.start_time,
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used // 1024 // 1024,
                'memory_available_mb': memory.available // 1024 // 1024,
                'network_sent_mb': network_sent // 1024 // 1024,
                'network_recv_mb': network_recv // 1024 // 1024
            },
            'lpr': {
                'frames_processed': self.frame_count,
                'detections_total': self.detection_count,
                'errors_total': self.error_count,
                'fps': fps,
                'detection_rate': detection_rate,
                'error_rate': error_rate
            }
        }
        
        return metrics
    
    def _log_metrics(self, metrics):
        """Log de métricas"""
        lpr_metrics = metrics['lpr']
        sys_metrics = metrics['system']
        
        print(f"📊 [{datetime.now().strftime('%H:%M:%S')}] "
              f"FPS: {lpr_metrics['fps']:.1f} | "
              f"Detecciones: {lpr_metrics['detections_total']} | "
              f"CPU: {sys_metrics['cpu_percent']:.1f}% | "
              f"RAM: {sys_metrics['memory_percent']:.1f}%")
    
    def increment_frame_count(self):
        """Incrementar contador de frames"""
        self.frame_count += 1
    
    def increment_detection_count(self):
        """Incrementar contador de detecciones"""
        self.detection_count += 1
    
    def increment_error_count(self):
        """Incrementar contador de errores"""
        self.error_count += 1
    
    def reset_counters(self):
        """Resetear contadores"""
        self.frame_count = 0
        self.detection_count = 0
        self.error_count = 0
        self.last_reset = time.time()
        print("📊 Contadores reseteados")
    
    def get_current_metrics(self):
        """Obtener métricas actuales"""
        return self._collect_metrics()
    
    def get_metrics_history(self, count=10):
        """Obtener historial de métricas"""
        history = []
        temp_queue = queue.Queue()
        
        # Extraer hasta 'count' métricas del queue
        while not self.metrics_queue.empty() and len(history) < count:
            try:
                metric = self.metrics_queue.get_nowait()
                history.append(metric)
                temp_queue.put(metric)
            except queue.Empty:
                break
        
        # Devolver métricas al queue
        while not temp_queue.empty():
            self.metrics_queue.put(temp_queue.get())
        
        return history
    
    def save_metrics_to_file(self, filepath):
        """Guardar métricas a archivo JSON"""
        try:
            current_metrics = self.get_current_metrics()
            history = self.get_metrics_history(50)
            
            data = {
                'current': current_metrics,
                'history': history,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"📄 Métricas guardadas en: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error guardando métricas: {e}")
            return False
    
    def get_performance_summary(self):
        """Obtener resumen de rendimiento"""
        current = self.get_current_metrics()
        
        uptime_hours = current['uptime'] / 3600
        
        summary = {
            'uptime_hours': uptime_hours,
            'avg_fps': current['lpr']['fps'],
            'total_frames': current['lpr']['frames_processed'],
            'total_detections': current['lpr']['detections_total'],
            'detection_ratio': (current['lpr']['detections_total'] / 
                              max(current['lpr']['frames_processed'], 1)) * 100,
            'avg_cpu': current['system']['cpu_percent'],
            'avg_memory': current['system']['memory_percent'],
            'errors_total': current['lpr']['errors_total']
        }
        
        return summary
    
    def print_performance_report(self):
        """Imprimir reporte detallado de rendimiento"""
        summary = self.get_performance_summary()
        
        print("\n📊 REPORTE DE RENDIMIENTO LPR")
        print("=" * 40)
        print(f"⏱️  Tiempo activo: {summary['uptime_hours']:.1f} horas")
        print(f"🎬 FPS promedio: {summary['avg_fps']:.1f}")
        print(f"📹 Frames totales: {summary['total_frames']}")
        print(f"🎯 Detecciones: {summary['total_detections']}")
        print(f"📊 Ratio detección: {summary['detection_ratio']:.1f}%")
        print(f"💻 CPU promedio: {summary['avg_cpu']:.1f}%")
        print(f"💾 RAM promedio: {summary['avg_memory']:.1f}%")
        print(f"❌ Errores: {summary['errors_total']}")
        print("=" * 40)

# Instancia global para facilitar uso
performance_monitor = PerformanceMonitor()

def start_performance_monitoring(interval=10):
    """Iniciar monitoreo global"""
    performance_monitor.log_interval = interval
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Detener monitoreo global"""
    performance_monitor.stop_monitoring()

def log_frame_processed():
    """Log frame procesado"""
    performance_monitor.increment_frame_count()

def log_detection_found():
    """Log detección encontrada"""
    performance_monitor.increment_detection_count()

def log_error_occurred():
    """Log error ocurrido"""
    performance_monitor.increment_error_count()

def get_performance_report():
    """Obtener reporte de rendimiento"""
    return performance_monitor.get_performance_summary()

def print_performance_status():
    """Imprimir estado actual"""
    performance_monitor.print_performance_report()

if __name__ == "__main__":
    # Test del monitor
    print("🧪 Testing Performance Monitor...")
    
    monitor = PerformanceMonitor(log_interval=2)
    monitor.start_monitoring()
    
    # Simular actividad
    for i in range(10):
        monitor.increment_frame_count()
        if i % 3 == 0:
            monitor.increment_detection_count()
        if i % 7 == 0:
            monitor.increment_error_count()
        time.sleep(0.5)
    
    # Mostrar reporte
    monitor.print_performance_report()
    
    # Guardar métricas
    monitor.save_metrics_to_file('test_metrics.json')
    
    monitor.stop_monitoring()
    print("✅ Test completado")
