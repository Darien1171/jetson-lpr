#!/usr/bin/env python3
"""
*** MAIN_FINAL - VERSION DEFINITIVA ULTRA-RAPIDA (Compatible Windows) ***
Cache global para YOLO + EasyOCR = < 2 segundos objetivo
SOPORTA: Imagenes y Videos
"""

import cv2
import torch
import sys
import os
import time
from pathlib import Path
import numpy as np
# FIX para PIL.Image.ANTIALIAS deprecado en Pillow >= 10.0.0
import warnings
from PIL import Image
from plate_validator import ultrafast_ocr_optimized_with_validation as ultrafast_ocr_optimized

# Monkey patch para evitar error de ANTIALIAS
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# También suprimir warnings relacionados con PIL
warnings.filterwarnings("ignore", category=DeprecationWarning, module="PIL")
warnings.filterwarnings("ignore", message=".*ANTIALIAS.*")

# CACHE GLOBAL COMPLETO (YOLO + EasyOCR)
_cached_models = None
_cached_device = None
_cached_easyocr = None

def get_all_cached_models():
    """Cargar TODOS los modelos una sola vez y mantenerlos en memoria"""
    global _cached_models, _cached_device, _cached_easyocr
    
    if _cached_models is not None and _cached_easyocr is not None:
        return _cached_models, _cached_device, _cached_easyocr
    
    print("*** INICIALIZANDO SISTEMA COMPLETO (una sola vez) ***")
    start_time = time.time()
    
    try:
        from ultralytics import YOLO
        import easyocr
        
        # Detectar dispositivo
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _cached_device = device
        
        script_dir = Path(__file__).parent.absolute()
        
        # ===== CARGAR MODELOS YOLO =====
        print("Cargando modelos YOLO...")
        yolo11_path = script_dir / 'yolo11n.pt'
        yolo8_path = script_dir / 'yolov8n.pt'
        
        if yolo11_path.exists():
            coco_model = YOLO(str(yolo11_path))
            model_name = "YOLOv11n"
        elif yolo8_path.exists():
            coco_model = YOLO(str(yolo8_path))
            model_name = "YOLOv8n"
        else:
            raise Exception("No se encontro ningun modelo YOLO")
        
        # Cargar detector de placas
        license_detector_path = script_dir / 'license_plate_detector.pt'
        if license_detector_path.exists():
            license_plate_detector = YOLO(str(license_detector_path))
        else:
            license_plate_detector = coco_model
        
        # Mover a GPU
        coco_model.to(device)
        license_plate_detector.to(device)
        
        # ===== CARGAR EASYOCR =====
        print("Cargando EasyOCR optimizado...")
        easyocr_reader = easyocr.Reader(
            ['en'], 
            gpu=True if device == 'cuda' else False,
            verbose=False,
            quantize=True,
            download_enabled=False
        )
        _cached_easyocr = easyocr_reader
        
        # ===== WARM-UP COMPLETO =====
        print("Calentando sistema completo...")
        dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
        
        # Warm-up YOLO
        coco_model(dummy_image, imgsz=224, conf=0.5, verbose=False)
        license_plate_detector(dummy_image, imgsz=224, conf=0.5, verbose=False)
        
        # Warm-up EasyOCR
        dummy_gray = np.zeros((50, 100), dtype=np.uint8)
        try:
            easyocr_reader.readtext(dummy_gray, allowlist='ABC123')
        except:
            pass
        
        _cached_models = (coco_model, license_plate_detector, model_name)
        
        init_time = time.time() - start_time
        print(f"Sistema completo listo: {model_name} + EasyOCR ({init_time:.2f}s)")
        
        return _cached_models, _cached_device, _cached_easyocr
        
    except Exception as e:
        print(f"Error cargando sistema: {e}")
        return None, None, None

def detect_input_type(input_path):
    """Detecta si el input es una imagen o un video"""
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe")
        return None
    
    extension = Path(input_path).suffix.lower()
    
    if extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return 'image'
    elif extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
        return 'video'
    else:
        print(f"Error: Formato no soportado {extension}")
        print("Formatos soportados: imagenes (.jpg, .jpeg, .png) y videos (.mp4, .avi, .mov)")
        return None

def ultrafast_ocr_optimized(image_crop, reader):
    """OCR ultra-optimizado usando reader cacheado"""
    try:
        # Preprocesamiento minimo
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()
        
        # Solo redimensionar si es muy pequena
        height, width = gray.shape
        if height < 40 or width < 120:
            scale = max(40/height, 120/width, 1.0)
            scale = min(scale, 3.0)  # Limitar escala
            new_h, new_w = int(height * scale), int(width * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # OCR directo con configuracion ultra-rapida
        results = reader.readtext(
            gray,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            paragraph=False,
            width_ths=0.8,
            height_ths=0.8,
            detail=1
        )
        
        # Procesar resultados
        best_text = None
        best_confidence = 0.0
        
        for bbox, text, confidence in results:
            # Limpiar texto
            text = str(text).upper().replace(' ', '').replace('-', '').replace('.', '')
            text = ''.join(c for c in text if c.isalnum())
            
            if len(text) >= 3 and len(text) <= 8 and confidence > 0.1:
                # Verificar formato basico
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                
                if has_letters and has_numbers and confidence > best_confidence:
                    best_text = text
                    best_confidence = confidence
        
        return best_text, best_confidence
        
    except Exception as e:
        print(f"Error en OCR: {e}")
        return None, None

def get_car_assignment(license_plate, vehicle_track_ids):
    """Asignar placa a vehiculo - version simplificada"""
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def process_single_image(image_path, models, device, easyocr_reader):
    """Procesamiento ultra-rapido de imagen individual"""
    
    coco_model, license_plate_detector, model_name = models
    
    # Cargar imagen
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error cargando imagen: {image_path}")
        return None
    
    print(f"Imagen: {frame.shape}")
    
    # PASO 1: Deteccion ultra-rapida de vehiculos
    step1_start = time.time()
    vehicles = [2, 3, 5, 7]
    detections = coco_model(
        frame,
        imgsz=224,
        conf=0.6,
        iou=0.7,
        max_det=10,
        device=device,
        verbose=False,
        half=True if device == 'cuda' else False,
        augment=False
    )[0]
    
    vehicle_detections = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            vehicle_detections.append([x1, y1, x2, y2, score])
    
    step1_time = time.time() - step1_start
    print(f"Vehiculos: {len(vehicle_detections)} en {step1_time:.3f}s")
    
    # PASO 2: Deteccion ultra-rapida de placas
    step2_start = time.time()
    plate_results = license_plate_detector(
        frame,
        imgsz=320,
        conf=0.2,
        iou=0.4,
        max_det=5,
        device=device,
        verbose=False,
        half=True if device == 'cuda' else False,
        augment=False
    )[0]
    
    plate_detections = plate_results.boxes.data.tolist()
    step2_time = time.time() - step2_start
    print(f"Placas: {len(plate_detections)} en {step2_time:.3f}s")
    
    # PASO 3: OCR ultra-optimizado
    step3_start = time.time()
    
    results = {}
    frame_nmr = 0
    results[frame_nmr] = {}
    
    # Crear track_ids simples para imagen
    track_ids = []
    for i, detection in enumerate(vehicle_detections):
        x1, y1, x2, y2, score = detection
        track_ids.append([x1, y1, x2, y2, i])
    
    # Si no hay vehiculos, crear virtual
    if len(track_ids) == 0 and plate_detections:
        x1, y1, x2, y2, score, class_id = plate_detections[0]
        padding = 50
        virtual_x1 = max(0, x1 - padding)
        virtual_y1 = max(0, y1 - padding)
        virtual_x2 = min(frame.shape[1], x2 + padding)
        virtual_y2 = min(frame.shape[0], y2 + padding)
        track_ids = [[virtual_x1, virtual_y1, virtual_x2, virtual_y2, 0]]
    
    plates_processed = 0
    
    # Procesar cada placa detectada
    for detection in plate_detections:
        x1, y1, x2, y2, score, class_id = detection
        
        # Asignar placa a vehiculo
        xcar1, ycar1, xcar2, ycar2, car_id = get_car_assignment(detection, track_ids)
        
        if car_id == -1 and len(track_ids) > 0:
            xcar1, ycar1, xcar2, ycar2, car_id = track_ids[0]
        
        if car_id != -1:
            # Recorte con padding optimizado
            padding = 25
            y1_padded = max(0, int(y1) - padding)
            y2_padded = min(frame.shape[0], int(y2) + padding)
            x1_padded = max(0, int(x1) - padding)
            x2_padded = min(frame.shape[1], int(x2) + padding)
            
            plate_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded, :]
            
            # OCR ultra-optimizado con reader cacheado
            plate_text, plate_confidence = ultrafast_ocr_optimized(plate_crop, easyocr_reader)
            
            if plate_text:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': plate_text,
                        'bbox_score': score,
                        'text_score': plate_confidence
                    }
                }
                plates_processed += 1
                
                # Guardar recorte
                cv2.imwrite(f"plate_crop_{car_id}.jpg", plate_crop)
                print(f"   Placa {plates_processed}: {plate_text} (conf: {plate_confidence:.3f})")
    
    step3_time = time.time() - step3_start
    
    print(f"OCR: {step3_time:.3f}s")
    print(f"Total placas procesadas: {plates_processed}")
    
    return results

def process_video_optimized(video_path, models, device, easyocr_reader):
    """Procesamiento ultra-rapido de video"""
    
    coco_model, license_plate_detector, model_name = models
    
    # Cargar video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error abriendo video: {video_path}")
        return None
    
    # Info del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video: {fps:.1f} FPS, {frame_count} frames, {duration:.1f}s")
    
    # Importar SORT para tracking
    try:
        from sort.sort import Sort
        mot_tracker = Sort()
    except ImportError:
        print("Error: No se pudo importar SORT tracker")
        return None
    
    vehicles = [2, 3, 5, 7]
    results = {}
    frame_nmr = -1
    ret = True
    plates_total = 0
    
    process_start = time.time()
    
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        
        if ret:
            # Mostrar progreso cada 30 frames
            if frame_nmr % 30 == 0:
                progress = (frame_nmr / frame_count) * 100 if frame_count > 0 else 0
                elapsed = time.time() - process_start
                fps_current = frame_nmr / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_nmr}/{frame_count} ({progress:.1f}%) - {fps_current:.1f} FPS")
            
            results[frame_nmr] = {}
            
            # Deteccion ultra-rapida de vehiculos
            detections = coco_model(
                frame,
                imgsz=224,
                conf=0.6,
                iou=0.7,
                max_det=10,
                device=device,
                verbose=False,
                half=True if device == 'cuda' else False,
                augment=False
            )[0]
            
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])
            
            # Tracking de vehiculos
            if len(detections_) > 0:
                track_ids = mot_tracker.update(np.asarray(detections_))
            else:
                track_ids = np.empty((0, 5))
            
            # Deteccion ultra-rapida de placas
            license_plates = license_plate_detector(
                frame,
                imgsz=320,
                conf=0.2,
                iou=0.4,
                max_det=5,
                device=device,
                verbose=False,
                half=True if device == 'cuda' else False,
                augment=False
            )[0]
            
            # Procesar placas detectadas
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                
                # Asignar placa a vehiculo
                xcar1, ycar1, xcar2, ycar2, car_id = get_car_assignment(license_plate, track_ids)
                
                if car_id != -1:
                    # Recorte de placa
                    padding = 20
                    y1_padded = max(0, int(y1) - padding)
                    y2_padded = min(frame.shape[0], int(y2) + padding)
                    x1_padded = max(0, int(x1) - padding)
                    x2_padded = min(frame.shape[1], int(x2) + padding)
                    
                    license_plate_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded, :]
                    
                    # OCR optimizado
                    license_plate_text, license_plate_text_score = ultrafast_ocr_optimized(
                        license_plate_crop, easyocr_reader
                    )
                    
                    if license_plate_text:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                        plates_total += 1
    
    cap.release()
    
    process_time = time.time() - process_start
    avg_fps = frame_nmr / process_time if process_time > 0 else 0
    
    print(f"Video procesado: {frame_nmr} frames en {process_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Total placas detectadas: {plates_total}")
    
    return results

def write_csv_results(results, output_path):
    """Escribir resultados en CSV - version simplificada"""
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    
                    car_bbox = results[frame_nmr][car_id]['car']['bbox']
                    plate_bbox = results[frame_nmr][car_id]['license_plate']['bbox']
                    plate_score = results[frame_nmr][car_id]['license_plate']['bbox_score']
                    plate_text = results[frame_nmr][car_id]['license_plate']['text']
                    text_score = results[frame_nmr][car_id]['license_plate']['text_score']
                    
                    f.write(f'{frame_nmr},{car_id},[{car_bbox[0]} {car_bbox[1]} {car_bbox[2]} {car_bbox[3]}],[{plate_bbox[0]} {plate_bbox[1]} {plate_bbox[2]} {plate_bbox[3]}],{plate_score},{plate_text},{text_score}\n')

def final_ultrafast_process(input_path):
    """Procesamiento final ultra-rapido - imagenes y videos"""
    print(f"*** PROCESAMIENTO FINAL ULTRA-RAPIDO: {input_path} ***")
    
    # Detectar tipo de archivo
    input_type = detect_input_type(input_path)
    if not input_type:
        return False
    
    print(f"Tipo detectado: {input_type}")
    
    total_start = time.time()
    
    # Cargar sistema completo (cached)
    models, device, easyocr_reader = get_all_cached_models()
    if models is None or easyocr_reader is None:
        print("Error: No se pudo cargar el sistema")
        return False
    
    coco_model, license_plate_detector, model_name = models
    print(f"Sistema: {model_name} + EasyOCR (cacheado)")
    
    # Procesar segun el tipo
    if input_type == 'image':
        results = process_single_image(input_path, models, device, easyocr_reader)
    elif input_type == 'video':
        results = process_video_optimized(input_path, models, device, easyocr_reader)
    else:
        print(f"Tipo no soportado: {input_type}")
        return False
    
    if results is None:
        print("Error en el procesamiento")
        return False
    
    total_time = time.time() - total_start
    
    # Guardar resultados
    write_csv_results(results, 'test.csv')
    
    # Contar placas detectadas
    total_plates = 0
    detected_texts = []
    for frame_data in results.values():
        for car_data in frame_data.values():
            if 'license_plate' in car_data and 'text' in car_data['license_plate']:
                total_plates += 1
                detected_texts.append(car_data['license_plate']['text'])
    
    # RESULTADOS FINALES
    print(f"\nRESULTADO FINAL ULTRA-RAPIDO:")
    if total_plates > 0:
        unique_plates = list(set(detected_texts))
        print(f"   Total placas detectadas: {total_plates}")
        print(f"   Placas unicas: {len(unique_plates)}")
        print(f"   Textos: {', '.join(unique_plates)}")
        print(f"   CSV compatible generado")
    else:
        print(f"   Sin placas legibles")
    
    # METRICAS FINALES
    print(f"\nMETRICAS FINALES:")
    print(f"   Tipo: {input_type}")
    print(f"   TOTAL: {total_time:.3f}s")
    
    if input_type == 'image':
        if total_time < 2.0:
            print(f"   OBJETIVO SUPERADO! < 2s")
        elif total_time < 3.0:
            print(f"   OBJETIVO ALCANZADO! < 3s")
        elif total_time < 5.0:
            print(f"   EXCELENTE: < 5s muy bueno")
        else:
            print(f"   MEJORADO: Pero necesita mas optimizacion")
    else:  # video
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        processing_rate = file_size_mb / total_time if total_time > 0 else 0
        print(f"   Velocidad: {processing_rate:.2f} MB/s")
        print(f"   Rendimiento: Excelente para video")
    
    return True

def main():
    """Funcion principal final"""
    print("*** LPR FINAL ULTRA-RAPIDO - IMAGENES Y VIDEOS ***")
    print("=" * 55)
    
    if len(sys.argv) < 2:
        print("Uso: python main_final.py <archivo>")
        print("Formatos soportados:")
        print("  Imagenes: .jpg, .jpeg, .png, .bmp, .tiff")
        print("  Videos: .mp4, .avi, .mov, .mkv, .wmv")
        print("\nEjemplos:")
        print('  python main_final.py "images/placa1.jpg"')
        print('  python main_final.py "videos/colombia.mp4"')
        return
    
    input_file = sys.argv[1]
    
    if final_ultrafast_process(input_file):
        print(f"\nPROCESAMIENTO FINAL COMPLETADO")
        print(f"Compatible con pipeline original:")
        print(f"   python add_missing_data.py")
        print(f"   python visualize.py")
    else:
        print(f"\nError en procesamiento final")

if __name__ == "__main__":
    main()
