import ast
import cv2
import numpy as np
import pandas as pd
import os
import sys


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Dibuja bordes estilizados en las esquinas del bounding box
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def detect_input_type(results_df):
    """
    Detecta si los datos provienen de una imagen o un video
    """
    unique_frames = results_df['frame_nmr'].nunique()
    max_frame = results_df['frame_nmr'].max()
    
    is_single_image = unique_frames == 1 and max_frame == 0
    
    return 'image' if is_single_image else 'video'


def find_original_source():
    """
    Intenta encontrar el archivo de origen (imagen o video) basándose en archivos comunes
    """
    # Buscar videos comunes
    video_patterns = ['./videos/colombia.mp4', './colombia.mp4', './video.mp4', './input.mp4']
    for pattern in video_patterns:
        if os.path.exists(pattern):
            return pattern, 'video'
    
    # Buscar imágenes comunes
    image_patterns = ['./image.jpg', './input.jpg', './test.jpg', './placa.jpg']
    for pattern in image_patterns:
        if os.path.exists(pattern):
            return pattern, 'image'
    
    # Buscar en directorio actual cualquier imagen o video
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            return file, 'image'
        elif file.lower().endswith(('.mp4', '.avi', '.mov')):
            return file, 'video'
    
    return None, None


def process_image_visualization(results, source_path=None):
    """
    Crea visualización para una imagen individual con marcos delgados estilo YOLO
    """
    print("=== GENERANDO VISUALIZACIÓN PARA IMAGEN ===")
    
    if source_path and os.path.exists(source_path):
        print(f"Usando imagen fuente: {source_path}")
        frame = cv2.imread(source_path)
    else:
        print("No se especificó imagen fuente, creando imagen en blanco")
        # Crear imagen en blanco basada en los bounding boxes
        max_x = max([ast.literal_eval(bbox.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))[2] 
                    for bbox in results['car_bbox']])
        max_y = max([ast.literal_eval(bbox.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))[3] 
                    for bbox in results['car_bbox']])
        frame = np.ones((int(max_y) + 100, int(max_x) + 100, 3), dtype=np.uint8) * 50

    # Preparar datos de placas para obtener los mejores textos
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        car_results = results[results['car_id'] == car_id]
        if not car_results.empty:
            max_score_idx = car_results['license_number_score'].astype(float).idxmax()
            best_result = car_results.loc[max_score_idx]
            
            license_plate[car_id] = {
                'license_plate_number': best_result['license_number']
            }

    # Dibujar anotaciones en la imagen con marcos delgados estilo YOLO
    annotated_frame = frame.copy()
    
    for row_indx in range(len(results)):
        try:
            # Dibujar vehículo con rectángulo verde delgado (estilo YOLO)
            car_bbox_str = results.iloc[row_indx]['car_bbox']
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                car_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(annotated_frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), 
                         (0, 255, 0), 3)  # Verde, grosor 3 (delgado)

            # Dibujar placa con rectángulo rojo delgado (estilo YOLO)
            plate_bbox_str = results.iloc[row_indx]['license_plate_bbox']
            x1, y1, x2, y2 = ast.literal_eval(
                plate_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
            )
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Rojo, grosor 2 (delgado)

            # Agregar texto de la placa cerca del marco
            car_id = results.iloc[row_indx]['car_id']
            if car_id in license_plate:
                license_text = license_plate[car_id]['license_plate_number']
                
                # Calcular posición del texto cerca de la placa
                text_x = int(x1)
                text_y = int(y1) - 20  # Justo arriba de la placa
                
                # Si el texto queda fuera del frame por arriba, ponerlo abajo
                if text_y < 50:
                    text_y = int(y2) + 60  # Justo abajo de la placa
                
                # Fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                
                # Crear rectángulo de fondo para el texto
                cv2.rectangle(annotated_frame, 
                            (text_x - 5, text_y - text_height - 5),
                            (text_x + text_width + 5, text_y + 5),
                            (255, 255, 255), -1)  # Fondo blanco sólido
                
                cv2.rectangle(annotated_frame, 
                            (text_x - 5, text_y - text_height - 5),
                            (text_x + text_width + 5, text_y + 5),
                            (0, 0, 0), 1)  # Borde negro delgado

                # Texto de la placa en negro sobre fondo blanco
                cv2.putText(annotated_frame, license_text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        except Exception as e:
            print(f"Error procesando fila {row_indx}: {e}")
            continue

    # Guardar imagen anotada
    output_path = './annotated_result.jpg'
    cv2.imwrite(output_path, annotated_frame)
    print(f"[OK] Imagen anotada guardada: {output_path}")
    
    # También crear una versión redimensionada para visualización
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    display_output_path = './annotated_result_display.jpg'
    cv2.imwrite(display_output_path, display_frame)
    print(f"[OK] Imagen de visualización guardada: {display_output_path}")
    
    return output_path


def process_video_visualization(results, source_path):
    """
    Crea visualización para video con marcos delgados estilo YOLO
    """
    print("=== GENERANDO VISUALIZACIÓN PARA VIDEO ===")
    print(f"Usando video fuente: {source_path}")
    
    # Preparar datos de placas para obtener los mejores textos
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {
            'license_plate_number': results[(results['car_id'] == car_id) &
                                          (results['license_number_score'] == max_)]['license_number'].iloc[0]
        }

    # Procesar video frame por frame
    cap = cv2.VideoCapture(source_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                try:
                    # Dibujar vehículo con rectángulo verde delgado (estilo YOLO)
                    car_bbox_str = df_.iloc[row_indx]['car_bbox']
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), 
                                 (0, 255, 0), 3)  # Verde, grosor 3 (delgado)

                    # Dibujar placa con rectángulo rojo delgado (estilo YOLO)
                    plate_bbox_str = df_.iloc[row_indx]['license_plate_bbox']
                    x1, y1, x2, y2 = ast.literal_eval(plate_bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Rojo, grosor 2 (delgado)

                    # Agregar texto de la placa cerca del marco
                    car_id = df_.iloc[row_indx]['car_id']
                    if car_id in license_plate:
                        license_text = license_plate[car_id]['license_plate_number']
                        
                        # Calcular posición del texto cerca de la placa
                        text_x = int(x1)
                        text_y = int(y1) - 15  # Justo arriba de la placa
                        
                        # Si el texto queda fuera del frame por arriba, ponerlo abajo
                        if text_y < 40:
                            text_y = int(y2) + 40  # Justo abajo de la placa
                        
                        # Fondo para el texto
                        (text_width, text_height), _ = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        
                        # Asegurar que el texto no se salga del frame
                        if text_x + text_width > width:
                            text_x = width - text_width - 10
                        if text_x < 0:
                            text_x = 5
                        
                        # Crear rectángulo de fondo para el texto
                        cv2.rectangle(frame, 
                                    (text_x - 3, text_y - text_height - 3),
                                    (text_x + text_width + 3, text_y + 3),
                                    (255, 255, 255), -1)  # Fondo blanco sólido
                        
                        cv2.rectangle(frame, 
                                    (text_x - 3, text_y - text_height - 3),
                                    (text_x + text_width + 3, text_y + 3),
                                    (0, 0, 0), 1)  # Borde negro delgado

                        # Texto de la placa en negro sobre fondo blanco
                        cv2.putText(frame, license_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

                except Exception as e:
                    print(f"Error procesando frame {frame_nmr}, fila {row_indx}: {e}")
                    continue

            out.write(frame)

    out.release()
    cap.release()
    print("[OK] Video anotado guardado: ./out.mp4")
    return './out.mp4'


def main():
    """
    Función principal que determina el tipo de entrada y genera la visualización apropiada
    """
    # Verificar archivo de entrada
    input_file = './test_interpolated.csv'
    if not os.path.exists(input_file):
        print(f"Error: No se encontró {input_file}")
        print("Asegúrate de haber ejecutado add_missing_data.py primero")
        return

    # Cargar datos
    try:
        results = pd.read_csv(input_file)
        print(f"Datos cargados: {len(results)} filas")
    except Exception as e:
        print(f"Error al cargar {input_file}: {e}")
        return

    if results.empty:
        print("Error: El archivo CSV está vacío")
        return

    # Detectar tipo de entrada
    input_type = detect_input_type(results)
    print(f"Tipo detectado: {input_type}")

    # Buscar archivo fuente
    source_path = None
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        if not os.path.exists(source_path):
            print(f"Warning: Archivo especificado {source_path} no existe")
            source_path = None

    if not source_path:
        source_path, detected_type = find_original_source()
        if source_path:
            print(f"Archivo fuente detectado automáticamente: {source_path}")
        else:
            print("No se pudo detectar archivo fuente automáticamente")

    # Generar visualización según el tipo
    if input_type == 'image':
        output_path = process_image_visualization(results, source_path)
        print(f"\n[OK] Visualización de imagen completada")
        print(f"Archivo generado: {output_path}")
    else:
        if not source_path or not source_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Error: Para videos se requiere especificar el archivo de video fuente")
            print("Uso: python visualize.py video.mp4")
            return
        
        output_path = process_video_visualization(results, source_path)
        print(f"\n[OK] Visualización de video completada")
        print(f"Archivo generado: {output_path}")


if __name__ == "__main__":
    main()
