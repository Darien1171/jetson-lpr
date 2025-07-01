import csv
import numpy as np
from scipy.interpolate import interp1d
import os


def interpolate_bounding_boxes(data):
    """
    Interpolación de bounding boxes - adaptado para manejar tanto videos como imágenes
    """
    if not data:
        print("Warning: No hay datos para interpolar")
        return []
    
    # Extract necessary data columns from input data
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    
    # Verificar si es una imagen (solo frame 0) o video (múltiples frames)
    is_single_image = len(np.unique(frame_numbers)) == 1 and frame_numbers[0] == 0
    
    if is_single_image:
        print("Detectado: Procesamiento de imagen única (frame 0)")
        print("No se requiere interpolación para imágenes individuales")
        
        # Para imágenes, simplemente copiar los datos sin interpolación
        for i, row in enumerate(data):
            interpolated_row = {
                'frame_nmr': row['frame_nmr'],
                'car_id': row['car_id'],
                'car_bbox': row['car_bbox'],
                'license_plate_bbox': row['license_plate_bbox'],
                'license_plate_bbox_score': row.get('license_plate_bbox_score', '0'),
                'license_number': row.get('license_number', '0'),
                'license_number_score': row.get('license_number_score', '0')
            }
            interpolated_data.append(interpolated_row)
        
        print(f"Datos copiados para {len(interpolated_data)} detecciones")
        return interpolated_data
    
    else:
        print("Detectado: Procesamiento de video (múltiples frames)")
        print("Aplicando interpolación para frames faltantes...")
        
        # Código original de interpolación para videos
        for car_id in unique_car_ids:
            frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
            print(f"Procesando car_id {car_id}: frames {frame_numbers_}")

            # Filter data for a specific car ID
            car_mask = car_ids == car_id
            car_frame_numbers = frame_numbers[car_mask]
            car_bboxes_interpolated = []
            license_plate_bboxes_interpolated = []

            first_frame_number = car_frame_numbers[0]
            last_frame_number = car_frame_numbers[-1]

            for i in range(len(car_bboxes[car_mask])):
                frame_number = car_frame_numbers[i]
                car_bbox = car_bboxes[car_mask][i]
                license_plate_bbox = license_plate_bboxes[car_mask][i]

                if i > 0:
                    prev_frame_number = car_frame_numbers[i-1]
                    prev_car_bbox = car_bboxes_interpolated[-1]
                    prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                    if frame_number - prev_frame_number > 1:
                        # Interpolate missing frames' bounding boxes
                        frames_gap = frame_number - prev_frame_number
                        x = np.array([prev_frame_number, frame_number])
                        x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                        interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                        interpolated_car_bboxes = interp_func(x_new)
                        interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                        interpolated_license_plate_bboxes = interp_func(x_new)

                        car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                        license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

                car_bboxes_interpolated.append(car_bbox)
                license_plate_bboxes_interpolated.append(license_plate_bbox)

            for i in range(len(car_bboxes_interpolated)):
                frame_number = first_frame_number + i
                row = {}
                row['frame_nmr'] = str(frame_number)
                row['car_id'] = str(car_id)
                row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
                row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

                if str(frame_number) not in frame_numbers_:
                    # Imputed row, set the following fields to '0'
                    row['license_plate_bbox_score'] = '0'
                    row['license_number'] = '0'
                    row['license_number_score'] = '0'
                else:
                    # Original row, retrieve values from the input data if available
                    original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                    row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                    row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                    row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

                interpolated_data.append(row)

        return interpolated_data


def main():
    """
    Función principal que verifica la existencia del archivo y procesa los datos
    """
    input_file = 'test.csv'
    output_file = 'test_interpolated.csv'
    
    # Verificar que existe el archivo de entrada
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo {input_file}")
        print("Asegúrate de haber ejecutado main.py primero")
        return
    
    print(f"Leyendo datos desde {input_file}...")
    
    # Load the CSV file
    try:
        with open(input_file, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        
        print(f"Datos cargados: {len(data)} filas")
        
        if not data:
            print("Warning: El archivo CSV está vacío")
            return
        
    except Exception as e:
        print(f"Error al leer {input_file}: {e}")
        return

    # Interpolate missing data
    print("Iniciando proceso de interpolación...")
    interpolated_data = interpolate_bounding_boxes(data)
    
    if not interpolated_data:
        print("Error: No se generaron datos interpolados")
        return

    # Write updated data to a new CSV file
    print(f"Guardando datos interpolados en {output_file}...")
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    
    try:
        with open(output_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)
        
        print(f"[OK] Archivo {output_file} generado exitosamente")
        print(f"  Datos procesados: {len(interpolated_data)} filas")
        
        # Mostrar resumen
        unique_frames = len(set(row['frame_nmr'] for row in interpolated_data))
        unique_cars = len(set(row['car_id'] for row in interpolated_data))
        
        print(f"  Frames únicos: {unique_frames}")
        print(f"  Vehículos únicos: {unique_cars}")
        
        print(f"\nSiguiente paso: python visualize.py")
        
    except Exception as e:
        print(f"Error al guardar {output_file}: {e}")


if __name__ == "__main__":
    main()
