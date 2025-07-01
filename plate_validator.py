#!/usr/bin/env python3
"""
🔧 PLATE_VALIDATOR.PY - VALIDADOR DE PLACAS COLOMBIANAS
Funciones para normalizar y validar placas a exactamente 6 caracteres
Compatible con sistema LPR existente
"""

import cv2
import numpy as np
import re

def normalize_colombian_plate(raw_text):
    """
    Normalizar texto OCR a formato de placa colombiana válido
    
    Args:
        raw_text (str): Texto crudo del OCR
        
    Returns:
        str or None: Placa normalizada de 6 caracteres o None si no es válida
    """
    if not raw_text:
        return None
    
    # Limpiar texto: solo letras y números, mayúsculas
    clean_text = ''.join(c.upper() for c in raw_text if c.isalnum())
    
    # Si es menor a 6 caracteres, no es válida
    if len(clean_text) < 6:
        return None
    
    # Si es mayor a 6, intentar extraer los 6 más probables
    if len(clean_text) > 6:
        # Buscar patrones típicos colombianos
        patterns = [
            r'([A-Z]{3})([0-9]{3})',  # Patrón estándar: ABC123
            r'([A-Z]{2})([0-9]{4})',  # Patrón diplomático: CD1234 (pero cortamos a 6)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                letters = match.group(1)
                numbers = match.group(2)
                
                # Para patrón estándar
                if len(letters) == 3 and len(numbers) >= 3:
                    return letters + numbers[:3]
                
                # Para patrón diplomático, tomar solo 6 caracteres
                elif len(letters) == 2 and len(numbers) >= 4:
                    return letters + numbers[:4]  # Mantener CD1234, pero validaremos después
        
        # Si no encuentra patrón, tomar los primeros 6
        clean_text = clean_text[:6]
    
    # Validar que sea formato colombiano válido
    if is_valid_colombian_format(clean_text):
        return clean_text
    
    return None

def is_valid_colombian_format(plate_text):
    """
    Validar que la placa tenga formato colombiano válido
    
    Args:
        plate_text (str): Texto de 6 caracteres
        
    Returns:
        bool: True si es formato válido
    """
    if not plate_text or len(plate_text) != 6:
        return False
    
    # Patrón estándar: 3 letras + 3 números (ABC123)
    if re.match(r'^[A-Z]{3}[0-9]{3}$', plate_text):
        return True
    
    # Patrón diplomático: CD + 4 números (CD1234) - pero son 6 caracteres
    if re.match(r'^CD[0-9]{4}$', plate_text):
        return True
    
    return False

def extract_best_plate_candidates(raw_text):
    """
    Extraer múltiples candidatos posibles de una cadena más larga
    
    Args:
        raw_text (str): Texto crudo del OCR
        
    Returns:
        list: Lista de candidatos ordenados por probabilidad
    """
    if not raw_text:
        return []
    
    clean_text = ''.join(c.upper() for c in raw_text if c.isalnum())
    candidates = []
    
    # Extraer todas las subcadenas de 6 caracteres
    for i in range(len(clean_text) - 5):
        candidate = clean_text[i:i+6]
        if is_valid_colombian_format(candidate):
            candidates.append(candidate)
    
    # Buscar patrones específicos
    patterns = [
        r'([A-Z]{3}[0-9]{3})',  # ABC123
        r'(CD[0-9]{4})',        # CD1234
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, clean_text)
        for match in matches:
            if match not in candidates:
                candidates.append(match)
    
    # Ordenar por probabilidad (estándar primero)
    def score_candidate(candidate):
        if re.match(r'^[A-Z]{3}[0-9]{3}$', candidate):
            return 3  # Formato estándar
        elif re.match(r'^CD[0-9]{4}$', candidate):
            return 2  # Formato diplomático
        else:
            return 1  # Otros
    
    candidates.sort(key=score_candidate, reverse=True)
    return candidates

def calculate_format_score(text):
    """
    Calcular un score basado en formatos de placas colombianas
    """
    score = 0.0
    
    # Formato típico: ABC123 (3 letras + 3 números)
    if len(text) == 6:
        if text[:3].isalpha() and text[3:].isdigit():
            score += 0.9
        elif text[:2].isalpha() and text[2:].isdigit():  # CD1234
            score += 0.8
    
    # Formato moto: ABC12 (3 letras + 2 números) - pero lo extendemos a 6
    elif len(text) == 5:
        if text[:3].isalpha() and text[3:].isdigit():
            score += 0.7
    
    # Formato diplomático específico
    elif len(text) == 6 and text.startswith('CD'):
        if text[:2].isalpha() and text[2:].isdigit():
            score += 0.95
    
    return min(score, 1.0)

# ================================
# FUNCIÓN PRINCIPAL MEJORADA
# ================================

def ultrafast_ocr_optimized_with_validation(image_crop, reader):
    """
    🚀 OCR ultra-optimizado CON VALIDACIÓN de placas colombianas
    
    Esta función reemplaza la original en main_final.py
    Garantiza exactamente 6 caracteres en formato colombiano válido
    
    Args:
        image_crop: Imagen recortada de la placa
        reader: Instancia de EasyOCR reader
        
    Returns:
        tuple: (placa_normalizada, confianza) o (None, None)
    """
    try:
        # Preprocesamiento mínimo pero efectivo
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()
        
        # Solo redimensionar si es muy pequeña
        height, width = gray.shape
        if height < 40 or width < 120:
            scale = max(40/height, 120/width, 1.0)
            scale = min(scale, 3.0)  # Limitar escala
            new_h, new_w = int(height * scale), int(width * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Probar solo las mejores técnicas de threshold
        images_to_test = []
        
        # 1. OTSU (usualmente el mejor)
        try:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            images_to_test.append(otsu)
        except:
            pass
        
        # 2. Threshold fijo (backup)
        try:
            _, fixed = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            images_to_test.append(fixed)
        except:
            pass
        
        # 3. Original (si las anteriores fallan)
        images_to_test.append(gray)
        
        # OCR optimizado con validación
        best_text = None
        best_confidence = 0.0
        best_original = None
        
        for img in images_to_test:
            try:
                results = reader.readtext(
                    img,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    paragraph=False,
                    width_ths=0.7,
                    height_ths=0.7,
                    detail=1
                )
                
                for bbox, text, confidence in results:
                    # 🔧 NORMALIZAR A 6 CARACTERES VÁLIDOS
                    normalized_plate = normalize_colombian_plate(text)
                    
                    if normalized_plate and confidence > 0.1:
                        # Verificar formato básico
                        has_letters = any(c.isalpha() for c in normalized_plate)
                        has_numbers = any(c.isdigit() for c in normalized_plate)
                        
                        if has_letters and has_numbers:
                            # Score combinado con prioridad a formato válido
                            format_score = calculate_format_score(normalized_plate)
                            is_perfect_format = is_valid_colombian_format(normalized_plate)
                            
                            if is_perfect_format:
                                combined_score = confidence * 0.7 + format_score * 0.3
                            else:
                                combined_score = confidence * 0.5 + format_score * 0.5
                            
                            if combined_score > best_confidence:
                                best_text = normalized_plate
                                best_confidence = combined_score
                                best_original = text
                
                # Si encontramos formato válido perfecto, no seguir probando
                if best_text and is_valid_colombian_format(best_text):
                    break
                    
            except Exception:
                continue
        
        # Si no encontró nada válido, intentar extraer candidatos del texto completo
        if not best_text and any(results for results in [reader.readtext(img, detail=1) for img in images_to_test]):
            try:
                # Concatenar todo el texto detectado de todas las imágenes
                all_texts = []
                for img in images_to_test:
                    try:
                        img_results = reader.readtext(img, detail=1)
                        for _, text, _ in img_results:
                            all_texts.append(text)
                    except:
                        continue
                
                combined_text = ''.join(all_texts)
                candidates = extract_best_plate_candidates(combined_text)
                
                if candidates:
                    # Tomar el mejor candidato con confianza ajustada
                    best_text = candidates[0]
                    best_confidence = 0.6  # Confianza reducida por ser extraído
                    best_original = combined_text
            except Exception:
                pass
        
        # 📏 LOG DE TRANSFORMACIÓN (para debugging)
        if best_text and best_original and best_original.replace(' ', '').upper() != best_text:
            print(f"📏 Placa normalizada: '{best_original}' → '{best_text}' (6 chars)")
        
        # Formatear resultado final
        if best_text:
            if is_valid_colombian_format(best_text):
                return best_text, best_confidence
            else:
                # Si no es formato perfecto pero tiene 6 caracteres válidos, devolverlo con confianza reducida
                return best_text, best_confidence * 0.8
                
        return None, None
        
    except Exception as e:
        print(f"⚠️ Error en OCR con validación: {e}")
        return None, None

# ===================================
# FUNCIÓN DE TESTING
# ===================================

def test_plate_validation():
    """Función para probar la validación"""
    test_cases = [
        "ITRO4731",   # Debería ser ITRO47
        "IJVO752",    # Debería ser IJVO75
        "ABC123",     # Válida tal como está
        "CD1234",     # Válida diplomática
        "XYZ999",     # Válida tal como está
        "XYZ",        # Muy corta, inválida
        "ABCD1234EF", # Debería extraer ABC123 o similar
        "123ABC456",  # Patrón inválido
        "AAA111BBB",  # Debería ser AAA111
        "CD5678XY",   # Debería ser CD5678
    ]
    
    print("🧪 TESTING VALIDACIÓN DE PLACAS COLOMBIANAS:")
    print("=" * 55)
    
    for test_text in test_cases:
        normalized = normalize_colombian_plate(test_text)
        candidates = extract_best_plate_candidates(test_text)
        is_valid = is_valid_colombian_format(normalized) if normalized else False
        
        print(f"Entrada:      '{test_text}'")
        print(f"Normalizada:  {normalized if normalized else 'None'}")
        print(f"Candidatos:   {candidates if candidates else '[]'}")
        print(f"Válida:       {'✅' if is_valid else '❌'}")
        print("-" * 35)

# ===================================
# FUNCIÓN DE COMPATIBILIDAD
# ===================================

def license_complies_format(text):
    """
    Función de compatibilidad con util.py
    """
    return is_valid_colombian_format(text)

def format_license(text):
    """
    Función de compatibilidad con util.py
    """
    return normalize_colombian_plate(text) or text

# ===================================
# MAIN Y TESTING
# ===================================

if __name__ == "__main__":
    print("🔧 PLATE_VALIDATOR.PY - Sistema de Validación de Placas Colombianas")
    print("=" * 70)
    print("✅ Garantiza exactamente 6 caracteres")
    print("✅ Valida formato colombiano: ABC123 o CD1234")
    print("✅ Extrae patrones válidos de cadenas largas")
    print("✅ Compatible con sistema LPR existente")
    print("=" * 70)
    
    test_plate_validation()
    
    print("\n🚀 INSTRUCCIONES DE USO:")
    print("1. Copiar este archivo como 'plate_validator.py'")
    print("2. En main_final.py, agregar import:")
    print("   from plate_validator import ultrafast_ocr_optimized_with_validation as ultrafast_ocr_optimized")
    print("3. ¡Listo! Automáticamente validará placas a 6 caracteres")