#!/usr/bin/env python3
"""
💾 Setup MySQL para PC de Desarrollo
Configura base de datos MySQL local para testing
"""

import mysql.connector
from mysql.connector import Error
import sys
import subprocess
import platform

def check_mysql_service():
    """Verificar si MySQL está corriendo"""
    try:
        if platform.system() == "Windows":
            # En Windows, verificar servicio MySQL
            result = subprocess.run(['sc', 'query', 'MySQL80'], 
                                  capture_output=True, text=True)
            if 'RUNNING' in result.stdout:
                return True
            else:
                print("⚠️ MySQL no está corriendo en Windows")
                print("💡 Iniciar con: net start MySQL80")
                return False
        else:
            # En Linux/Mac, verificar proceso MySQL
            result = subprocess.run(['pgrep', 'mysqld'], 
                                  capture_output=True)
            return result.returncode == 0
    except:
        return False

def install_mysql_connector():
    """Instalar mysql-connector-python si no está disponible"""
    try:
        import mysql.connector
        return True
    except ImportError:
        print("📦 Instalando mysql-connector-python...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'mysql-connector-python'], 
                          check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Error instalando mysql-connector-python")
            return False

def setup_mysql_development():
    """Configurar MySQL para desarrollo"""
    
    print("🔧 CONFIGURANDO MYSQL PARA DESARROLLO")
    print("=" * 45)
    
    # Verificar MySQL connector
    if not install_mysql_connector():
        return False
    
    # Verificar servicio MySQL
    if not check_mysql_service():
        print("❌ MySQL no está corriendo")
        print("🔧 Asegúrate de que MySQL esté instalado y corriendo")
        return False
    
    # Configuración root (personalizable)
    print("🔑 Configuración de conexión MySQL:")
    host = input("Host MySQL (localhost): ").strip() or "localhost"
    port = input("Puerto MySQL (3306): ").strip() or "3306"
    root_user = input("Usuario root (root): ").strip() or "root"
    root_password = input("Password root: ").strip()
    
    root_config = {
        'host': host,
        'port': int(port),
        'user': root_user,
        'password': root_password
    }
    
    try:
        print(f"\n🔗 Conectando a MySQL en {host}:{port}...")
        
        # Conectar como root
        connection = mysql.connector.connect(**root_config)
        cursor = connection.cursor()
        
        # Crear base de datos de desarrollo
        print("📊 Creando base de datos lpr_development...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS lpr_development")
        cursor.execute("USE lpr_development")
        
        # Crear usuario de desarrollo
        print("👤 Creando usuario lpr_dev_user...")
        cursor.execute("DROP USER IF EXISTS 'lpr_dev_user'@'localhost'")
        cursor.execute("CREATE USER 'lpr_dev_user'@'localhost' IDENTIFIED BY 'lpr_dev_pass'")
        cursor.execute("GRANT ALL PRIVILEGES ON lpr_development.* TO 'lpr_dev_user'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # Crear tablas
        print("🗃️ Creando tablas...")
        create_detections = """
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence FLOAT,
            plate_score FLOAT,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'desarrollo_pc',
            processed BOOLEAN DEFAULT FALSE,
            
            INDEX idx_timestamp (timestamp),
            INDEX idx_plate (plate_text),
            INDEX idx_location (camera_location)
        )
        """
        
        create_vehicles = """
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            vehicle_type ENUM('particular', 'moto', 'diplomatico') DEFAULT 'particular',
            authorized BOOLEAN DEFAULT TRUE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            
            INDEX idx_plate (plate_number),
            INDEX idx_authorized (authorized)
        )
        """
        
        cursor.execute(create_detections)
        cursor.execute(create_vehicles)
        
        # Insertar vehículos de prueba para desarrollo
        print("🚗 Insertando vehículos de prueba...")
        test_vehicles = [
            ('ABC123', 'Juan Perez - DEV', 'particular', True),
            ('XYZ789', 'Maria Garcia - DEV', 'particular', True),
            ('MOT45A', 'Carlos Lopez - DEV', 'moto', True),
            ('CD1234', 'Embajada Test - DEV', 'diplomatico', True),
            ('DEV456', 'Test No Autorizado', 'particular', False),
            ('TEST01', 'Vehículo Prueba 1', 'particular', True),
            ('TEST02', 'Vehículo Prueba 2', 'particular', True),
            ('MOTO01', 'Moto Prueba', 'moto', True)
        ]
        
        insert_query = """
        INSERT IGNORE INTO registered_vehicles 
        (plate_number, owner_name, vehicle_type, authorized) 
        VALUES (%s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, test_vehicles)
        connection.commit()
        
        # Verificar instalación
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles")
        vehicle_count = cursor.fetchone()[0]
        
        print("✅ MYSQL CONFIGURADO EXITOSAMENTE")
        print(f"   📊 Base de datos: lpr_development")
        print(f"   👤 Usuario: lpr_dev_user")
        print(f"   🚗 Vehículos de prueba: {vehicle_count}")
        print(f"   🔗 Conexión: {host}:{port}")
        
        # Guardar configuración para referencia
        config_info = f"""
# CONFIGURACIÓN MYSQL DESARROLLO
host: {host}
port: {port}
database: lpr_development
user: lpr_dev_user
password: lpr_dev_pass

# Conectar desde código:
mysql_config = {{
    'host': '{host}',
    'port': {port},
    'database': 'lpr_development',
    'user': 'lpr_dev_user',
    'password': 'lpr_dev_pass',
    'charset': 'utf8mb4'
}}
"""
        
        with open('mysql_dev_config.txt', 'w') as f:
            f.write(config_info)
        
        print(f"📄 Configuración guardada en: mysql_dev_config.txt")
        
        return True
        
    except Error as e:
        print(f"❌ Error configurando MySQL: {e}")
        print("\n🔧 POSIBLES SOLUCIONES:")
        print("1. Verificar que MySQL esté instalado y corriendo")
        print("2. Verificar usuario y password de root")
        print("3. Verificar permisos de conexión")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def test_connection():
    """Test de conexión con configuración de desarrollo"""
    try:
        config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'lpr_development',
            'user': 'lpr_dev_user',
            'password': 'lpr_dev_pass',
            'charset': 'utf8mb4'
        }
        
        print("🧪 Testing conexión con configuración de desarrollo...")
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Test básico
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT plate_number, owner_name FROM registered_vehicles LIMIT 3")
        samples = cursor.fetchall()
        
        print("✅ CONEXIÓN EXITOSA")
        print(f"   🚗 Vehículos registrados: {count}")
        print("   📋 Ejemplos:")
        for plate, owner in samples:
            print(f"      {plate} - {owner}")
        
        connection.close()
        return True
        
    except Error as e:
        print(f"❌ Error en test de conexión: {e}")
        return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup MySQL para desarrollo LPR')
    parser.add_argument('--test', action='store_true', help='Solo probar conexión')
    
    args = parser.parse_args()
    
    if args.test:
        test_connection()
    else:
        success = setup_mysql_development()
        
        if success:
            print("\n🎉 SETUP COMPLETADO")
            print("💡 Próximo paso: python test_lpr.py")
        else:
            print("\n❌ SETUP FALLÓ")
            print("💡 Revisar configuración MySQL")

if __name__ == "__main__":
    main()
