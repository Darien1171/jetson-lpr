#!/usr/bin/env python3
"""
💾 Setup MySQL para Jetson Orin Nano (Producción)
Configura base de datos MySQL para sistema LPR en producción
"""

import mysql.connector
from mysql.connector import Error
import sys
import subprocess
import platform

def check_mysql_service():
    """Verificar si MySQL está corriendo en Jetson"""
    try:
        # En Jetson/Linux, verificar proceso MySQL
        result = subprocess.run(['pgrep', 'mysqld'], 
                              capture_output=True)
        if result.returncode == 0:
            return True
        else:
            print("⚠️ MySQL no está corriendo")
            print("💡 Iniciar con: sudo systemctl start mysql")
            return False
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

def setup_mysql_jetson():
    """Configurar MySQL para producción en Jetson"""
    
    print("🤖 CONFIGURANDO MYSQL PARA JETSON ORIN NANO")
    print("=" * 50)
    
    # Verificar MySQL connector
    if not install_mysql_connector():
        return False
    
    # Verificar servicio MySQL
    if not check_mysql_service():
        print("❌ MySQL no está corriendo")
        print("🔧 Asegúrate de que MySQL esté instalado y corriendo")
        print("   sudo apt install mysql-server")
        print("   sudo systemctl start mysql")
        return False
    
    # Configuración root
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
        
        # Crear base de datos de producción
        print("📊 Creando base de datos parqueadero_jetson...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS parqueadero_jetson")
        cursor.execute("USE parqueadero_jetson")
        
        # Crear usuario de producción
        print("👤 Creando usuario lpr_user...")
        cursor.execute("DROP USER IF EXISTS 'lpr_user'@'localhost'")
        cursor.execute("CREATE USER 'lpr_user'@'localhost' IDENTIFIED BY 'lpr_password'")
        cursor.execute("GRANT ALL PRIVILEGES ON parqueadero_jetson.* TO 'lpr_user'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        # Crear tablas optimizadas para producción
        print("🗃️ Creando tablas de producción...")
        create_detections = """
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence FLOAT,
            plate_score FLOAT,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'entrada_principal',
            processed BOOLEAN DEFAULT FALSE,
            entry_type ENUM('entrada', 'salida') DEFAULT 'entrada',
            
            INDEX idx_timestamp (timestamp),
            INDEX idx_plate (plate_text),
            INDEX idx_location (camera_location),
            INDEX idx_processed (processed),
            INDEX idx_entry_type (entry_type)
        )
        """
        
        create_vehicles = """
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            owner_phone VARCHAR(20),
            vehicle_type ENUM('particular', 'moto', 'diplomatico', 'comercial') DEFAULT 'particular',
            vehicle_brand VARCHAR(50),
            vehicle_color VARCHAR(30),
            authorized BOOLEAN DEFAULT TRUE,
            authorization_start DATE,
            authorization_end DATE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            notes TEXT,
            
            INDEX idx_plate (plate_number),
            INDEX idx_authorized (authorized),
            INDEX idx_vehicle_type (vehicle_type),
            INDEX idx_authorization_end (authorization_end)
        )
        """
        
        create_access_log = """
        CREATE TABLE IF NOT EXISTS access_log (
            id INT AUTO_INCREMENT PRIMARY KEY,
            detection_id INT,
            plate_number VARCHAR(10) NOT NULL,
            access_granted BOOLEAN DEFAULT FALSE,
            access_reason VARCHAR(100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_location VARCHAR(100),
            
            FOREIGN KEY (detection_id) REFERENCES lpr_detections(id),
            INDEX idx_plate (plate_number),
            INDEX idx_timestamp (timestamp),
            INDEX idx_access (access_granted)
        )
        """
        
        cursor.execute(create_detections)
        cursor.execute(create_vehicles)
        cursor.execute(create_access_log)
        
        # Insertar vehículos de producción/prueba
        print("🚗 Insertando vehículos base...")
        production_vehicles = [
            ('ABC123', 'Administrador Sistema', '555-0001', 'particular', 'Toyota', 'Blanco', True, '2024-01-01', '2025-12-31', 'Vehículo administrativo'),
            ('XYZ789', 'Seguridad Parking', '555-0002', 'particular', 'Chevrolet', 'Azul', True, '2024-01-01', '2025-12-31', 'Vehículo de seguridad'),
            ('CD1234', 'Embajada Ejemplo', '555-0003', 'diplomatico', 'Mercedes', 'Negro', True, '2024-01-01', '2025-12-31', 'Vehículo diplomático'),
            ('MOT01A', 'Mensajería Rápida', '555-0004', 'moto', 'Yamaha', 'Rojo', True, '2024-01-01', '2025-12-31', 'Moto de mensajería'),
            ('TEST01', 'Vehículo Prueba', '555-0099', 'particular', 'Test', 'Test', False, '2024-01-01', '2024-01-01', 'Solo para pruebas - NO AUTORIZADO')
        ]
        
        insert_query = """
        INSERT IGNORE INTO registered_vehicles 
        (plate_number, owner_name, owner_phone, vehicle_type, vehicle_brand, vehicle_color, 
         authorized, authorization_start, authorization_end, notes) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.executemany(insert_query, production_vehicles)
        connection.commit()
        
        # Verificar instalación
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles")
        vehicle_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles WHERE authorized = TRUE")
        authorized_count = cursor.fetchone()[0]
        
        print("✅ MYSQL JETSON CONFIGURADO EXITOSAMENTE")
        print(f"   📊 Base de datos: parqueadero_jetson")
        print(f"   👤 Usuario: lpr_user")
        print(f"   🚗 Vehículos totales: {vehicle_count}")
        print(f"   ✅ Vehículos autorizados: {authorized_count}")
        print(f"   🔗 Conexión: {host}:{port}")
        
        # Guardar configuración para referencia
        config_info = f"""
# CONFIGURACIÓN MYSQL JETSON PRODUCCIÓN
host: {host}
port: {port}
database: parqueadero_jetson
user: lpr_user
password: lpr_password

# Conectar desde código:
mysql_config = {{
    'host': '{host}',
    'port': {port},
    'database': 'parqueadero_jetson',
    'user': 'lpr_user',
    'password': 'lpr_password',
    'charset': 'utf8mb4'
}}

# TABLAS CREADAS:
- lpr_detections: Detecciones de placas
- registered_vehicles: Vehículos registrados  
- access_log: Log de accesos

# COMANDOS ÚTILES:
sudo systemctl status mysql
sudo systemctl start mysql
sudo systemctl enable mysql
"""
        
        with open('mysql_jetson_config.txt', 'w') as f:
            f.write(config_info)
        
        print(f"📄 Configuración guardada en: mysql_jetson_config.txt")
        
        return True
        
    except Error as e:
        print(f"❌ Error configurando MySQL: {e}")
        print("\n🔧 POSIBLES SOLUCIONES:")
        print("1. Verificar que MySQL esté instalado: sudo apt install mysql-server")
        print("2. Iniciar MySQL: sudo systemctl start mysql")
        print("3. Verificar usuario y password de root")
        print("4. Configurar MySQL: sudo mysql_secure_installation")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def test_jetson_connection():
    """Test de conexión con configuración de Jetson"""
    try:
        config = {
            'host': 'localhost',
            'port': 3306,
            'database': 'parqueadero_jetson',
            'user': 'lpr_user',
            'password': 'lpr_password',
            'charset': 'utf8mb4'
        }
        
        print("🧪 Testing conexión Jetson...")
        
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        # Test básico
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM registered_vehicles WHERE authorized = TRUE")
        authorized = cursor.fetchone()[0]
        
        cursor.execute("SELECT plate_number, owner_name, vehicle_type FROM registered_vehicles WHERE authorized = TRUE LIMIT 3")
        samples = cursor.fetchall()
        
        print("✅ CONEXIÓN JETSON EXITOSA")
        print(f"   🚗 Vehículos registrados: {count}")
        print(f"   ✅ Vehículos autorizados: {authorized}")
        print("   📋 Ejemplos autorizados:")
        for plate, owner, v_type in samples:
            print(f"      {plate} - {owner} ({v_type})")
        
        # Test de rendimiento
        import time
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM lpr_detections WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
        detection_count = cursor.fetchone()[0]
        query_time = time.time() - start_time
        
        print(f"   📊 Detecciones 24h: {detection_count}")
        print(f"   ⚡ Query time: {query_time*1000:.1f}ms")
        
        connection.close()
        return True
        
    except Error as e:
        print(f"❌ Error en test Jetson: {e}")
        return False

def optimize_mysql_jetson():
    """Aplicar optimizaciones MySQL específicas para Jetson"""
    print("🚀 Aplicando optimizaciones MySQL para Jetson...")
    
    # Configuraciones MySQL optimizadas para Jetson Orin Nano
    mysql_optimizations = """
# MySQL Optimizations for Jetson Orin Nano
# Add to /etc/mysql/mysql.conf.d/jetson-lpr.cnf

[mysqld]
# Memory optimizations for 8GB Jetson
innodb_buffer_pool_size = 1G
innodb_log_file_size = 256M
innodb_flush_log_at_trx_commit = 2
innodb_flush_method = O_DIRECT

# Connection optimizations
max_connections = 50
wait_timeout = 600
interactive_timeout = 600

# Query cache
query_cache_type = 1
query_cache_size = 128M
query_cache_limit = 1M

# Table optimizations
tmp_table_size = 128M
max_heap_table_size = 128M

# Logging
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 2
"""
    
    try:
        with open('mysql_jetson_optimizations.cnf', 'w') as f:
            f.write(mysql_optimizations)
        
        print("📄 Optimizaciones guardadas en: mysql_jetson_optimizations.cnf")
        print("💡 Para aplicar:")
        print("   sudo cp mysql_jetson_optimizations.cnf /etc/mysql/mysql.conf.d/")
        print("   sudo systemctl restart mysql")
        
    except Exception as e:
        print(f"⚠️ Error creando archivo de optimizaciones: {e}")

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup MySQL para Jetson LPR')
    parser.add_argument('--test', action='store_true', help='Solo probar conexión')
    parser.add_argument('--optimize', action='store_true', help='Generar optimizaciones')
    
    args = parser.parse_args()
    
    if args.test:
        test_jetson_connection()
    elif args.optimize:
        optimize_mysql_jetson()
    else:
        success = setup_mysql_jetson()
        
        if success:
            print("\n🎉 SETUP JETSON COMPLETADO")
            print("💡 Próximo paso: python test_lpr.py")
            
            # Preguntar por optimizaciones
            optimize_response = input("\n¿Generar optimizaciones MySQL para Jetson? (S/n): ")
            if optimize_response.lower() != 'n':
                optimize_mysql_jetson()
        else:
            print("\n❌ SETUP JETSON FALLÓ")
            print("💡 Revisar configuración MySQL")

if __name__ == "__main__":
    main()
