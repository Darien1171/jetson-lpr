#!/usr/bin/env python3
"""
💾 Database Configuration for LPR Stream
Configuraciones específicas de base de datos
"""

class DatabaseConfig:
    """Configuraciones de base de datos para diferentes entornos"""
    
    # Configuración para desarrollo (PC)
    DEVELOPMENT = {
        'type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'parqueadero',
        'user': '',
        'password': '',
        'charset': 'utf8mb4',
        'autocommit': True,
        'connection_timeout': 30,
        'pool_size': 5,
        'pool_name': 'lpr_dev_pool'
    }
    
    # Configuración para producción (Jetson)
    PRODUCTION = {
        'type': 'mysql',
        'host': 'localhost',
        'port': 3306,
        'database': 'parqueadero_jetson',
        'user': 'lpr_user',
        'password': 'lpr_password',
        'charset': 'utf8mb4',
        'autocommit': True,
        'connection_timeout': 60,
        'pool_size': 10,
        'pool_name': 'lpr_prod_pool'
    }
    
    # Configuración para testing
    TESTING = {
        'type': 'sqlite',
        'database': ':memory:',
        'check_same_thread': False
    }
    
    @classmethod
    def get_config(cls, mode='development'):
        """Obtener configuración según modo"""
        if mode == 'development':
            return cls.DEVELOPMENT.copy()
        elif mode == 'production':
            return cls.PRODUCTION.copy()
        elif mode == 'testing':
            return cls.TESTING.copy()
        else:
            raise ValueError(f"Modo no soportado: {mode}")
    
    @classmethod
    def get_mysql_connection_string(cls, mode='development'):
        """Obtener string de conexión MySQL"""
        config = cls.get_config(mode)
        
        if config['type'] != 'mysql':
            raise ValueError("Solo disponible para MySQL")
        
        return f"mysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
    
    @classmethod
    def get_sqlalchemy_url(cls, mode='development'):
        """Obtener URL para SQLAlchemy"""
        config = cls.get_config(mode)
        
        if config['type'] == 'mysql':
            return f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"
        elif config['type'] == 'sqlite':
            return f"sqlite:///{config['database']}"
        else:
            raise ValueError(f"Tipo de BD no soportado: {config['type']}")

class TableSchemas:
    """Esquemas de tablas para el sistema LPR"""
    
    # Tabla de detecciones
    LPR_DETECTIONS = {
        'mysql': """
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence FLOAT,
            plate_score FLOAT,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'unknown',
            processed BOOLEAN DEFAULT FALSE,
            entry_type ENUM('entrada', 'salida') DEFAULT 'entrada',
            
            INDEX idx_timestamp (timestamp),
            INDEX idx_plate (plate_text),
            INDEX idx_location (camera_location),
            INDEX idx_processed (processed),
            INDEX idx_entry_type (entry_type)
        )
        """,
        
        'sqlite': """
        CREATE TABLE IF NOT EXISTS lpr_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            plate_text VARCHAR(10) NOT NULL,
            confidence REAL,
            plate_score REAL,
            vehicle_bbox TEXT,
            plate_bbox TEXT,
            camera_location VARCHAR(100) DEFAULT 'unknown',
            processed BOOLEAN DEFAULT 0,
            entry_type VARCHAR(10) DEFAULT 'entrada'
        )
        """
    }
    
    # Tabla de vehículos registrados
    REGISTERED_VEHICLES = {
        'mysql': """
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
        """,
        
        'sqlite': """
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number VARCHAR(10) UNIQUE NOT NULL,
            owner_name VARCHAR(100),
            owner_phone VARCHAR(20),
            vehicle_type VARCHAR(20) DEFAULT 'particular',
            vehicle_brand VARCHAR(50),
            vehicle_color VARCHAR(30),
            authorized BOOLEAN DEFAULT 1,
            authorization_start DATE,
            authorization_end DATE,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """
    }
    
    # Tabla de log de accesos
    ACCESS_LOG = {
        'mysql': """
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
        """,
        
        'sqlite': """
        CREATE TABLE IF NOT EXISTS access_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER,
            plate_number VARCHAR(10) NOT NULL,
            access_granted BOOLEAN DEFAULT 0,
            access_reason VARCHAR(100),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            camera_location VARCHAR(100),
            
            FOREIGN KEY (detection_id) REFERENCES lpr_detections(id)
        )
        """
    }
    
    @classmethod
    def get_create_statement(cls, table_name, db_type='mysql'):
        """Obtener statement de creación de tabla"""
        table_schemas = {
            'lpr_detections': cls.LPR_DETECTIONS,
            'registered_vehicles': cls.REGISTERED_VEHICLES,
            'access_log': cls.ACCESS_LOG
        }
        
        if table_name not in table_schemas:
            raise ValueError(f"Tabla no encontrada: {table_name}")
        
        if db_type not in table_schemas[table_name]:
            raise ValueError(f"Tipo de BD no soportado: {db_type}")
        
        return table_schemas[table_name][db_type]
    
    @classmethod
    def get_all_tables(cls, db_type='mysql'):
        """Obtener todas las tablas"""
        tables = ['lpr_detections', 'registered_vehicles', 'access_log']
        return {table: cls.get_create_statement(table, db_type) for table in tables}

class SampleData:
    """Datos de muestra para testing"""
    
    DEVELOPMENT_VEHICLES = [
        ('ABC123', 'Juan Perez - DEV', '555-0001', 'particular', 'Toyota', 'Blanco', True, '2024-01-01', '2025-12-31', 'Vehículo de desarrollo'),
        ('XYZ789', 'Maria Garcia - DEV', '555-0002', 'particular', 'Chevrolet', 'Azul', True, '2024-01-01', '2025-12-31', 'Vehículo de desarrollo'),
        ('MOT45A', 'Carlos Lopez - DEV', '555-0003', 'moto', 'Yamaha', 'Rojo', True, '2024-01-01', '2025-12-31', 'Moto de desarrollo'),
        ('CD1234', 'Embajada Test - DEV', '555-0004', 'diplomatico', 'Mercedes', 'Negro', True, '2024-01-01', '2025-12-31', 'Vehículo diplomático de prueba'),
        ('DEV456', 'Test No Autorizado', '555-0099', 'particular', 'Test', 'Test', False, '2024-01-01', '2024-01-01', 'Solo para pruebas - NO AUTORIZADO'),
        ('TEST01', 'Vehículo Prueba 1', '555-0101', 'particular', 'Honda', 'Verde', True, '2024-01-01', '2025-12-31', 'Prueba 1'),
        ('TEST02', 'Vehículo Prueba 2', '555-0102', 'particular', 'Nissan', 'Gris', True, '2024-01-01', '2025-12-31', 'Prueba 2'),
        ('MOTO01', 'Moto Prueba', '555-0201', 'moto', 'Kawasaki', 'Azul', True, '2024-01-01', '2025-12-31', 'Moto de prueba')
    ]
    
    PRODUCTION_VEHICLES = [
        ('ABC123', 'Administrador Sistema', '555-0001', 'particular', 'Toyota', 'Blanco', True, '2024-01-01', '2025-12-31', 'Vehículo administrativo'),
        ('XYZ789', 'Seguridad Parking', '555-0002', 'particular', 'Chevrolet', 'Azul', True, '2024-01-01', '2025-12-31', 'Vehículo de seguridad'),
        ('CD1234', 'Embajada Ejemplo', '555-0003', 'diplomatico', 'Mercedes', 'Negro', True, '2024-01-01', '2025-12-31', 'Vehículo diplomático'),
        ('MOT01A', 'Mensajería Rápida', '555-0004', 'moto', 'Yamaha', 'Rojo', True, '2024-01-01', '2025-12-31', 'Moto de mensajería'),
        ('TEST01', 'Vehículo Prueba', '555-0099', 'particular', 'Test', 'Test', False, '2024-01-01', '2024-01-01', 'Solo para pruebas - NO AUTORIZADO')
    ]
    
    @classmethod
    def get_sample_vehicles(cls, mode='development'):
        """Obtener vehículos de muestra según modo"""
        if mode == 'development':
            return cls.DEVELOPMENT_VEHICLES
        elif mode == 'production':
            return cls.PRODUCTION_VEHICLES
        else:
            return []

# Funciones de utilidad
def get_db_config(mode='development'):
    """Función helper para obtener configuración"""
    return DatabaseConfig.get_config(mode)

def get_table_schema(table_name, db_type='mysql'):
    """Función helper para obtener esquema de tabla"""
    return TableSchemas.get_create_statement(table_name, db_type)

def get_sample_data(mode='development'):
    """Función helper para obtener datos de muestra"""
    return SampleData.get_sample_vehicles(mode)

if __name__ == "__main__":
    # Test de configuraciones
    print("🧪 Testing Database Config...")
    
    # Test configuraciones
    dev_config = DatabaseConfig.get_config('development')
    prod_config = DatabaseConfig.get_config('production')
    
    print("Development config:", dev_config['database'])
    print("Production config:", prod_config['database'])
    
    # Test esquemas
    lpr_table = TableSchemas.get_create_statement('lpr_detections', 'mysql')
    print("LPR table schema available:", len(lpr_table) > 0)
    
    # Test datos de muestra
    dev_vehicles = SampleData.get_sample_vehicles('development')
    print(f"Development vehicles: {len(dev_vehicles)}")
    
    print("✅ Database Config test completado")
