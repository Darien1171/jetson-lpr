#!/bin/bash

# =============================================================================
# Script de Configuración PTZ Camera - Jetson Orin Nano
# Autor: Configuración para cámara PTZ con conexión Ethernet directa
# Uso: ./ptz_startup.sh
# =============================================================================

echo "==============================================="
echo "🎥 CONFIGURACIÓN CÁMARA PTZ - JETSON ORIN NANO"
echo "==============================================="
echo ""

# Configuración de variables
INTERFACE="enP8p1s0"
JETSON_IP="192.168.1.100"
CAMERA_IP="192.168.1.101"
RTSP_URL="rtsp://admin:admin@${CAMERA_IP}/cam/realmonitor?channel=1&subtype=1"

# Función para verificar si el comando fue exitoso
check_command() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ Error en: $1"
        exit 1
    fi
}

# Función para mostrar progreso
show_progress() {
    echo "🔄 $1..."
}

echo "📋 Configuración:"
echo "   - Interfaz: $INTERFACE"
echo "   - IP Jetson: $JETSON_IP"
echo "   - IP Cámara: $CAMERA_IP"
echo ""

# ==========================================
# 1. CONFIGURACIÓN DE RED
# ==========================================
show_progress "Configurando interfaz de red"

# Limpiar configuración previa
sudo ip addr flush dev $INTERFACE
check_command "Limpieza de configuración de red"

# Asignar IP a la Jetson
sudo ip addr add ${JETSON_IP}/24 dev $INTERFACE
check_command "Asignación de IP ${JETSON_IP}"

# Configurar parámetros del enlace Ethernet
sudo ethtool -s $INTERFACE speed 100 duplex full autoneg off 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Configuración Ethernet optimizada"
else
    echo "⚠️  Advertencia: No se pudo optimizar configuración Ethernet (continúa normalmente)"
fi

# Verificar que la interfaz esté activa
ip addr show $INTERFACE | grep -q $JETSON_IP
check_command "Verificación de configuración de red"

echo ""

# ==========================================
# 2. VERIFICACIÓN DE CONECTIVIDAD
# ==========================================
show_progress "Verificando conectividad con la cámara"

# Esperar un momento para que la red se estabilice
sleep 3

# Buscar la cámara en la red
echo "📡 Escaneando red local..."
CAMERA_FOUND=$(sudo arp-scan --interface=$INTERFACE --localnet 2>/dev/null | grep "$CAMERA_IP")

if [ -n "$CAMERA_FOUND" ]; then
    echo "✅ Cámara encontrada:"
    echo "   $CAMERA_FOUND"
else
    echo "❌ Cámara no encontrada en $CAMERA_IP"
    echo ""
    echo "🔧 Intentos de solución:"
    echo "   1. Verificar cable Ethernet"
    echo "   2. Verificar que la cámara esté encendida"
    echo "   3. Esperar 30 segundos para que la cámara inicie completamente"
    echo ""
    read -p "¿Continuar de todas formas? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Verificar conectividad básica
echo "🏓 Probando conectividad..."
if ping -c 2 -W 3 $CAMERA_IP > /dev/null 2>&1; then
    echo "✅ Ping exitoso a $CAMERA_IP"
else
    echo "⚠️  Sin respuesta al ping (puede ser normal si la cámara bloquea ICMP)"
fi

echo ""

# ==========================================
# 3. VERIFICACIÓN DE SERVICIOS
# ==========================================
show_progress "Verificando servicios de la cámara"

# Verificar puertos de la cámara
echo "🔍 Escaneando puertos de la cámara..."
if command -v nmap > /dev/null 2>&1; then
    PORTS=$(nmap -sS --open -p 80,554,8899 $CAMERA_IP 2>/dev/null | grep "open")
    if [ -n "$PORTS" ]; then
        echo "✅ Puertos encontrados:"
        echo "$PORTS" | sed 's/^/   /'
    else
        echo "⚠️  No se detectaron puertos abiertos"
    fi
else
    echo "⚠️  nmap no instalado - saltando verificación de puertos"
fi

# Probar stream RTSP
echo "📹 Verificando stream RTSP..."
if command -v ffprobe > /dev/null 2>&1; then
    STREAM_INFO=$(timeout 10 ffprobe -v quiet -show_entries stream=codec_type,codec_name -of csv=p=0 "$RTSP_URL" 2>/dev/null)
    if [ -n "$STREAM_INFO" ]; then
        echo "✅ Stream RTSP disponible:"
        echo "$STREAM_INFO" | sed 's/^/   /'
    else
        echo "⚠️  No se pudo verificar el stream RTSP"
    fi
else
    echo "⚠️  ffprobe no disponible - saltando verificación de stream"
fi

echo ""

# ==========================================
# 4. INICIO DEL STREAM
# ==========================================
show_progress "Iniciando visualización del stream"

echo "🎬 Configuración del stream:"
echo "   - URL: $RTSP_URL"
echo "   - Calidad: Baja (subtype=1) para mínima latencia"
echo "   - Audio: Deshabilitado"
echo "   - Latencia: ~0 segundos"
echo ""

# Verificar que GStreamer esté disponible
if ! command -v gst-launch-1.0 > /dev/null 2>&1; then
    echo "❌ GStreamer no está instalado"
    echo "💡 Instalar con: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad"
    echo ""
    echo "🔄 Intentando con ffplay como alternativa..."
    
    if command -v ffplay > /dev/null 2>&1; then
        echo "✅ Usando ffplay..."
        ffplay -fflags nobuffer -flags low_delay -framedrop -rtsp_transport udp -an "$RTSP_URL"
    else
        echo "❌ Ni GStreamer ni ffplay están disponibles"
        exit 1
    fi
else
    echo "✅ Iniciando GStreamer..."
    echo "💡 Presiona Ctrl+C para detener"
    echo ""
    
    # Comando GStreamer optimizado
    gst-launch-1.0 rtspsrc location="$RTSP_URL" latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! autovideosink sync=false
fi

echo ""
echo "==============================================="
echo "🏁 FIN DEL SCRIPT"
echo "==============================================="