#!/usr/bin/env python3
"""
🎯 QUICK START LPR - Un solo comando para todo
"""

def main():
    print("🚀 LPR QUICK START")
    print("=" * 30)
    
    # Detectar entorno
    import platform
    import os
    
    if os.path.exists('/etc/nv_tegra_release'):
        print("🤖 Jetson Orin Nano detectada")
        mode = 'production'
    else:
        print("💻 PC de desarrollo detectado")
        mode = 'development'
    
    print(f"Modo automático: {mode}")
    
    # Confirmar
    response = input("\n¿Ejecutar instalación y demo? (S/n): ")
    if response.lower() == 'n':
        print("❌ Cancelado")
        return
    
    # Ejecutar instalador
    print("\n🔄 Ejecutando instalador...")
    
    import subprocess
    import sys
    
    try:
        # Instalar
        result = subprocess.run([
            sys.executable, 'install_lpr_system.py', 
            '--mode', mode
        ], check=True)
        
        print("\n✅ INSTALACIÓN COMPLETADA")
        
        # Preguntar si ejecutar demo
        demo_response = input("\n¿Ejecutar demo del stream? (S/n): ")
        if demo_response.lower() != 'n':
            print("\n🎬 Iniciando demo...")
            subprocess.run([sys.executable, 'demo_stream.py'])
        
    except subprocess.CalledProcessError:
        print("\n❌ Error en instalación")
        print("💡 Ejecutar manualmente: python install_lpr_system.py --mode", mode)

if __name__ == "__main__":
    main()
