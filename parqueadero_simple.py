#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Gestión de Parqueadero - Versión Simplificada
Compatible con Jetson Orin Nano
"""

import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import re

class ParqueaderoSimple:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.root = tk.Tk()
        self.setup_window()
        self.conectar_db()
        self.create_main_interface()
        
    def setup_window(self):
        """Configura la ventana principal"""
        self.root.title("Sistema de Parqueadero")
        self.root.geometry("900x600")
        
        # Centrar ventana
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (600 // 2)
        self.root.geometry(f"900x600+{x}+{y}")
        
    def conectar_db(self):
        """Conecta a la base de datos MySQL"""
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                database='parqueadero',
                user='usuario_Parqueadero',
                password='Sena2025!'
            )
            if self.connection.is_connected():
                self.cursor = self.connection.cursor()
                return True
        except Error as e:
            messagebox.showerror("Error", f"No se pudo conectar a la base de datos:\n{e}")
            return False
    
    def create_main_interface(self):
        """Crea la interfaz principal"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Título
        title_label = tk.Label(
            main_frame, 
            text="🏢 SISTEMA DE PARQUEADERO",
            font=("Arial", 18, "bold"),
            fg="#2E86AB"
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Notebook para pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Crear pestañas
        self.create_entrada_tab()
        self.create_salida_tab()
        self.create_usuarios_tab()
        self.create_vehiculos_tab()
        self.create_consultas_tab()
        
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def create_entrada_tab(self):
        """Pestaña de entrada de vehículos"""
        entrada_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(entrada_frame, text="🚪 Entrada")
        
        # Título
        tk.Label(entrada_frame, text="ENTRADA DE VEHÍCULO", 
                font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Campo placa
        tk.Label(entrada_frame, text="Placa:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.entrada_placa_var = tk.StringVar()
        entrada_placa_entry = tk.Entry(entrada_frame, textvariable=self.entrada_placa_var, 
                                     font=("Arial", 12), width=20)
        entrada_placa_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        # Botón procesar
        btn_entrada = tk.Button(entrada_frame, text="REGISTRAR ENTRADA", 
                              command=self.procesar_entrada,
                              font=("Arial", 12, "bold"), bg="#28A745", fg="white",
                              padx=20, pady=5)
        btn_entrada.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Resultado
        self.entrada_resultado = tk.Text(entrada_frame, height=15, width=70, font=("Courier", 10))
        self.entrada_resultado.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Scrollbar para resultado
        scroll_entrada = ttk.Scrollbar(entrada_frame, orient="vertical", command=self.entrada_resultado.yview)
        scroll_entrada.grid(row=3, column=2, sticky=(tk.N, tk.S))
        self.entrada_resultado.configure(yscrollcommand=scroll_entrada.set)
    
    def create_salida_tab(self):
        """Pestaña de salida de vehículos"""
        salida_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(salida_frame, text="🚪 Salida")
        
        # Título
        tk.Label(salida_frame, text="SALIDA DE VEHÍCULO", 
                font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Campo placa
        tk.Label(salida_frame, text="Placa:", font=("Arial", 12)).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.salida_placa_var = tk.StringVar()
        salida_placa_entry = tk.Entry(salida_frame, textvariable=self.salida_placa_var, 
                                    font=("Arial", 12), width=20)
        salida_placa_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        # Botón procesar
        btn_salida = tk.Button(salida_frame, text="REGISTRAR SALIDA", 
                             command=self.procesar_salida,
                             font=("Arial", 12, "bold"), bg="#DC3545", fg="white",
                             padx=20, pady=5)
        btn_salida.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Resultado
        self.salida_resultado = tk.Text(salida_frame, height=15, width=70, font=("Courier", 10))
        self.salida_resultado.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Scrollbar para resultado
        scroll_salida = ttk.Scrollbar(salida_frame, orient="vertical", command=self.salida_resultado.yview)
        scroll_salida.grid(row=3, column=2, sticky=(tk.N, tk.S))
        self.salida_resultado.configure(yscrollcommand=scroll_salida.set)
    
    def create_usuarios_tab(self):
        """Pestaña de registro de usuarios"""
        usuarios_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(usuarios_frame, text="👤 Usuarios")
        
        # Título
        tk.Label(usuarios_frame, text="REGISTRO DE USUARIO", 
                font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Campos
        campos = [("Nombre:", "nombre"), ("Documento:", "documento"), 
                 ("Teléfono:", "telefono"), ("Correo:", "correo")]
        
        self.usuario_vars = {}
        for i, (label, var_name) in enumerate(campos):
            tk.Label(usuarios_frame, text=label, font=("Arial", 12)).grid(
                row=i+1, column=0, sticky=tk.W, pady=5)
            var = tk.StringVar()
            entry = tk.Entry(usuarios_frame, textvariable=var, font=("Arial", 12), width=30)
            entry.grid(row=i+1, column=1, sticky=tk.W, pady=5, padx=10)
            self.usuario_vars[var_name] = var
        
        # Botón guardar
        btn_usuario = tk.Button(usuarios_frame, text="GUARDAR USUARIO", 
                              command=self.guardar_usuario,
                              font=("Arial", 12, "bold"), bg="#17A2B8", fg="white",
                              padx=20, pady=5)
        btn_usuario.grid(row=len(campos)+1, column=0, columnspan=2, pady=20)
    
    def create_vehiculos_tab(self):
        """Pestaña de registro de vehículos"""
        vehiculos_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(vehiculos_frame, text="🚗 Vehículos")
        
        # Título
        tk.Label(vehiculos_frame, text="REGISTRO DE VEHÍCULO", 
                font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Documento propietario
        tk.Label(vehiculos_frame, text="Documento propietario:", font=("Arial", 12)).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.doc_prop_var = tk.StringVar()
        doc_entry = tk.Entry(vehiculos_frame, textvariable=self.doc_prop_var, font=("Arial", 12), width=20)
        doc_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
        
        btn_buscar = tk.Button(vehiculos_frame, text="Buscar", command=self.buscar_propietario,
                             bg="#6C757D", fg="white")
        btn_buscar.grid(row=1, column=2, pady=5, padx=5)
        
        # Info propietario
        self.info_prop_label = tk.Label(vehiculos_frame, text="", font=("Arial", 10))
        self.info_prop_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Campos vehículo
        campos_veh = [("Placa:", "placa"), ("Marca:", "marca"), ("Modelo:", "modelo"),
                     ("SOAT número:", "soat_num"), ("SOAT venc (YYYY-MM-DD):", "soat_venc"),
                     ("Tecno número:", "tecno_num"), ("Tecno venc (YYYY-MM-DD):", "tecno_venc")]
        
        self.vehiculo_vars = {}
        for i, (label, var_name) in enumerate(campos_veh):
            tk.Label(vehiculos_frame, text=label, font=("Arial", 12)).grid(
                row=i+3, column=0, sticky=tk.W, pady=2)
            var = tk.StringVar()
            entry = tk.Entry(vehiculos_frame, textvariable=var, font=("Arial", 12), width=25)
            entry.grid(row=i+3, column=1, sticky=tk.W, pady=2, padx=10)
            self.vehiculo_vars[var_name] = var
        
        # Botón guardar
        btn_vehiculo = tk.Button(vehiculos_frame, text="GUARDAR VEHÍCULO", 
                               command=self.guardar_vehiculo,
                               font=("Arial", 12, "bold"), bg="#FFC107", fg="black",
                               padx=20, pady=5)
        btn_vehiculo.grid(row=len(campos_veh)+3, column=0, columnspan=2, pady=20)
        
        self.id_propietario = None
    
    def create_consultas_tab(self):
        """Pestaña de consultas"""
        consultas_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(consultas_frame, text="📋 Consultas")
        
        # Título
        tk.Label(consultas_frame, text="CONSULTAS Y REPORTES", 
                font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Botones de consulta
        btn_actuales = tk.Button(consultas_frame, text="Vehículos en Parqueadero", 
                               command=self.mostrar_vehiculos_actuales,
                               font=("Arial", 11), bg="#007BFF", fg="white", width=25, pady=5)
        btn_actuales.grid(row=1, column=0, pady=5, padx=5)
        
        btn_estadisticas = tk.Button(consultas_frame, text="Estadísticas", 
                                   command=self.mostrar_estadisticas,
                                   font=("Arial", 11), bg="#6F42C1", fg="white", width=25, pady=5)
        btn_estadisticas.grid(row=1, column=1, pady=5, padx=5)
        
        # Campo para historial por placa
        tk.Label(consultas_frame, text="Historial por placa:", font=("Arial", 12)).grid(
            row=2, column=0, sticky=tk.W, pady=(20, 5))
        self.hist_placa_var = tk.StringVar()
        hist_entry = tk.Entry(consultas_frame, textvariable=self.hist_placa_var, font=("Arial", 12), width=15)
        hist_entry.grid(row=2, column=1, sticky=tk.W, pady=(20, 5), padx=10)
        
        btn_historial = tk.Button(consultas_frame, text="Ver Historial", 
                                command=self.mostrar_historial,
                                bg="#20C997", fg="white")
        btn_historial.grid(row=2, column=2, pady=(20, 5), padx=5)
        
        # Área de resultados
        self.consulta_resultado = tk.Text(consultas_frame, height=20, width=80, font=("Courier", 9))
        self.consulta_resultado.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Scrollbar para consultas
        scroll_consulta = ttk.Scrollbar(consultas_frame, orient="vertical", command=self.consulta_resultado.yview)
        scroll_consulta.grid(row=3, column=3, sticky=(tk.N, tk.S))
        self.consulta_resultado.configure(yscrollcommand=scroll_consulta.set)
    
    def procesar_entrada(self):
        """Procesa entrada de vehículo"""
        placa = self.entrada_placa_var.get().strip().upper()
        
        if not self.validar_placa(placa):
            messagebox.showerror("Error", "Formato de placa inválido")
            return
        
        try:
            vehiculo = self.buscar_vehiculo(placa)
            
            if not vehiculo:
                messagebox.showerror("Error", "Vehículo no encontrado. Debe registrarse primero.")
                return
            
            id_vehiculo, placa, marca, modelo, nombre, telefono, documento = vehiculo
            ultimo_estado = self.obtener_ultimo_estado(id_vehiculo)
            
            if ultimo_estado and ultimo_estado[0] == 'entrada':
                resultado = f"VEHÍCULO YA EN PARQUEADERO\n" + "="*40 + "\n"
                resultado += f"Placa: {placa}\nVehículo: {marca} {modelo}\n"
                resultado += f"Propietario: {nombre}\nEntrada: {ultimo_estado[1]}\n"
                self.entrada_resultado.delete(1.0, tk.END)
                self.entrada_resultado.insert(tk.END, resultado)
                messagebox.showwarning("Advertencia", f"El vehículo {placa} ya está en el parqueadero")
                return
            
            # Registrar entrada
            fecha_actual = datetime.now()
            query = "INSERT INTO estado (fecha, estado, idVehiculo) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (fecha_actual, 'entrada', id_vehiculo))
            self.connection.commit()
            
            resultado = f"ENTRADA REGISTRADA\n" + "="*30 + "\n"
            resultado += f"Placa: {placa}\nVehículo: {marca} {modelo}\n"
            resultado += f"Propietario: {nombre}\nTeléfono: {telefono}\n"
            resultado += f"Fecha: {fecha_actual.strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            self.entrada_resultado.delete(1.0, tk.END)
            self.entrada_resultado.insert(tk.END, resultado)
            self.entrada_placa_var.set("")
            
            messagebox.showinfo("Éxito", f"Entrada registrada para {placa}")
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def procesar_salida(self):
        """Procesa salida de vehículo"""
        placa = self.salida_placa_var.get().strip().upper()
        
        if not self.validar_placa(placa):
            messagebox.showerror("Error", "Formato de placa inválido")
            return
        
        try:
            vehiculo = self.buscar_vehiculo(placa)
            
            if not vehiculo:
                messagebox.showerror("Error", "Vehículo no encontrado")
                return
            
            id_vehiculo, placa, marca, modelo, nombre, telefono, documento = vehiculo
            ultimo_estado = self.obtener_ultimo_estado(id_vehiculo)
            
            if not ultimo_estado or ultimo_estado[0] == 'salida':
                resultado = f"VEHÍCULO NO ESTÁ EN PARQUEADERO\n" + "="*35 + "\n"
                resultado += f"Placa: {placa}\nVehículo: {marca} {modelo}\n"
                resultado += f"Propietario: {nombre}\n"
                self.salida_resultado.delete(1.0, tk.END)
                self.salida_resultado.insert(tk.END, resultado)
                messagebox.showwarning("Advertencia", f"El vehículo {placa} no está en el parqueadero")
                return
            
            # Registrar salida
            fecha_actual = datetime.now()
            fecha_entrada = ultimo_estado[1]
            
            if isinstance(fecha_entrada, str):
                fecha_entrada = datetime.strptime(fecha_entrada, '%Y-%m-%d %H:%M:%S')
            
            tiempo_permanencia = fecha_actual - fecha_entrada
            horas = int(tiempo_permanencia.total_seconds() // 3600)
            minutos = int((tiempo_permanencia.total_seconds() % 3600) // 60)
            
            query = "INSERT INTO estado (fecha, estado, idVehiculo) VALUES (%s, %s, %s)"
            self.cursor.execute(query, (fecha_actual, 'salida', id_vehiculo))
            self.connection.commit()
            
            resultado = f"SALIDA REGISTRADA\n" + "="*25 + "\n"
            resultado += f"Placa: {placa}\nVehículo: {marca} {modelo}\n"
            resultado += f"Propietario: {nombre}\nTeléfono: {telefono}\n"
            resultado += f"Entrada: {fecha_entrada.strftime('%Y-%m-%d %H:%M:%S')}\n"
            resultado += f"Salida: {fecha_actual.strftime('%Y-%m-%d %H:%M:%S')}\n"
            resultado += f"Permanencia: {horas}h {minutos}m\n"
            
            self.salida_resultado.delete(1.0, tk.END)
            self.salida_resultado.insert(tk.END, resultado)
            self.salida_placa_var.set("")
            
            messagebox.showinfo("Éxito", f"Salida registrada para {placa}")
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def guardar_usuario(self):
        """Guarda nuevo usuario"""
        try:
            nombre = self.usuario_vars['nombre'].get().strip()
            documento = self.usuario_vars['documento'].get().strip()
            telefono = self.usuario_vars['telefono'].get().strip()
            correo = self.usuario_vars['correo'].get().strip()
            
            if not all([nombre, documento, telefono, correo]):
                messagebox.showerror("Error", "Todos los campos son obligatorios")
                return
            
            if not documento.isdigit():
                messagebox.showerror("Error", "El documento debe contener solo números")
                return
            
            if not telefono.isdigit():
                messagebox.showerror("Error", "El teléfono debe contener solo números")
                return
            
            if not self.validar_email(correo):
                messagebox.showerror("Error", "Formato de correo inválido")
                return
            
            # Verificar duplicados
            self.cursor.execute("SELECT documento FROM usuario WHERE documento = %s", (documento,))
            if self.cursor.fetchone():
                messagebox.showerror("Error", "Ya existe un usuario con ese documento")
                return
            
            # Insertar
            query = "INSERT INTO usuario (nombre, documento, telefono, correo) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(query, (nombre, documento, telefono, correo))
            self.connection.commit()
            
            messagebox.showinfo("Éxito", f"Usuario registrado exitosamente\nID: {self.cursor.lastrowid}")
            
            # Limpiar campos
            for var in self.usuario_vars.values():
                var.set("")
                
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def buscar_propietario(self):
        """Busca propietario por documento"""
        documento = self.doc_prop_var.get().strip()
        
        if not documento:
            messagebox.showerror("Error", "Ingrese el documento")
            return
        
        try:
            self.cursor.execute("SELECT idUsuario, nombre, telefono FROM usuario WHERE documento = %s", (documento,))
            usuario = self.cursor.fetchone()
            
            if usuario:
                self.id_propietario, nombre, telefono = usuario
                self.info_prop_label.config(text=f"Propietario: {nombre} - Tel: {telefono}", fg="green")
            else:
                self.id_propietario = None
                self.info_prop_label.config(text="Propietario no encontrado", fg="red")
                
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
    
    def guardar_vehiculo(self):
        """Guarda nuevo vehículo"""
        if not self.id_propietario:
            messagebox.showerror("Error", "Debe buscar un propietario válido")
            return
        
        try:
            placa = self.vehiculo_vars['placa'].get().strip().upper()
            marca = self.vehiculo_vars['marca'].get().strip()
            modelo_str = self.vehiculo_vars['modelo'].get().strip()
            soat_num = self.vehiculo_vars['soat_num'].get().strip()
            soat_venc = self.vehiculo_vars['soat_venc'].get().strip()
            tecno_num = self.vehiculo_vars['tecno_num'].get().strip()
            tecno_venc = self.vehiculo_vars['tecno_venc'].get().strip()
            
            if not all([placa, marca, modelo_str, soat_num, soat_venc, tecno_num, tecno_venc]):
                messagebox.showerror("Error", "Todos los campos son obligatorios")
                return
            
            if not self.validar_placa(placa):
                messagebox.showerror("Error", "Formato de placa inválido")
                return
            
            try:
                modelo = int(modelo_str)
                if not (1950 <= modelo <= datetime.now().year + 1):
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Modelo debe ser un año válido")
                return
            
            # Verificar placa duplicada
            self.cursor.execute("SELECT placa FROM vehiculo WHERE placa = %s", (placa,))
            if self.cursor.fetchone():
                messagebox.showerror("Error", "Ya existe un vehículo con esa placa")
                return
            
            # Validar fechas
            try:
                soat_vencimiento = datetime.strptime(soat_venc, "%Y-%m-%d").date()
                tecno_vencimiento = datetime.strptime(tecno_venc, "%Y-%m-%d").date()
            except ValueError:
                messagebox.showerror("Error", "Formato de fecha inválido (YYYY-MM-DD)")
                return
            
            # Insertar vehículo
            query = """INSERT INTO vehiculo 
                      (placa, marca, modelo, soat_numero, soat_vencimiento, 
                       tecno_numero, tecno_vencimiento, idUsuario) 
                      VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            
            valores = (placa, marca, modelo, soat_num, soat_vencimiento, tecno_num, tecno_vencimiento, self.id_propietario)
            self.cursor.execute(query, valores)
            self.connection.commit()
            
            messagebox.showinfo("Éxito", f"Vehículo registrado exitosamente\nPlaca: {placa}")
            
            # Limpiar campos
            for var in self.vehiculo_vars.values():
                var.set("")
            self.doc_prop_var.set("")
            self.info_prop_label.config(text="")
            self.id_propietario = None
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
            if self.connection:
                self.connection.rollback()
    
    def mostrar_vehiculos_actuales(self):
        """Muestra vehículos en parqueadero"""
        try:
            query = """
            SELECT v.placa, v.marca, v.modelo, u.nombre, e.fecha as entrada
            FROM vehiculo v
            JOIN usuario u ON v.idUsuario = u.idUsuario
            JOIN estado e ON v.idVehiculo = e.idVehiculo
            WHERE e.estado = 'entrada'
            AND e.fecha = (
                SELECT MAX(e2.fecha) 
                FROM estado e2 
                WHERE e2.idVehiculo = v.idVehiculo
            )
            AND NOT EXISTS (
                SELECT 1 FROM estado e3 
                WHERE e3.idVehiculo = v.idVehiculo 
                AND e3.estado = 'salida' 
                AND e3.fecha > e.fecha
            )
            ORDER BY e.fecha DESC
            """
            
            self.cursor.execute(query)
            vehiculos = self.cursor.fetchall()
            
            resultado = f"VEHÍCULOS EN PARQUEADERO ({len(vehiculos)})\n" + "="*50 + "\n\n"
            
            if not vehiculos:
                resultado += "No hay vehículos en el parqueadero\n"
            else:
                for i, (placa, marca, modelo, nombre, entrada) in enumerate(vehiculos, 1):
                    if isinstance(entrada, str):
                        entrada = datetime.strptime(entrada, '%Y-%m-%d %H:%M:%S')
                    
                    tiempo_permanencia = datetime.now() - entrada
                    horas = int(tiempo_permanencia.total_seconds() // 3600)
                    minutos = int((tiempo_permanencia.total_seconds() % 3600) // 60)
                    
                    resultado += f"{i:2d}. {placa} - {marca} {modelo}\n"
                    resultado += f"    Propietario: {nombre}\n"
                    resultado += f"    Entrada: {entrada.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    resultado += f"    Permanencia: {horas}h {minutos}m\n\n"
            
            self.consulta_resultado.delete(1.0, tk.END)
            self.consulta_resultado.insert(tk.END, resultado)
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas del parqueadero"""
        try:
            # Obtener estadísticas
            self.cursor.execute("SELECT COUNT(*) FROM usuario")
            total_usuarios = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM vehiculo")
            total_vehiculos = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM estado")
            total_movimientos = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(*) FROM estado WHERE DATE(fecha) = CURDATE()")
            movimientos_hoy = self.cursor.fetchone()[0]
            
            # Vehículos en parqueadero
            query_en_parqueadero = """
            SELECT COUNT(DISTINCT v.idVehiculo)
            FROM vehiculo v
            JOIN estado e ON v.idVehiculo = e.idVehiculo
            WHERE e.estado = 'entrada'
            AND e.fecha = (
                SELECT MAX(e2.fecha) 
                FROM estado e2 
                WHERE e2.idVehiculo = v.idVehiculo
            )
            AND NOT EXISTS (
                SELECT 1 FROM estado e3 
                WHERE e3.idVehiculo = v.idVehiculo 
                AND e3.estado = 'salida' 
                AND e3.fecha > e.fecha
            )
            """
            self.cursor.execute(query_en_parqueadero)
            en_parqueadero = self.cursor.fetchone()[0]
            
            resultado = "ESTADÍSTICAS DEL PARQUEADERO\n" + "="*40 + "\n\n"
            resultado += f"👥 Total usuarios registrados: {total_usuarios}\n"
            resultado += f"🚗 Total vehículos registrados: {total_vehiculos}\n"
            resultado += f"🏢 Vehículos en parqueadero: {en_parqueadero}\n"
            resultado += f"📊 Total movimientos históricos: {total_movimientos}\n"
            resultado += f"📅 Movimientos hoy: {movimientos_hoy}\n\n"
            
            # Estadísticas adicionales
            self.cursor.execute("""
                SELECT COUNT(*) FROM vehiculo 
                WHERE soat_vencimiento BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY)
            """)
            soat_vencer = self.cursor.fetchone()[0]
            
            self.cursor.execute("""
                SELECT COUNT(*) FROM vehiculo 
                WHERE tecno_vencimiento BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY)
            """)
            tecno_vencer = self.cursor.fetchone()[0]
            
            resultado += "⚠️ ALERTAS:\n"
            resultado += f"   SOAT por vencer (30 días): {soat_vencer}\n"
            resultado += f"   Tecnomecánica por vencer (30 días): {tecno_vencer}\n"
            
            self.consulta_resultado.delete(1.0, tk.END)
            self.consulta_resultado.insert(tk.END, resultado)
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
    
    def mostrar_historial(self):
        """Muestra historial de vehículo por placa"""
        placa = self.hist_placa_var.get().strip().upper()
        
        if not placa:
            messagebox.showerror("Error", "Ingrese una placa")
            return
        
        try:
            # Buscar vehículo
            vehiculo = self.buscar_vehiculo(placa)
            if not vehiculo:
                messagebox.showerror("Error", "Vehículo no encontrado")
                return
            
            _, placa, marca, modelo, nombre, telefono, documento = vehiculo
            
            # Obtener historial
            query = """
            SELECT fecha, estado
            FROM estado e
            JOIN vehiculo v ON e.idVehiculo = v.idVehiculo
            WHERE v.placa = %s
            ORDER BY fecha DESC
            LIMIT 30
            """
            
            self.cursor.execute(query, (placa,))
            registros = self.cursor.fetchall()
            
            resultado = f"HISTORIAL DE {placa}\n" + "="*30 + "\n"
            resultado += f"Vehículo: {marca} {modelo}\n"
            resultado += f"Propietario: {nombre}\n"
            resultado += f"Teléfono: {telefono}\n\n"
            
            if not registros:
                resultado += "Sin historial de movimientos\n"
            else:
                resultado += "ÚLTIMOS 30 MOVIMIENTOS:\n" + "-"*35 + "\n"
                
                for i, (fecha, estado) in enumerate(registros):
                    duracion = ""
                    
                    # Calcular duración si es salida
                    if estado == 'salida' and i < len(registros) - 1:
                        fecha_siguiente = registros[i + 1][0]
                        if registros[i + 1][1] == 'entrada':
                            if isinstance(fecha, str):
                                fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
                            if isinstance(fecha_siguiente, str):
                                fecha_siguiente = datetime.strptime(fecha_siguiente, '%Y-%m-%d %H:%M:%S')
                            
                            diferencia = fecha - fecha_siguiente
                            horas = int(diferencia.total_seconds() // 3600)
                            minutos = int((diferencia.total_seconds() % 3600) // 60)
                            duracion = f" ({horas}h {minutos}m)"
                    
                    fecha_str = fecha.strftime('%Y-%m-%d %H:%M:%S') if isinstance(fecha, datetime) else str(fecha)
                    estado_icon = "🟢" if estado == 'entrada' else "🔴"
                    
                    resultado += f"{estado_icon} {estado.upper():8} - {fecha_str}{duracion}\n"
            
            self.consulta_resultado.delete(1.0, tk.END)
            self.consulta_resultado.insert(tk.END, resultado)
            
        except Error as e:
            messagebox.showerror("Error", f"Error: {e}")
    
    # Métodos auxiliares
    def validar_placa(self, placa):
        """Valida formato de placa colombiana"""
        patron = r'^[A-Z]{3}[0-9]{2}[0-9A-Z]$'
        return re.match(patron, placa.upper()) is not None
    
    def validar_email(self, email):
        """Valida formato de email"""
        patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(patron, email) is not None
    
    def buscar_vehiculo(self, placa):
        """Busca un vehículo por placa"""
        try:
            query = """SELECT v.idVehiculo, v.placa, v.marca, v.modelo, 
                             u.nombre, u.telefono, u.documento
                      FROM vehiculo v
                      JOIN usuario u ON v.idUsuario = u.idUsuario
                      WHERE v.placa = %s"""
            
            self.cursor.execute(query, (placa,))
            return self.cursor.fetchone()
        except Error as e:
            messagebox.showerror("Error", f"Error buscando vehículo: {e}")
            return None
    
    def obtener_ultimo_estado(self, id_vehiculo):
        """Obtiene el último estado de un vehículo"""
        try:
            query = """SELECT estado, fecha FROM estado 
                      WHERE idVehiculo = %s 
                      ORDER BY fecha DESC LIMIT 1"""
            
            self.cursor.execute(query, (id_vehiculo,))
            return self.cursor.fetchone()
        except Error as e:
            messagebox.showerror("Error", f"Error obteniendo estado: {e}")
            return None
    
    def run(self):
        """Ejecuta la aplicación"""
        if self.connection:
            try:
                self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
                self.root.mainloop()
            except Exception as e:
                messagebox.showerror("Error", f"Error ejecutando aplicación: {e}")
        else:
            messagebox.showerror("Error", "No se pudo conectar a la base de datos")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        if messagebox.askokcancel("Salir", "¿Está seguro que desea salir?"):
            if self.connection and self.connection.is_connected():
                self.cursor.close()
                self.connection.close()
            self.root.destroy()

def main():
    """Función principal"""
    try:
        app = ParqueaderoSimple()
        app.run()
    except Exception as e:
        print(f"Error iniciando aplicación: {e}")

if __name__ == "__main__":
    main()
