# Script para tomar datos de la muestra
import csv
import socket
import math
import time
import threading
import os
import board
from glob import glob
import time
import datetime
import zoneinfo
import numpy as np
import pandas as pd
from ftplib import FTP
from gpiozero import LED
import logging
import pytz
from adafruit_as7341 import AS7341
from picamera2 import Picamera2, Preview
from led_functions import *

# Definición del arreglo de LEDs
led_array = LED(17) # LEDs de iluminación para el sensor

# Zona horaria
zona_santiago = zoneinfo.ZoneInfo("America/Santiago")

# Inicialización del sensor AS7341
sensor = AS7341(board.I2C())

# Valores de Ganancia 

#   Parámetro      ||   Valor
# sensor.gain = 0  ||     0.5
# sensor.gain = 1  ||      1
# sensor.gain = 2  ||      2
# sensor.gain = 3  ||      4
# sensor.gain = 4  ||      8
# sensor.gain = 5  ||      16
# sensor.gain = 6  ||      32
# sensor.gain = 7  ||      64
# sensor.gain = 8  ||      128
# sensor.gain = 9  ||      256
# sensor.gain = 10 ||      512

# Sensor ATIME (Por defecto es 100)
sensor.atime = 29

# Sensor ASTEP (Por defecto es 999)
sensor.astep = 599

# Sensor GAIN (Por defecto es 8 (128))
sensor.gain = 8

# Definición de contador total de muestras
meassurement = 1

# Configuración inicial de la cámara
picam2 = Picamera2()

# FTP server details
ftp_server = "192.168.0.102"
port = 2121
username = "leia"
password = "qwerty"

# Path to meassures
directory_path = "/home/pi/demo/data"

log_file_path = "/home/pi/logs/log_rpiftpsensor.log"

# -------------------------------------------------------------------
# ----------------------- Definición de funciones -------------------
# -------------------------------------------------------------------

# Definición bar_graph()
# Obtiene las lecturas de los 10+1 canales
def bar_graph(read_value):
    scaled = int(read_value / 1000)
    return "[%5d] " % read_value + (scaled * "*")

# Función para obtener los datos del sensor
def datos_sensor():
    # Obtiene las lecturas de los canales y los almacena en un diccionario
    datos_sensor = [
        bar_graph(sensor.channel_415nm),
        bar_graph(sensor.channel_445nm),
        bar_graph(sensor.channel_480nm),
        bar_graph(sensor.channel_515nm),
        bar_graph(sensor.channel_555nm),
        bar_graph(sensor.channel_590nm),
        bar_graph(sensor.channel_630nm),
        bar_graph(sensor.channel_680nm),
        bar_graph(sensor.channel_clear),
        bar_graph(sensor.channel_nir)
    ]
    return datos_sensor

def mostrar_datos():

    datos = datos_sensor()
    hora_santiago = datetime.datetime.now(zona_santiago)
    datos_str = ",".join(map(str, datos))
    foto_id = f"{hora_santiago.strftime('%Y%m%d_%H%M%S')}_foto.jpg"
    guardar_datos(datos_str, foto_id, hora_santiago)
    return datos

def guardar_datos(datos, foto_id, hora_santiago):

    foto_id = f"{hora_santiago.strftime('%Y%m%d_%H%M%S')}_foto.jpg"

    datos = datos.split(",")
    datos.append(foto_id)
    fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")

    with open(f"/home/pi/demo/data/{hora_santiago.strftime('%Y%m%d_%H%M%S')}_data.csv", mode='a', newline='') as archivo_csv:
        escritor_csv = csv.writer(archivo_csv)
        escritor_csv.writerow([fecha_hora] + datos)

def tomar_foto():
    try:
        picam2.start()    
        hora_santiago = datetime.datetime.now(zona_santiago)
        nombre_foto = f"/home/pi/demo/data/{hora_santiago.strftime('%Y%m%d_%H%M%S')}_foto.jpg"
        fecha_hora_actual = hora_santiago.strftime("%Y%m%d_%H%M%S")
        
        picam2.capture_file(nombre_foto)
        #picam2.close()
        time.sleep(1)

        with open(nombre_foto, "rb") as f:
            data = f.read()

    except Exception as e:
        print("Error al tomar la foto")

# Function to get the name of the latest file of a specific type
def get_last_file(directory, file_extension):
    files = glob(os.path.join(directory, f"*.{file_extension}"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0] if files else None


# Function to upload a file to the FTP server
def upload_file(ftp, file_path):
    if file_path:
        remote_path = os.path.basename(file_path)
        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {remote_path}', file)
        print(f"Successfully uploaded {file_path} to {ftp_server}")
    else:
        print("No file to upload.")

# Configuración del logger
def setup_logger():
    logger = logging.getLogger('MeasurementLogger')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def log_with_time(logger, message):
    chile_tz = pytz.timezone('Chile/Continental')
    now = datetime.datetime.now(chile_tz)
    logger.info(f"{now.strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def main():
    logger = setup_logger()

    times = 3

    try: 
        blink_leds(times)

        log_with_time(logger, "Starting the data/photo loop")
        print("Starting the data/photo loop")
        while True:
            try:
                log_with_time(logger, "Starting measure")
                print("Starting measure")
                # Definición del vector de datos totales de la medida
                datos_medida_final = []

                hora_santiago = datetime.datetime.now(zona_santiago)

                # Comienza recopilando los datos de la muestra
                log_with_time(logger, "Cámara Inicializada")
                print("Cámara Inicializada")

                log_with_time(logger, "Iniciando toma de foto y datos de la muestra")
                print("Iniciando toma de foto y datos de la muestra")
                led_array.on()
                time.sleep(1)

                for _ in range(meassurement):
                    try:
                        datos_medida_final.append(mostrar_datos())
                    except Exception as e:
                        log_with_time(logger, f"Error al mostrar datos: {e}")
                        print(f"Error al mostrar datos: {e}")

                log_with_time(logger, "Datos tomados")
                print("Datos tomados")

                try:
                    tomar_foto()
                except Exception as e:
                    log_with_time(logger, f"Error al tomar la foto: {e}")
                    print(f"Error al tomar la foto: {e}")

                log_with_time(logger, "Foto tomada")
                print("Foto tomada")
                print("Juego de luces")
                log_with_time(logger, "Juego de luces")

                secuencia_encendido_leds_cruzado()
                secuencia_encendido_leds_cruzado_inverso()

                log_with_time(logger, "Mostrando datos a continuación")
                print("Mostrando datos a continuación")
                print("Medida tomada:", datos_medida_final[-1:])

                # Get the last .jpg and .csv files
                last_jpg_file = get_last_file(directory_path, "jpg")
                log_with_time(logger, f"The last jpg file is {last_jpg_file}")
                print(f"The last jpg file is {last_jpg_file}")

                last_csv_file = get_last_file(directory_path, "csv")
                log_with_time(logger, f"The last csv file is {last_csv_file}")
                print(f"The last csv file is {last_csv_file}")

            except KeyboardInterrupt:
                log_with_time(logger, "Script interrumpido por el usuario (CTRL+C).")
                print("Script interrumpido por el usuario (CTRL+C).")
                encender_led(led_red)
                break

    except Exception as e:
        log_with_time(logger, f"Se ha producido un error inesperado al iniciar: {e}")
        print(f"Se ha producido un error inesperado al iniciar: {e}")

        
if __name__ == "__main__":
    main()
