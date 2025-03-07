import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from datetime import datetime, timedelta
import csv
import socket
import math
import time
import threading
import board
from glob import glob
import time
import datetime
import zoneinfo
from ftplib import FTP
from gpiozero import LED
import logging
import pytz
from adafruit_as7341 import AS7341
from picamera2 import Picamera2, Preview
from led_functions import *

matplotlib.use("Agg")  # Use non-GUI backend

# Definición del arreglo de LEDs
led_array = LED(17)  # LEDs de iluminación para el sensor

# Zona horaria
zona_santiago = zoneinfo.ZoneInfo("America/Santiago")

# Inicialización del sensor AS7341
sensor = AS7341(board.I2C())

meassurement = 1

# Valores de Ganancia
sensor.atime = 29
sensor.astep = 599
sensor.gain = 8

# Configuración inicial de la cámara
picam2 = Picamera2()

# FTP server details
ftp_server = "192.168.0.102"
port = 2121
username = "leia"
password = "qwerty"

# Path to meassures
directory_path = "/home/pi/midi/data/"
numpy_folder = directory_path + 'numpy_files/'
log_file_path = "/home/pi/midi/logs/log_rpiftpsensor.log"

os.makedirs(numpy_folder, exist_ok=True)

# -------------------------------------------------------------------
# ----------------------- Definición de funciones -------------------
# -------------------------------------------------------------------

# Definición bar_graph()
def bar_graph(read_value):
    scaled = int(read_value / 1000)
    return "[%5d] " % read_value + (scaled * "*")

# Función para obtener los datos del sensor
def datos_sensor():
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

#def mostrar_datos():
#    datos = datos_sensor()
#    hora_santiago = datetime.datetime.now(zona_santiago)
#    datos_str = ",".join(map(str, datos))
#    foto_id = f"{hora_santiago.strftime('%Y%m%d_%H%M%S')}_foto.png"
#    guardar_datos(datos_str, foto_id, hora_santiago, npy_file, r_mean, g_mean, b_mean, r_var, g_var, b_var)
#    return datos

def tomar_foto():
    try:
        picam2.start()
        hora_santiago = datetime.datetime.now(zona_santiago)
        nombre_foto = f"/home/pi/midi/data/{hora_santiago.strftime('%Y%m%d_%H%M%S')}_foto.png"
        
        picam2.capture_file(nombre_foto)

        # Load image
        image = cv2.imread(nombre_foto)
        if image is None:
            print(f"Error: Could not load {nombre_foto}, skipping...")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Define the scale factor 
        scale_factor = 1  # Change to adjust the scale

        # Calculate the new image dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Define non-integer center and radius
        center_x = width / 2.02  # Example: keep it centered
        center_y = height * 0.39 # Move the circle up slightly
        radius = min(width, height) / 3.8  # Adjust the radius proportionally

        # Convert to integers for OpenCV
        center = (int(center_x), int(center_y))
        radius = int(radius)

        cv2.circle(mask, center, radius, 255, thickness=-1)
        
        # Apply mask
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

        # Scaled image
        scaled_image = cv2.resize(src=masked_image, 
                        dsize=(new_width, new_height), 
                        interpolation=cv2.INTER_AREA)

        # Save masked image
        masked_image_bgr = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)

        masked_image_path = os.path.join(directory_path, f'{hora_santiago.strftime("%Y%m%d_%H%M%S")}_masked.png')
        success = cv2.imwrite(masked_image_path, masked_image_bgr)

        #scaled_image_path = os.path.join(directory_path, f'{nombre_foto}_mnsb.png')
        
        #cv2.imwrite(masked_image_path, cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))  # Save masked image in BGR format

        output_image_path = os.path.join(directory_path, f'{nombre_foto}')
        print(output_image_path)
        plt.imshow(scaled_image)
        plt.axis('off')
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

        # Normalize the image
        b, g, r = cv2.split(scaled_image)
        b_norm = cv2.normalize(b.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        g_norm = cv2.normalize(g.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        r_norm = cv2.normalize(r.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
        normalized_image = cv2.merge((b_norm, g_norm, r_norm))
        
        # Compute mean and variance for each channel
        r_mean, g_mean, b_mean = np.mean(r_norm), np.mean(g_norm), np.mean(b_norm)
        r_var, g_var, b_var = np.var(r_norm), np.var(g_norm), np.var(b_norm)

        # Save numpy file for future processing
        output_npy_path = os.path.join(numpy_folder, f'{hora_santiago.strftime("%Y%m%d_%H%M%S")}_normalized.npy')
        np.save(output_npy_path, normalized_image)

        # Call guardar_datos to save all data in the CSV
        datos_sensor_values = datos_sensor()  # Get sensor data
        guardar_datos(
            ','.join(datos_sensor_values),  # Sensor data as comma-separated string
            nombre_foto,  # Image file path
            hora_santiago,  # Timestamp
            output_npy_path,  # Numpy file path
            r_mean, g_mean, b_mean, r_var, g_var, b_var  # Stats for the image
        )

        print(f"Processed {nombre_foto}: R_mean={r_mean:.5f}, G_mean={g_mean:.5f}, B_mean={b_mean:.5f}, "
              f"R_var={r_var:.5f}, G_var={g_var:.5f}, B_var={b_var:.5f}")

    except Exception as e:
        print(f"Error al tomar la foto: {e}")


def guardar_datos(datos, foto_id, hora_santiago, npy_file, r_mean, g_mean, b_mean, r_var, g_var, b_var):
    try:
        # Prepare the data
        datos = datos.split(",")  # Split sensor data into list
        datos.append(foto_id)  # Add image path
        datos.append(npy_file)  # Add numpy file path
        datos.append(f"{r_mean:.5f}")  # Add R mean
        datos.append(f"{g_mean:.5f}")  # Add G mean
        datos.append(f"{b_mean:.5f}")  # Add B mean
        datos.append(f"{r_var:.5f}")  # Add R variance
        datos.append(f"{g_var:.5f}")  # Add G variance
        datos.append(f"{b_var:.5f}")  # Add B variance        
        
        fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
        
        # Save everything in the same row
        with open(f"/home/pi/midi/data/{hora_santiago.strftime('%Y%m%d_%H%M%S')}_data.csv", mode='a', newline='') as archivo_csv:
            escritor_csv = csv.writer(archivo_csv)
            escritor_csv.writerow([fecha_hora] + datos)

    except Exception as e:
        print(f"Error al guardar los datos: {e}")


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

    times = 1

    try: 
        blink_leds(times)

        log_with_time(logger, "Starting the data/photo loop")
        print("Starting the data/photo loop")
        
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

            #for _ in range(meassurement):
            #     try:
            #        datos_medida_final.append(mostrar_datos())
            #     except Exception as e:
            #        log_with_time(logger, f"Error al mostrar datos: {e}")
            #        print(f"Error al mostrar datos: {e}")

            log_with_time(logger, "Datos tomados")
            print("Datos tomados")

            try:
                #pass
                tomar_foto()
            except Exception as e:
                log_with_time(logger, f"Error al tomar la foto: {e}")
                print(f"Error al tomar la foto: {e}")

            log_with_time(logger, "Foto tomada")
            print("Foto tomada")
            print("Juego de luces")
            log_with_time(logger, "Juego de luces")

#            secuencia_encendido_leds_cruzado()
#            secuencia_encendido_leds_cruzado_inverso()

            #log_with_time(logger, "Mostrando datos a continuación")
            #print("Mostrando datos a continuación")
            #print("Medida tomada:", datos_medida_final[-1:])

            # Get the last .png and .csv files
            last_png_file = get_last_file(directory_path, "png")
            log_with_time(logger, f"The last png file is {last_png_file}")
            print(f"The last png file is {last_png_file}")

            last_csv_file = get_last_file(directory_path, "csv")
            log_with_time(logger, f"The last csv file is {last_csv_file}")
            print(f"The last csv file is {last_csv_file}")

        except KeyboardInterrupt:
            log_with_time(logger, "Script interrumpido por el usuario (CTRL+C).")
            print("Script interrumpido por el usuario (CTRL+C).")
            encender_led(led_red)
        

    except Exception as e:
        log_with_time(logger, f"Se ha producido un error inesperado al iniciar: {e}")
        print(f"Se ha producido un error inesperado al iniciar: {e}")

if __name__ == "__main__":
    while True:
           
          command = input("Enter a command: ").strip().lower()

          if command == "m":
               main()
          elif command == "q":
               break
          else:
               print("Unknown command!")
