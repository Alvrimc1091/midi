# Import necessary libraries
import pickle
import re
import os
from gpiozero import LED
import time
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Define LED pins
led_red = LED(26)  # Rojo -> Alerta
led_green = LED(6)  # Verde -> Trigo
led_blue = LED(22)  # Azul -> Poroto
led_white = LED(5)  # Blanco -> Vacío
led_yellow = LED(27)  # Amarillo -> Maíz

# Define the custom transformer for vector normalization
class VectorNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.apply_along_axis(normalizar_vector, 1, X)

# Function to normalize vectors
def normalizar_vector(vector):
    vector_np = np.array(vector)
    if vector_np.size == 0:
        return vector_np
    magnitud = np.linalg.norm(vector_np)
    if magnitud == 0:
        return vector_np  # Avoid division by zero
    return vector_np / magnitud

# Load the SVM model
BASE_DIR = '/home/pi/demo/models'  # Update the base directory path if needed
svm_model_path = f'{BASE_DIR}/svm_pipeline.pkl'

with open(svm_model_path, 'rb') as file:
    svm_model = pickle.load(file)

# Function to find the most recent CSV file
def find_latest_csv(directory):
    csv_files = [f for f in os.listdir(directory) if re.match(r'^\d{8}_\d{6}_data\.csv$', f)]
    if not csv_files:
        raise FileNotFoundError("No matching CSV files found.")
    latest_file = max(csv_files, key=lambda f: re.match(r'(\d{8}_\d{6})', f).group(1))
    return os.path.join(directory, latest_file)

# Function to extract data from CSV
def extract_data_from_csv(csv_path):
    with open(csv_path, 'r') as file:
        line = file.readline().strip()
        # Extract RGB mean and variance values
        data_values = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', line)
        return list(map(float, data_values[12:18]))  # Extract r_mean, g_mean, b_mean, r_var, g_var, b_var

# Main script
if __name__ == "__main__":
    try:
        # Define the data directory path
        DATA_DIR = '/home/pi/demo/data/'
        
        # Find the latest CSV file
        latest_csv = find_latest_csv(DATA_DIR)
        print(f"Latest CSV file: {latest_csv}")
 
        # Extract data for prediction (r_mean, g_mean, b_mean, r_var, g_var, b_var)
        data_vector = extract_data_from_csv(latest_csv)
        print(f"Data vector: {data_vector}")
        
        # Normalize the vector
        normalized_vector = normalizar_vector(data_vector)
        
        # Make prediction
        prediction = svm_model.predict([normalized_vector])
        
        # Control LED based on prediction
        if prediction == 'jpc':
            led_red.on()
            time.sleep(2)
            led_red.off()
        elif prediction == 'fosfato':
            led_white.on()
            time.sleep(2)
            led_white.off()
        elif prediction == 'conchuela':
            led_yellow.on()
            time.sleep(2)
            led_yellow.off()
        elif prediction == 'lisina':
            led_green.on()
            time.sleep(2)
            led_green.off()
        elif prediction == 'gluten':
            led_blue.on()
            time.sleep(2)
            led_blue.off()
        elif prediction == 'sesquicarbonato':
            led_white.on()
            led_red.on()
            time.sleep(2)
            led_white.off()
            led_red.off()
        
        print(f"Prediction: {prediction}")

    except Exception as e:
        print(f"Error: {e}")
