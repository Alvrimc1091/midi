# Import necessary libraries
import pickle
import os
import re
import time
import numpy as np
from gpiozero import LED
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter

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

# Load the 7 models
BASE_DIR = '/home/pi/midi/models'  # Update the base directory path if needed
model_names = ['svm', 'random_forest', 'logistic_regression', 'knn', 'adb', 'mlp', 'qda']
models = {}

for model_name in model_names:
    model_path = f'{BASE_DIR}/7_models_{model_name}_pipeline.pkl'
    if not os.path.exists(model_path):
        print(f"Warning: Model {model_name} not found at {model_path}. Skipping...")
        continue
    with open(model_path, 'rb') as file:
        models[model_name] = pickle.load(file)

# Function to extract data from CSV
def extract_data_from_csv(csv_path):
    with open(csv_path, 'r') as file:
        line = file.readline().strip()
        if not line:
            raise ValueError(f"CSV file {csv_path} is empty or incorrectly formatted.")
        data_values = re.findall(r'([0-9]*\.[0-9]+|[0-9]+)', line)
        if len(data_values) < 6:
            raise ValueError(f"Insufficient data extracted from {csv_path}.")
        return list(map(float, data_values[-6:]))  # Extract r_mean, g_mean, b_mean, r_var, g_var, b_var

# Function to aggregate the predictions from all models
def aggregate_predictions(predictions):
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0]  # Get the most frequent prediction
    return most_common

# LED mapping dictionary
led_mapping = {
    'jpc': [led_red],
    'fosfato': [led_white],
    'conchuela': [led_yellow],
    'lisina': [led_green],
    'gluten': [led_blue],
    'sesquicarbonato': [led_white, led_red],
}

# Interactive loop
if __name__ == "__main__":
    DATA_DIR = '/home/pi/midi/data/'  # Base directory for data

    while True:
        # Ask for user input
        file_name = input("\nEnter the CSV file name (or type 'q' or 'exit' to quit): ").strip()
        
        if file_name.lower() in ['q', 'exit']:
            print("Exiting program...")
            break
        
        file_path = os.path.join(DATA_DIR, file_name)

        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_name}' does not exist in {DATA_DIR}. Try again.")
            continue

        try:
            # Extract data for prediction
            data_vector = extract_data_from_csv(file_path)
            print(f"Data vector: {data_vector}")

            # Make predictions with all 7 models
            predictions = []
            for model_name, model in models.items():
                prediction = model.predict([data_vector])[0]  # Get prediction for the model
                predictions.append(prediction)
                print(f"Prediction from {model_name}: {prediction}")

            # Aggregate predictions
            final_prediction = aggregate_predictions(predictions)
            print(f"Final aggregated prediction: {final_prediction}")

            # Control LEDs based on prediction
            if final_prediction in led_mapping:
                for led in led_mapping[final_prediction]:
                    led.on()
                time.sleep(2)
                for led in led_mapping[final_prediction]:
                    led.off()
        
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
