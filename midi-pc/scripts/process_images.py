import cv2  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from datetime import datetime, timedelta

matplotlib.use("Agg")  # Use non-GUI backend

# Base directory path
leia_file_path = '/home/alvaro/Documents/Workspace/midi/data/static_training/'

# Folder containing images (example: "2025-02-04")
input_folder = leia_file_path + 'sesquicarbonato'
data_models_path = leia_file_path #+ 'data_models'

# Extract date from the folder name (assuming format YYYY-MM-DD)
folder_name = os.path.basename(os.path.normpath(input_folder))
date_formatted = folder_name.replace("-", "_")  # Convert YYYY-MM-DD to YYYY_MM_DD
date_formatted_empty = folder_name.replace("-", "")  # Convert YYYY-MM-DD to YYYYMMDD

# Define output directories
output_base_path = leia_file_path + f'camera/data/files_mnsb_data/sesquicarbonato/'
masked_folder = output_base_path + 'mnsb_files/'
numpy_folder = output_base_path + 'numpy_files/'
csv_output_folder = leia_file_path + '/camera/data/csvs_mnsb_data/'

# Ensure directories exist
os.makedirs(masked_folder, exist_ok=True)
os.makedirs(numpy_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

# Load CSV file
csv_file_path = os.path.join(data_models_path, f'sesquicarbonato_data.csv')
if not os.path.exists(csv_file_path):
    print(f"Error: CSV file {csv_file_path} not found!")
    exit()

df = pd.read_csv(csv_file_path)

# Ensure the last column is named 'photo_id'
if 'photo_id' not in df.columns:
    print("Error: CSV file does not contain 'photo_id' column!")
    exit()

# Get list of available images
available_images = set(os.listdir(input_folder))

def find_closest_image(photo_id):
    """Find the closest matching image file by adjusting the seconds."""
    base_name, ext = os.path.splitext(photo_id)
    try:
        dt = datetime.strptime(base_name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    
    for delta in [0, -1, 1]:
        adjusted_name = dt + timedelta(seconds=delta)
        adjusted_filename = adjusted_name.strftime("%Y%m%d_%H%M%S") + base_name[15:] + ext
        if adjusted_filename in available_images:
            return adjusted_filename
    return None

# Create a new column for numpy file names
df['numpy_file'] = ""

# Process each image in the folder
for index, row in df.iterrows():
    image_filename = row['photo_id']  # e.g., '20250204_000126_foto.jpg'
    matched_image = find_closest_image(image_filename)
    
    if not matched_image:
        print(f"Warning: No matching image found for {image_filename}, skipping...")
        continue
    
    input_image_path = os.path.join(input_folder, matched_image)
    filename = os.path.splitext(matched_image)[0]  # '20250204_000126_foto'

    # Load image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load {input_image_path}, skipping...")
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # Define the scale factor 
    # Decrease the size by 3 times
    scale_factor = 1/2.0

    # Calculate the new image dimensions
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define circle parameters (adjust as needed)
    circle_x = width // 2  # Adjustable horizontal position
    circle_y = int(height * 0.38)  # Adjustable vertical position
    circle_radius = min(width, height) // 3 # Adjustable size

    # Draw a filled white circle (complete and adjustable)
    cv2.circle(mask, (circle_x, circle_y), circle_radius, 255, thickness=-1)

    # Apply mask to the image
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    print(masked_image.shape)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(masked_image, (5, 5), 0)  # (5,5) is the kernel size

    # Scaled image
    scaled_image = cv2.resize(src= blurred_image, 
                          dsize =(new_width, new_height), 
                          interpolation=cv2.INTER_AREA)
    
    print(scaled_image.shape)
    
    output_image_path = os.path.join(masked_folder, f'{filename}_mnsb.png')
    plt.imshow(scaled_image)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Load the masked image
    scaled_image = cv2.imread(output_image_path)
    if scaled_image is None:
        print(f"Error: Could not load {output_image_path}, skipping normalization...")
        continue
    
    # Normalize
    image_rgb = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(image_rgb)
    b_norm = cv2.normalize(b.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    g_norm = cv2.normalize(g.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    r_norm = cv2.normalize(r.astype('float32'), None, 0, 1, cv2.NORM_MINMAX)
    normalized_image = cv2.merge((b_norm, g_norm, r_norm))
    
    # Save numpy file
    output_npy_path = os.path.join(numpy_folder, f'{filename}_normalized.npy')
    np.save(output_npy_path, normalized_image)
    
    df.at[index, 'numpy_file'] = output_npy_path
    print(f"Processed {filename}: Masked/Scaled -> {output_image_path}, Normalized -> {output_npy_path}")

# Save updated CSV
df.to_csv(os.path.join(csv_output_folder, f'sesquicarbonato_processed.csv'), index=False)
print("Final DataFrame before saving:")
print(df.head())  # Display first rows
print(df.tail())  # Display last rows

print("Processing complete!")