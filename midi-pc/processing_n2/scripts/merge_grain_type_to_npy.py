import pandas as pd
import os

# Define file paths
processed_csv_path = '/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/camera/processing_n2/data/csvs_mnsb_data/20241008_processed.csv'
grain_type_csv_path = '/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/data_models/20241008_data_labeled_leia.csv'
output_csv_path = '/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/camera/processing_n2/data/csvs_merged_mnsb_data/20241008_data_leia.csv'

# Ensure output directory exists
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

# Load the processed CSV (contains numpy_file column)
df_processed = pd.read_csv(processed_csv_path)

# Load the grain type CSV (contains grain_type column)
df_grain_type = pd.read_csv(grain_type_csv_path)

# Ensure both DataFrames have the 'photo_id' column for merging
if 'photo_id' not in df_processed.columns or 'photo_id' not in df_grain_type.columns:
    print("Error: One of the CSV files is missing the 'photo_id' column!")
    exit()

# Merge dataframes on 'photo_id'
df_merged = pd.merge(df_processed, df_grain_type[['photo_id', 'grain_type']], on='photo_id', how='left')

# Save the merged CSV
df_merged.to_csv(output_csv_path, index=False)

print(f"Merging complete! Merged CSV saved to {output_csv_path}")
