import pandas as pd
import os

csv_file = "/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/camera/processing_n2/data/csvs_merged_mnsb_data/20241008_data_leia.csv"  # Change to the correct file path
filtered_csv_file = "/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/camera/processing_n2/data/csvs_merged_mnsb_data/20241008_data_leia_filtered_grain_parcial.csv"  # Change to the correct file path

# Define the filter value for grain_type
grain_type_filter = "parcial"  # Change this to the value you want to filter out

def delete_files_based_on_filter(csv_file, grain_type_filter):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Strip spaces and ensure proper column names
    df.columns = df.columns.str.strip()
    
    # Replace NaN or empty values in 'grain_type' with an empty string
    df["grain_type"].fillna("", inplace=True)
    
    # Filter the rows that match the grain_type_filter or are empty
    filtered_df = df[(df["grain_type"] != grain_type_filter) & (df["grain_type"].str.strip() != "")]
    
    # Get the file paths to delete
    files_to_delete = df[(df["grain_type"] == grain_type_filter) | (df["grain_type"].str.strip() == "")]["numpy_file"].tolist()
    
    # Delete files
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    
    # Save the filtered data back to the CSV file
    filtered_df.to_csv(filtered_csv_file, index=False)
    print(f"Filtered CSV saved as {filtered_csv_file}")

# Execute the function
delete_files_based_on_filter(csv_file, grain_type_filter)
