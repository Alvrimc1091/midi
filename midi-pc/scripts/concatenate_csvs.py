import pandas as pd
import glob
import os

def concatenate_csvs(input_folder, output_file):
    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the folder.")
        return
    
    # Read and concatenate all CSVs
    dataframes = [pd.read_csv(file) for file in csv_files]
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the concatenated dataframe to a new CSV
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

# Example usage
input_folder = "/home/alvaro/Documents/Workspace/midi/data/static_training/jpc"  # Change this to your folder path
output_file = "/home/alvaro/Documents/Workspace/midi/data/static_training/jpc_data.csv"  # Change this to your desired output file
concatenate_csvs(input_folder, output_file)
