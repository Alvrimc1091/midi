import pandas as pd
import glob

# Define the path pattern to match the .csv files
file_pattern = '/home/alvaro/Documents/Workspace/midi/data/static_training/*.csv'

# Use glob to find all files matching the pattern
file_list = glob.glob(file_pattern)

# Initialize a list to collect all data
data = []

# Read each file and collect the data
for file in file_list:
    # Read the CSV file with specified column names
    df = pd.read_csv(file, header=None, names=['date', '415nm', '445nm', '480nm', '515nm', '555nm', '590nm', '630nm', '680nm', 'clear', 'nir', 'photo_id', 'grain_type'])
    
    # Combine 'date' and 'time' columns into a single datetime column for sorting
#    df['datetime'] = pd.to_datetime(df['date'])
    
    # Drop the original 'date' column if not needed
#    df = df.drop(columns=['date'])
    
    # Append data to the list
    data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)

# Sort the combined DataFrame by the new 'datetime' column
sorted_df = combined_df.sort_values(by='date').reset_index(drop=True)

# Drop the 'datetime' column if you don't want it in the final CSV
#sorted_df = sorted_df.drop(columns=['date'])

# Save the sorted data to a new CSV file
sorted_df.to_csv('/home/alvaro/Documents/Workspace/midi/data/static_training/midi_data.csv', index=False)

print("Data has been successfully aggregated and sorted into 'data_test.csv'")
