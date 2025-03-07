import pandas as pd

# CSV file path
csv_path = "/mnt/c/Users/aimc2/Documents/Granos/Casablanca_TestMeassures/Data_leia/camera/processing_n2/data/csvs_merged_mnsb_data/20241008_data_leia.csv"

# Load CSV
data = pd.read_csv(csv_path)

# Check for NaN values in all columns
nan_counts = data.isna().sum()
print("🔍 NaN values in each column:\n", nan_counts)

# Find rows where 'grain_type' is NaN
nan_rows = data[data['grain_type'].isna()]
if not nan_rows.empty:
    print("\n⚠️ Rows with NaN in 'grain_type':")
    print(nan_rows)

# Check for unexpected types in 'grain_type'
invalid_rows = data[~data['grain_type'].apply(lambda x: isinstance(x, str))]
if not invalid_rows.empty:
    print("\n🚨 Rows where 'grain_type' is not a string:")
    print(invalid_rows)

print("\n✅ Check completed. If any issues were found, they are listed above.")
