import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import re

# Define the file paths to all the CSVs you want to combine
file_paths = [
    "/home/pi/demo/data/fosfato_data.csv",
    "/home/pi/demo/data/lisina_data.csv",
    "/home/pi/demo/data/sesquicarbonato_data.csv",
    "/home/pi/demo/data/conchuela_data.csv",
    "/home/pi/demo/data/jpc_data.csv",
    "/home/pi/demo/data/gluten_data.csv"
]

# Initialize an empty list to store all the DataFrames and the corresponding days
all_data = []
days = []

# Function to extract the number inside the brackets
def extract_value_from_brackets(value):
    match = re.search(r'\[([0-9]+)\]', str(value))
    if match:
        return int(match.group(1))
    return np.nan  # If no number is found, return NaN

# Extract day from each filename and store data
for file_path in file_paths:
    day = file_path.split("/")[-1].split("_")[0]  # Extract the day (e.g., "20241007")
    days.append(day)  # Add day label to list
    
    # Read the data
    data = pd.read_csv(file_path, header=None)  # No header in this case

    # Assuming the features are in specific columns and the last column is the file name
    features = data.iloc[:, 1:9]  # Columns 1 to 8 (values in brackets)
    
    # Apply the function to extract the numeric value from each element in the features
    X = features.apply(lambda x: x.map(extract_value_from_brackets))

    # Append the processed data to the all_data list
    all_data.append(X)

# Combine all data and repeat day labels
combined_data = pd.concat(all_data, ignore_index=True)
day_labels = np.repeat(days, [len(data) for data in all_data])

# Handle missing values by filling them with zeros (you can choose other strategies)
combined_data_filled = combined_data.fillna(0)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_data_filled)

# Check the number of features
n_features = X_scaled.shape[1]
n_components = min(n_features, 50)  # Set n_components to the smaller of n_features and 50


pca = PCA(n_components=50)  # Reduce to 50 components before applying t-SNE
X_pca = pca.fit_transform(X_scaled)

# Now apply t-SNE to the PCA-reduced data
tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=500)
X_embedded = tsne.fit_transform(X_scaled)

# Generate a colormap for the days
unique_days = sorted(set(day_labels))
colors = cm.jet(np.linspace(0, 1, len(unique_days)))  # Use colormap for unique days
day_to_color = {day: color for day, color in zip(unique_days, colors)}
point_colors = [day_to_color[day] for day in day_labels]

# Plot the t-SNE results with coloring
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=point_colors, alpha=0.7)
plt.title("t-SNE Visualization of Combined Grain Data by Day", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)

# Add a legend for the days
handles = [plt.Line2D([], [], marker='o', color=color, linestyle='', markersize=8) for color in colors]
plt.legend(handles, unique_days, title="Days", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)

# Save the plot
plt.tight_layout()
plt.savefig("/home/pi/demo/results/combined_data_colored.png")
plt.show()
