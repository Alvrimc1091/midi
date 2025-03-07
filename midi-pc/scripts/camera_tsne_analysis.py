import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt


# Load CSV file
csv_path = "/home/alvaro/Documents/Workspace/midi/data/static_training/camera/data/csvs_mnsb_data/midi_data_processed.csv"
data = pd.read_csv(csv_path)

# Extract paths to .npy files and grain types
npy_files = data['numpy_file']
grain_types = data['grain_type'].str.strip()

# Function to process .npy files
def process_npy(file_path):
    try:
        arr = np.load(file_path)  # Load .npy file
        arr = arr.reshape(-1, 3)  # Flatten while keeping color channels
        arr = arr[np.any(arr != 0, axis=1)]  # Remove zero pixels
        return arr.mean(axis=0)  # Aggregate by mean of each color channel
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all .npy files
feature_vectors = [process_npy(f) for f in npy_files]
feature_vectors = np.array([fv for fv in feature_vectors if fv is not None])

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_vectors)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)

# Assign colors to grain types
unique_grains = grain_types.unique()
colors = plt.cm.get_cmap("tab10", len(unique_grains))
color_map = {grain: colors(i) for i, grain in enumerate(unique_grains)}

# Plot results
plt.figure(figsize=(10, 8))
for grain in unique_grains:
    indices = grain_types == grain
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
                label=grain, color=color_map[grain], alpha=0.7)

plt.title("t-SNE Clustering of MIDI Data", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(title="MIDI Type")
plt.grid(True)

# Save plot
output_path = '/home/alvaro/Documents/Workspace/midi/data/static_training/camera/camera_tsne_clustering_midi.png'
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")
