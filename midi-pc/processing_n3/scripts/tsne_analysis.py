import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt

# Load CSV file
csv_path = "/home/alvaro/Documents/Workspace/midi/processing_n3/data/csvs_mnsb_data/midi_data.csv"
data = pd.read_csv(csv_path)

# Extract RGB mean values and grain types
grain_types = data['grain_type'].str.strip().values  # Convert to NumPy array

r_means = data['r_mean'].values
g_means = data['g_mean'].values
b_means = data['b_mean'].values

r_var = data['r_var'].values
g_var = data['g_var'].values
b_var = data['b_var'].values

# Combine RGB means into a feature matrix
feature_vectors = np.column_stack((r_means, g_means, b_means, r_var, g_var, b_var))

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_vectors)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X_scaled)

# Assign colors to grain types
unique_grains = np.unique(grain_types)  # Ensure unique values
colors = plt.cm.get_cmap("tab10", len(unique_grains))
color_map = {grain: colors(i) for i, grain in enumerate(unique_grains)}

# Plot results
plt.figure(figsize=(10, 8))
for grain in unique_grains:
    indices = grain_types == grain  # Ensure boolean mask works
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
                label=grain, color=color_map[grain], alpha=0.7)

plt.title("t-SNE Clustering of MIDI Data (RGB Means)", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(title="MIDI Type")
plt.grid(True)

# Save plot
output_path = "/home/alvaro/Documents/Workspace/midi/processing_n3/data/tsne_clustering_midi_rgb.png"
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")
