
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt


# Load the CSV file
file_path = "/home/alvaro/Documents/Workspace/midi/data/static_training/midi_data.csv"
data = pd.read_csv(file_path)

# Strip leading/trailing spaces in the 'grain_type' column
data['grain_type'] = data['grain_type'].str.strip()

# Select the relevant columns for t-SNE
features = ['415nm', '445nm', '480nm', '515nm', '555nm', '590nm', '630nm', '680nm']

# Convert feature values to numeric safely
X = data[features].applymap(lambda x: pd.to_numeric(str(x).strip("[] *"), errors='coerce'))

# Handle missing values if necessary
X.fillna(X.mean(), inplace=True)

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Extract the 'grain' column for coloring
labels = data['grain_type']

# Perform t-SNE (ensure perplexity < dataset size / 3)
perplexity_value = min(30, len(X) // 3)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
X_embedded = tsne.fit_transform(X_scaled)

# Manually assign colors for 'harina' and 'poroto'
color_map = {'bicarbonato': 'red', 'sesquicarbonato': 'purple', 'jpc': 'green', 'fosfato': 'brown', 'lisina': 'pink', 'gluten': 'blue', 'conchuela': 'black'}

# Plot the t-SNE results with colors
plt.figure(figsize=(10, 8))
for grain in color_map.keys():
    indices = (labels == grain).to_numpy()  # Ensure correct boolean indexing
    plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
                label=grain, color=color_map[grain], alpha=0.7)

# Add legend and labels
plt.title("t-SNE Visualization of MIDI Data", fontsize=16)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(title="MIDI Products")
plt.grid(True)

# Save the figure
output_path = "/home/alvaro/Documents/Workspace/midi/models/all_midi/midi_data.png"
plt.savefig(output_path)
plt.show()
