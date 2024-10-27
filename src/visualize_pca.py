import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba  # Correct import for color conversion

# Paths to the CSV and label files
csv_path = '/Users/joshualunger/DeepEmotionBackup/data/derivative/PCA_local/sub-20_08.csv'
labels_path = '/Users/joshualunger/DeepEmotionBackup/data/resampled_annotations/av1o08'

# Load the PCA data
pca_data = pd.read_csv(csv_path, header=None)
pca_array = pca_data.to_numpy()

# Load the labels data (emotion column)
labels_data = pd.read_csv(labels_path, sep='\t')
labels = labels_data['emotion'].to_numpy()

# Ensure labels length matches the PCA data length
labels = labels[:pca_array.shape[0]]  # Truncate labels if they are longer than the PCA data

# Filter out rows where the emotion is '0' (no emotion)
valid_indices = labels != '0'
pca_array_filtered = pca_array[valid_indices]
labels_filtered = labels[valid_indices]

# Get unique emotions and assign custom colors to specific emotions
unique_emotions = np.unique(labels_filtered)
custom_colors = {
    'DISAPPOINTMENT': 'darkgrey',      # A muted color to reflect a negative emotion
    'SADNESS': 'blue',                 # Blue commonly associated with sadness
    'HAPPINESS': 'yellow',             # Bright yellow for happiness
    'HOPE': 'lightgreen',              # Green symbolizes growth and hope
    'ADMIRATION GRATITUDE': 'gold',    # Gold for admiration and gratitude, suggesting value
    'COMPASSION': 'lightpink',         # Pink, a soft and warm color for compassion
    'GLOATING': 'purple',              # Purple, often associated with superiority or ego
    'SATISFACTION': 'orange',          # Orange reflects a sense of fulfillment and energy
    'PRIDE': 'red'                     # Red, bold and strong, often linked with pride
}


# Get the vibrant colormap for other emotions
colors_map = colormaps.get_cmap('plasma')

# Create a color mapping that prioritizes custom colors and falls back to the colormap for other emotions
emotion_color_mapping = {
    emotion: to_rgba(custom_colors.get(emotion, colors_map(i / len(unique_emotions))))  # Convert to RGBA
    for i, emotion in enumerate(unique_emotions)
}

# Assign colors to the filtered PCA points based on emotion labels
point_colors = [emotion_color_mapping[emotion] for emotion in labels_filtered]

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Hide the axes
ax.set_axis_off()

# Scatter plot with colored points based on emotions
scatter = ax.scatter(pca_array_filtered[:, 0], pca_array_filtered[:, 1], pca_array_filtered[:, 2],
                     c=point_colors, edgecolor='k', s=50, alpha=0.6)

# Add titles and labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Create a legend based on unique emotions
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=emotion_color_mapping[emotion], 
                      markersize=10, label=emotion) for emotion in unique_emotions]
# ax.legend(handles=handles, title="Emotions", loc='upper left')

# Function to update the view angle for rotation
def update_view(angle):
    ax.view_init(elev=30, azim=angle)

# Create an animation
ani = FuncAnimation(fig, update_view, frames=np.arange(0, 360, 2), interval=100)

# Save the animation as an mp4 file
ani.save('/Users/joshualunger/DeepEmotionBackup/visuals/sub-20_run-08.gif', writer='ffmpeg')

