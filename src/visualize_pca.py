import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from collections import Counter
import hydra
from omegaconf import DictConfig

# Global variables
session = 8
subject = 1
csv_path = f'/Users/joshualunger/DeepEmotionBackup/data/derivative/PCA_local/sub-0{subject}_0{session}.csv'
labels_path = '/Users/joshualunger/DeepEmotionBackup/data/resampled_annotations/av1o6_resampled.tsv'

@hydra.main(config_path="./configs", config_name="base", version_base="1.2")
def main(cfg: DictConfig) -> None:
    def visualize_pca(cfg: DictConfig):
        global csv_path, labels_path, session  # Use global paths and session

        # Load the PCA data
        pca_data = pd.read_csv(csv_path, header=None)
        pca_array = pca_data.to_numpy()
        print(f"pca length {pca_array.shape[0]} (including NONE)")

        # Load the labels data
        labels_data = pd.read_csv(labels_path, sep='\t')

        # Session-specific time offsets from the config
        session_offsets = cfg.data.session_offsets

        # Ensure the session offset is within range
        if session > len(session_offsets):
            raise ValueError(f"Session {session} is out of range for provided session offsets")

        # Apply the session-specific offset
        session_offset = session_offsets[session-1]

        # Filter labels to include only those within the current session's PCA data range
        labels_data = labels_data[(labels_data['offset'] >= session_offset) & 
                                  (labels_data['offset'] < session_offset + 2 * pca_array.shape[0])]

        # Ensure labels length matches the PCA data length (truncate if necessary)
        labels = labels_data['emotion'].to_numpy()

        # Filter out rows where the emotion is 'NONE'
        valid_indices = labels != 'NONE'
        pca_array_filtered = pca_array[valid_indices]
        labels_filtered = labels[valid_indices]

        print(f"Filtered labels length {pca_array_filtered.shape[0]})")
        emotion_counts = Counter(labels_filtered)
        print(dict(emotion_counts))

        # Custom colors for specific emotions from the config (if available)
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
            emotion: to_rgba(custom_colors.get(emotion, colors_map(i / len(unique_emotions))))
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
        ani.save(f'/Users/joshualunger/DeepEmotionBackup/visuals/sub-0{subject}_run-0{session}.gif', writer='ffmpeg')

    visualize_pca(cfg)

if __name__ == "__main__":
    main()
