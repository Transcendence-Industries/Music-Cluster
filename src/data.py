import logging
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def create_labeled_spectrogram(spectro_db, sr):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectro_db, sr=sr, x_axis="time", y_axis="mel", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")


def create_clean_spectrogram(spectro_db, sr):
    plt.figure(figsize=(6, 6))
    librosa.display.specshow(spectro_db, sr=sr, x_axis=None, y_axis=None, cmap="magma")
    plt.axis("off")


def convert_mp3_to_spectrogram(input_file, output_file):
    logging.debug(f"Loading music file '{input_file}'...")
    y, sr = librosa.load(input_file, sr=None)
    spectro = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectro_db = librosa.power_to_db(spectro, ref=np.max)

    # create_labeled_spectrogram(spectro_db, sr)
    # plt.show()

    create_clean_spectrogram(spectro_db, sr)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close()
    logging.debug(f"Saved spectrogram to '{output_file}'.")


def convert_spectrogram_to_array(file_path, shape):
    img = load_img(file_path, target_size=shape)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array


def display_spectrogram_clusters(cluster_dict):
    # Calculate total size of plot
    max_len = max(len(path_list) for path_list in cluster_dict.values())
    total_count = len(cluster_dict)
    fig, axes = plt.subplots(total_count, max_len, figsize=(max_len * 5, total_count * 5))

    # Handle case where only one cluster exists
    if len(cluster_dict) == 1:
        axes = [axes]
    elif cluster_dict == 1:
        axes = [[ax] for ax in axes]

    # Display all spectrograms on plot
    for row, (cluster, path_list) in enumerate(cluster_dict.items()):
        for col in range(max_len):
            ax = axes[row][col]

            if col < len(path_list):
                img = mpimg.imread(path_list[col])
                ax.imshow(img)
                ax.set_title(f"Cluster {cluster} - Image {col + 1}")
            else:
                ax.axis("off")

            ax.axis("off")

    plt.tight_layout()
    plt.show()
