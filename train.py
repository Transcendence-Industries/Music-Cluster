import os
import logging
import pickle
import numpy as np

from data import convert_mp3_to_spectrogram, convert_spectrogram_to_array
from model import MusicClust_Model

DATA_PATH = "./data"
SAMPLE_PATH = "./samples"
MODEL_PATH = "./models"


def prepare_samples():
    for root, dirs, files in os.walk(SAMPLE_PATH):
        logging.info(f"Preparing samples from directory '{root}'...")
        for file in files:
            if file.endswith(".mp3"):
                input_path = os.path.join(root, file)
                output_path = input_path.replace(SAMPLE_PATH, DATA_PATH, 1).replace(".mp3", ".png", 1)

                # Prepare samples
                if not os.path.exists(output_path):
                    output_dir = root.replace(SAMPLE_PATH, DATA_PATH, 1)

                    try:
                        os.makedirs(output_dir)
                    except:
                        pass

                    # Create spectrogram
                    convert_mp3_to_spectrogram(input_path, output_path)
                else:
                    logging.debug(f"Spectrogram for music file '{input_path}' already exists.")


def train_on_spectrograms(input_dim, batch_size, n_epochs):
    spectrograms = []
    file_paths = []

    # Loading spectrograms
    logging.info(f"Loading spectrograms from directory '{DATA_PATH}'...")

    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(root, file)

                # Converting spectrogram to image-array
                img_array = convert_spectrogram_to_array(file_path, input_dim)

                spectrograms.append(img_array)
                file_paths.append(file_path)
                logging.debug(f"Loaded spectrogram '{file_path}' and converted it to an image-array.")

    spectrograms = np.array(spectrograms)

    # Build and train model
    model = MusicClust_Model()
    model.create(input_dim=input_dim)
    model.train(input_data=spectrograms, batch_size=batch_size, n_epochs=n_epochs)
    timestamp = model.save(MODEL_PATH)

    # Add spectrograms to model
    with open(os.path.join(MODEL_PATH, timestamp, "spectrograms.pkl"), "wb") as f:
        pickle.dump(spectrograms, f)

    # Add file-paths to model
    with open(os.path.join(MODEL_PATH, timestamp, "file_paths.pkl"), "wb") as f:
        pickle.dump(file_paths, f)

    logging.info(f"Finished training for model '{timestamp}'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Prepare samples and train model
    prepare_samples()
    train_on_spectrograms(input_dim=(512, 512), batch_size=4, n_epochs=10)
