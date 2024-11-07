import os
import logging
import pickle

from data import display_spectrogram_clusters
from model import MusicClust_Model

MODEL_PATH = "./models"


def create_clusters_for_model(model_name, n_clusters):
    model_path = os.path.join(MODEL_PATH, model_name)

    if os.path.isdir(model_path):
        logging.info(f"Loading model '{model_name}'...")

        # Load spectrograms from model
        with open(os.path.join(model_path, "spectrograms.pkl"), "rb") as f:
            spectrograms = pickle.load(f)

        # Load file-paths from model
        with open(os.path.join(model_path, "file_paths.pkl"), "rb") as f:
            file_paths = pickle.load(f)

        model = MusicClust_Model(logging=False)
        model.load(model_path)

        # Create clusters with spectrograms and file-paths
        clusters = model.clusterize(spectrograms, n_clusters=n_clusters)
        logging.info(f"Finished creating {n_clusters} clusters.")

        # Convert clusters to dict
        result_dict = {}
        for file_path, cluster in zip(file_paths, clusters):
            if cluster not in result_dict.keys():
                result_dict[cluster] = []

            result_dict[cluster].append(file_path)

        return result_dict
    else:
        logging.error(f"Can't find model '{model_name}'!")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create clusters
    clusts = create_clusters_for_model(model_name="2024-10-24_01-47-21", n_clusters=5)

    # Fancy printing of clusters
    for i, c in clusts.items():
        print(f"{i}: {c}")

    # Fancy plotting of clusters
    display_spectrogram_clusters(clusts)
