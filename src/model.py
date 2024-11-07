import os
import keras
import mlflow
import numpy as np
from datetime import datetime
from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Reshape
from sklearn.cluster import KMeans

from logger import MLFlow_Logger


class MusicClust_Model:
    def __init__(self, logging=True):
        self.logging = logging
        self.main_model = None
        self.predict_model = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.input_dim = None

        if logging:
            self.logger = MLFlow_Logger(experiment=self.__class__.__name__)

    def create(self, input_dim: (int, int)):
        self.input_dim = input_dim

        encoder_input = Input(shape=input_dim + (3,))
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = MaxPooling2D((2, 2), padding="same")(x)

        latent_space = Flatten()(x)
        latent_space = Dense(64, activation="relu")(latent_space)

        x = Dense(np.prod(x.shape[1:]), activation="relu")(latent_space)
        x = Reshape((input_dim[0] // 8, input_dim[1] // 8, 128))(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        decoder_output = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

        self.main_model = Model(encoder_input, decoder_output)
        self.predict_model = Model(encoder_input, latent_space)
        self.main_model.summary()
        self.predict_model.summary()

    def load(self, path: str):
        self.predict_model = keras.models.load_model(path)
        self.predict_model.summary()

    def save(self, path: str):
        self.predict_model.save(os.path.join(path, self.timestamp))
        return self.timestamp

    def train(self, input_data: np.array, batch_size: int, n_epochs: int):
        self.logger.create_run(run=self.timestamp)
        self.logger.log_parameters({"CUSTOM input_dim": self.input_dim,
                                    "CUSTOM n_episodes": n_epochs,
                                    "CUSTOM batch_size": batch_size})

        self.main_model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        self.main_model.fit(input_data, input_data, epochs=n_epochs, batch_size=batch_size,
                            callbacks=[mlflow.tensorflow.MLflowCallback()])
        self.logger.end_run()

    def clusterize(self, input_data: np.array, n_clusters: int):
        latent_features = self.predict_model.predict(input_data)
        k_means = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = k_means.fit_predict(latent_features)
        return clusters
