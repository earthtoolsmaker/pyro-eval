import os
import json
import logging
from typing import List, Dict
import numpy as np

from.data_structures import CustomImage
from .model import Model
from .utils import make_dict_json_compatible 

class PredictionManager:
    def __init__(
            self,
            model: Model,
            prediction_file: str,
            use_existing_predictions: bool = True
        ):
        self.model = model
        self.prediction_file = prediction_file
        self.use_existing_predictions = use_existing_predictions
        self.predictions = self.load_predictions() if self.use_existing_predictions else {}

    def load_predictions(self) -> Dict[str, np.ndarray]:
        """
        Loads predictions from a json file
        """
        loaded_predictions = {}
        try:
            with open(self.prediction_file, "r") as f:
                data = json.load(f)

            for image_name, prediction in data.items():
                loaded_predictions[image_name] = np.array(prediction)

            return loaded_predictions

        except Exception as e:
            logging.error(f"Failed to load predictions from {self.prediction_file}: {e}")
            return {}

    def save_predictions(self):
        # Load existing predictions to make sure we save everything
        loaded_predictions = self.load_predictions()
        new_predictions = self.predictions.copy()
        new_predictions.update(loaded_predictions)
        try:
            with open(self.prediction_file, "w") as f:
                json.dump(make_dict_json_compatible(new_predictions), f)
        except:
            logging.error(f"Unable to save predictions in {self.prediction_file}")

    def predict(self, images: List[CustomImage]):
        """
        Run predictions on an image list
        If use_existing_predictions is False, existing predictions are not loaded and everything is recomputed
        If use_existing_predictions is True, predictions are loaded from a json and only the missing ones are computed
        """

        for image in images:
            if image.name not in self.predictions:
                image.prediction = self.model.inference(image)
                self.predictions[image.name] = image.prediction
            else:
                image.prediction = self.predictions[image.name]