import json
import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from .dataset import EvaluationDataset
from .data_structures import CustomImage
from .model import Model
from .utils import compute_metrics, make_dict_json_compatible, find_matches


class ModelEvaluator:
    def __init__(self,
                 dataset: EvaluationDataset,
                 config: dict = {},
                 device: str = None,
                 use_previous_predictions: bool = True):

        self.dataset = dataset
        self.config = config
        self.use_previous_predictions = use_previous_predictions
        self.model_path = self.config.get("model_path", None)
        self.inference_params = self.config.get("inference_params", {})
        self.iou_threshold = self.config.get("iou", 0.1)

        # Load model
        self.model = Model(self.model_path, self.inference_params, device)

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

        # Track image prediction status for further analysis
        self.predictions = {
            "tp": [],
            "tn": [],
            "fp": [],
            "fn": [],
        }

        self.prediction_file = self.get_prediction_path()

    def get_prediction_path(self):
        abs_model_path = Path(self.model_path).resolve()
        output_dir = Path("data/predictions")

        # Try to take the relative path from the models folder, only work if the model path points to this folder
        try:
            relative = abs_model_path.relative_to(abs_model_path.parents[abs_model_path.parts.index("models")])
        except ValueError:
            # otherwise we use only the last subfolders
            relative = Path(*abs_model_path.parts[-2:])

        output_name = "_".join(relative.parts).replace(".pt", "") + ".json"

        return output_dir / output_name

    def run_predictions(self, image_list : List[CustomImage] = None):
        """
        Run predictions on a list of CustomImage objects
        By default runs on all images in the dataset
        Saves results in a json to avoid recomputation on different runs with the same model
        """
        # Run pred for each CustomImage in the EvaluationDataset
        image_list = image_list or self.images
        predictions = {}
        for image in image_list:
            image.prediction = self.model.inference(image)
            predictions[image.name] = image.prediction

        # Save predictions for later use
        if not os.path.isfile(self.prediction_file):
            with open(self.prediction_file, 'w') as fp:
                json.dump(make_dict_json_compatible(predictions), fp)

    def load_predictions(self):
        """
        Load prediction from a json file.
        Predictions are saved in a json named following the model path.
        """
        if not os.path.isfile(self.prediction_file):
            logging.error(f"Prediction file not found : {self.prediction_file}")
            logging.info("Running predictions.")
            self.run_predictions()
        else:
            # Load predictions from json file
            with open(self.prediction_file, 'r') as fp:
                predictions = json.load(fp)
            missing_predictions = []
            for image in self.images:
                if image.name in predictions:
                    image.prediction = np.array(predictions[image.name])
                else:
                    missing_predictions.append(image)

            # Run predictions on images which are missing in the json file
            if len(missing_predictions):
                self.run_predictions(image_list=missing_predictions)

    def track_predictions(self, fp, tp, fn, image_path):
        """
        Track and stroe predictions for each image
        """
        if fp > 0:
            self.predictions["fp"].append(image_path)
        elif tp > 0:
            self.predictions["tp"].append(image_path)
        if fn > 0:
            self.predictions["fn"].append(image_path)
        else:
            self.predictions["tn"].append(image_path)

    def evaluate(self):
        """
        Compares predictions and labels to evaluate the model performance on the dataset
        """
        if self.use_previous_predictions:
            self.load_predictions()
        else:
            self.run_predictions()

        nb_fp, nb_tp, nb_fn = 0, 0, 0

        for image in self.images:
            # Labels
            gt_boxes = image.boxes_xyxy
            # Predictions
            pred_boxes = image.prediction
            fp, tp, fn = find_matches(gt_boxes, pred_boxes, self.iou_threshold)
            self.track_predictions(fp, tp, fn, image.path)

            nb_fp += fp
            nb_tp += tp
            nb_fn += fn
        metrics = compute_metrics(
            false_positives=nb_fp, true_positives=nb_tp, false_negatives=nb_fn
        )

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fp": int(nb_fp),
            "tp": int(nb_tp),
            "fn": int(nb_fn),
            "predictions": self.predictions,
        }
