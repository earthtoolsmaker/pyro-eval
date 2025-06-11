import numpy as np

from .dataset import EvaluationDataset
from .model import Model
from .path_manager import get_prediction_path
from .prediction_manager import PredictionManager
from .utils import compute_metrics, find_matches, make_dict_json_compatible, timing


class ModelEvaluator:
    def __init__(
        self,
        dataset: EvaluationDataset,
        model: Model,
        prediction_manager: PredictionManager,
        config: dict = {},
    ):

        self.dataset = dataset
        self.model = model
        self.prediction_manager = prediction_manager
        self.config = config["model"]
        self.iou_threshold = self.config["iou"]

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

        # Track image prediction status for further analysis
        self.predictions = {
            "tp": [],
            "tn": [],
            "fp": [],
            "fn": [],
        }

        self.prediction_file = get_prediction_path(self.config["model_path"])

    def track_predictions(
            self,
            image_path: str,
            fp: int,
            tp: int,
            fn: int
        ) -> None:
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

    @timing("Model evaluation")
    def evaluate(self):
        """
        Compares predictions and labels to evaluate the model performance on the dataset
        """
        self.prediction_manager.predict(self.images)
        self.prediction_manager.save_predictions()

        nb_fp, nb_tp, nb_fn = 0, 0, 0

        for image in self.images:
            # Labels
            gt_boxes = image.boxes_xyxy
            # Predictions - last element of the array is the confidence
            pred_boxes = np.array([pred[:-1] for pred in image.prediction])
            img_fp, img_tp, img_fn = find_matches(gt_boxes, pred_boxes, self.iou_threshold)
            self.track_predictions(image.path, img_fp, img_tp, img_fn)

            nb_fp += img_fp
            nb_tp += img_tp
            nb_fn += img_fn
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
            "tn": len(self.predictions["tn"]),
            "predictions": self.predictions,
        }
