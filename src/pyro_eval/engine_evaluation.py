import json
import logging
import os
from collections import deque

import numpy as np
import pandas as pd
from pyroengine.engine import Engine
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from .data_structures import Sequence
from .dataset import EvaluationDataset
from .model import Model
from .path_manager import get_prediction_csv
from .prediction_manager import PredictionManager
from .utils import compute_metrics, export_model, generate_run_id, make_dict_json_compatible, timing

logging.getLogger("pyroengine.engine").setLevel(logging.WARNING)


class EngineEvaluator:
    # TODO : as EngineEvaluator and ModelEvaluator share some attributes and methods the should inherits from an EvaluatorClass
    # that manages instanciation, saving, run_id etc.
    def __init__(
        self,
        dataset: EvaluationDataset,
        model: Model,
        prediction_manager: PredictionManager,
        config: dict = {},
        save: bool = False,
        run_id: str = None,
        device: str = None,
    ):

        self.dataset = dataset
        self.model = model
        self.prediction_manager = prediction_manager
        self.config = config["engine"]
        self.save = save  # If save is True we regularly dump results and the config used
        self.run_id = run_id if run_id else generate_run_id()
        self.results_data = [
            "sequence_id",
            "image",
            "sequence_label",
            "ground_truth_boxes",
            "image_label",
            "prediction",
            "confidence",
            "timedelta",
        ]
        self.model_path = self.config.get("model_path", None)
        self.needs_deletion = False
        self.run_model_path = None
        self.engine = self.instanciate_engine()
        self.device = device

        # Retrieve images from the dataset
        self.images = self.dataset.get_all_images()

    @timing("Engine evaluation")
    def evaluate(self):
        # Run Engine predictions on each sequence of the dataset
        self.run_engine_dataset()

        # Compute metrics from predictions
        self.metrics = {
            "run_id": self.run_id,
            "image_metrics": self.compute_image_level_metrics(),
            "sequence_metrics": self.compute_sequence_level_metrics(),
        }

        # Save metrics in a json file
        if self.save:
            with open(os.path.join(self.result_dir, "engine_metrics.json"), "w") as fip:
                json.dump(make_dict_json_compatible(self.metrics), fip)

        return self.metrics

    def instanciate_engine(self):
        """
        Creates a pyro Engine instance
        """
        # We need to convert .pt local paths to .onnx as that's the only format supported by the engine
        if self.model_path is not None:
            if self.model_path.endswith(".onnx"):
                self.run_model_path = self.model_path
            elif self.model_path.endswith(".pt"):
                logging.info(f"Exporting model file from pt to onnx format.")
                self.run_model_path = export_model(self.model_path)
                self.needs_deletion = True  # We remove the local .onnx file created
            else:
                raise RuntimeError(
                    f"Model format not supported by the Engine : {self.model_path}"
                )

        engine = Engine(
            nb_consecutive_frames=self.config["nb_consecutive_frames"],
            conf_thresh=self.config["conf_thresh"],
            max_bbox_size=self.config["max_bbox_size"],
            model_path=self.run_model_path,
        )

        return engine

    def run_engine_sequence(self, sequence: Sequence):
        """
        Instanciate an Engine and run predictions on a Sequence containing a list of images.
        Returns a dataframe containing image info and the confidence predicted
        """

        # Initialize a new Engine for each sequence
        # TODO : better handle default values

        sequence_results = pd.DataFrame(columns=self.results_data)
        missing_predictions = []

        for image in sequence.images:
            # Run prediction on a single image
            if image.prediction is not None:
                # Use the previously computed prediction stored in the prediciton json file
                confidence = self.engine.predict(frame=None, fake_pred=np.array(image.prediction[:4]))
            else:
                pil_image = image.load()
                missing_predictions.append(image)
                confidence = self.engine.predict(pil_image)

            sequence_results.loc[len(sequence_results)] = [
                sequence.sequence_id,  # sequence_id
                image.path,  # image
                sequence.label,  # sequence_label
                image.boxes,  # ground_truth_boxes
                image.label,  # image_label
                bool(confidence > self.engine.conf_thresh),  # prediction (True/False)
                confidence,  # confidence
                image.timedelta,  # timedelta
            ]

        # We run and save the predicitons on the missing images to be able to later load them 
        self.prediction_manager.predict(images=missing_predictions)

        # Clear states to reset the engine for the next sequence
        self.engine._states = {
            "-1": {
                "last_predictions": deque(maxlen=self.config["nb_consecutive_frames"]),
                "ongoing": False,
                "last_image_sent": None,
                "last_bbox_mask_fetch": None,
            },
        }
        return sequence_results

    def run_engine_dataset(self):
        """
        Function that processes predictions through the Engine on sequences of images
        """

        self.predictions_df = pd.DataFrame(columns=self.results_data)

        try:
            for sequence in self.dataset:
                sequence_results = self.run_engine_sequence(sequence)

                # Add sequence results to result dataframe
                self.predictions_df = pd.concat([self.predictions_df, sequence_results])
                # Checkpoint to save predictions regularly
                self.prediction_manager.save_predictions()

        finally:
            if self.needs_deletion:
                try:
                    os.remove(self.run_model_path)
                except:
                    logging.error(
                        f"Temporary model file could not be removed : {self.run_model_path}"
                    )

        # Final saving of the predictions
        self.prediction_manager.save_predictions()

        if self.save:
            pred_csv = get_prediction_csv(self.run_id)
            logging.info(f"Saving predictions in {pred_csv}")
            self.predictions_df.to_csv(pred_csv, index=False)

    def compute_image_level_metrics(self):
        """
        Computes image-based metrics on the predicion dataframes.
        Those metrics do not take sequences into account.
        """

        y_true = self.predictions_df["image_label"].astype(bool)
        y_pred = self.predictions_df["prediction"].astype(bool)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

        metrics = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        logging.info("Image-level metrics")
        logging.info(
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
        )
        logging.info(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }

    def compute_sequence_level_metrics(self):
        """
        Computes sequence-based metrics from the prediction dataframe
        """
        sequence_metrics = []

        for sequence_id, group in self.predictions_df.groupby("sequence_id"):
            sequence_label = group["sequence_label"].iloc[0]
            has_detection = group["prediction"].any()
            detection_timedeltas = group[group["prediction"]]["timedelta"]
            detection_timedeltas = pd.to_timedelta(
                detection_timedeltas, errors="coerce"
            )
            detection_delay = (
                detection_timedeltas.min() if not detection_timedeltas.empty else None
            )

            if not detection_timedeltas.empty:
                detection_delay = np.min(detection_timedeltas)
            else:
                detection_delay = None

            sequence_metrics.append(
                {
                    "sequence_id": sequence_id,
                    "label": sequence_label,
                    "has_detection": has_detection,
                    "detection_delay": detection_delay,
                }
            )

        sequence_df = pd.DataFrame(sequence_metrics)

        tp_sequences = sequence_df[
            (sequence_df["label"] == True) & (sequence_df["has_detection"] == True)
        ]
        fn_sequences = sequence_df[
            (sequence_df["label"] == True) & (sequence_df["has_detection"] == False)
        ]
        fp_sequences = sequence_df[
            (sequence_df["label"] == False) & (sequence_df["has_detection"] == True)
        ]
        tn_sequences = sequence_df[
            (sequence_df["label"] == False) & (sequence_df["has_detection"] == False)
        ]
        metrics = compute_metrics(
            false_positives=len(fp_sequences),
            true_positives=len(tp_sequences),
            false_negatives=len(fn_sequences),
        )

        predictions = {
            "tp": tp_sequences["sequence_id"].to_list(),
            "fn": fn_sequences["sequence_id"].to_list(),
            "fp": fp_sequences["sequence_id"].to_list(),
            "tn": tn_sequences["sequence_id"].to_list(),
        }
        logging.info("Sequence-level metrics")
        logging.info(
            f"Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}"
        )
        logging.info(
            f"TP: {len(tp_sequences)}, FP: {len(fp_sequences)}, FN: {len(fn_sequences)}, TN: {len(tn_sequences)}"
        )

        if not tp_sequences["detection_delay"].isnull().all():
            avg_detection_delay = tp_sequences["detection_delay"].dropna().mean()
            logging.info(
                f"Avg. delay before detection (TP sequences): {avg_detection_delay}"
            )
        else:
            logging.info("No detection delay info available for TP sequences.")

        return {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": len(tp_sequences),
            "fp": len(fp_sequences),
            "fn": len(fn_sequences),
            "tn": len(tn_sequences),
            "avg_detection_delay": (
                avg_detection_delay
                if not tp_sequences["detection_delay"].isnull().all()
                else None
            ),
            "predictions": predictions,
        }

