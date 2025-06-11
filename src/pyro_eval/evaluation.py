import json
import logging
from pathlib import Path
from typing import Dict

from pandas import Timedelta
from pyroengine.engine import Engine
from pyroengine.vision import Classifier

from .dataset import EvaluationDataset
from .engine_evaluation import EngineEvaluator
from .model import Model
from .model_evaluation import ModelEvaluator
from .path_manager import get_prediction_path
from .prediction_manager import PredictionManager
from .utils import make_dict_json_compatible, generate_run_id, get_class_default_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EvaluationPipeline:
    def __init__(
        self,
        dataset: Dict[str, EvaluationDataset],
        config: dict = {},
        run_id: str = "",
        device: str | None = None,
        use_existing_predictions: bool = True,
    ):

        self.model_dataset = dataset.get("model")
        self.engine_dataset = dataset.get("engine")
        self.config = self.get_config(config)
        self.model_path = self.config["model_path"]
        self.run_id = run_id or generate_run_id()
        self.metrics = {}

        # Load model
        self.model = Model(self.model_path, self.config, device)

        # Object used to store and manage predictions (load, update, save)
        self.prediction_manager = PredictionManager(
            model=self.model,
            prediction_file=get_prediction_path(self.model.hash),
            use_existing_predictions=use_existing_predictions
        )

        # Evaluate the model performance on single images
        if "model" in self.config["eval"]:
            self.model_evaluator = ModelEvaluator(
                dataset=self.model_dataset,
                model=self.model,
                prediction_manager=self.prediction_manager,
                config=self.config,
            )

        # Evaluate the engine performance on series of images
        if "engine" in self.config["eval"]:
            self.engine_evaluator = EngineEvaluator(
                dataset=self.engine_dataset,
                model=self.model,
                prediction_manager=self.prediction_manager,
                config=self.config,
                run_id=self.run_id,
                device=device,
            )

    def get_config(self, config):
        """
        Assign default parameters to config dict
        Get the default parameters from an Engine and Classifier instances
        """
        if "model_path" not in config:
            raise ValueError("A model_path must be provided in the evaluation config.")

        engine_default_values = get_class_default_params(Engine)
        model_default_values = get_class_default_params(Classifier)

        engine_config = config.get("engine", {})
        engine_config.setdefault("model_path", config.get("model_path"))
        engine_config.setdefault("nb_consecutive_frames", engine_default_values["nb_consecutive_frames"])
        engine_config.setdefault("conf_thresh", engine_default_values["conf_thresh"])
        engine_config.setdefault("max_bbox_size", model_default_values["max_bbox_size"])

        model_config = config.get("model", {})
        model_config.setdefault("model_path", config.get("model_path"))
        model_config.setdefault("iou", model_default_values["iou"])
        model_config.setdefault("conf", model_default_values["conf"])
        model_config.setdefault("imgsz", model_default_values["imgsz"])

        config.setdefault("eval", ["model", "engine"])
        config["engine"] = engine_config
        config["model"] = model_config

        return config

    def run(self):
        if "model" in self.config["eval"]:
            logging.info("Compute model metrics")
            self.metrics["model_metrics"] = self.model_evaluator.evaluate()
            self.display_metrics(subset=["model"])
        if "engine" in self.config["eval"]:
            logging.info("Compute engine metrics")
            self.metrics["engine_metrics"] = self.engine_evaluator.evaluate()

        self.display_metrics()

    def save_metrics(self, save_dir: Path):
        """
        Save results in a json file
        """
        filepath_result = save_dir / self.run_id / "metrics.json"
        filepath_result.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving metrics in {filepath_result}")

        dataset_info = {
            subset : {
                "ID": dataset.dataset_ID,
                "datapath": str(dataset.datapath),
                "Number of images": len(dataset),
                "Number of sequences": len(dataset.sequences),
                "hash" : dataset.hash,
            }
            for subset, dataset in zip(
                ["model", "engine"],
                [self.model_dataset, self.engine_dataset]
                )
        }

        self.config["model"]["hash"] = self.model.hash
        self.metrics.update(
            {"config": self.config, "run_id": self.run_id, "dataset": dataset_info}
        )

        metrics_dump = make_dict_json_compatible(self.metrics)

        with open(filepath_result, "w") as fp:
            json.dump(metrics_dump, fp)

    def display_metrics(self, subset=["model", "engine"]):
        def format_metric(value):
            if isinstance(value, float):
                return f"{value:.2f}"
            elif isinstance(value, int):
                return f"{value}"
            elif isinstance(value, Timedelta):
                return str(value)
            else:
                return "N/A"

        model_metrics = self.metrics.get("model_metrics", {})
        logging.info(f"Run ID: {self.run_id}")
        if "model" in subset:
            logging.info("Model Metrics:")
            logging.info(
                f"  Precision:        {format_metric(model_metrics.get('precision', 'N/A'))}"
            )
            logging.info(
                f"  Recall:           {format_metric(model_metrics.get('recall', 'N/A'))}"
            )
            logging.info(
                f"  F1 Score:         {format_metric(model_metrics.get('f1', 'N/A'))}"
            )
            logging.info(
                f"  False positives:  {format_metric(model_metrics.get('fp', 'N/A'))}"
            )
            logging.info(
                f"  True positives:   {format_metric(model_metrics.get('tp', 'N/A'))}"
            )
            logging.info(
                f"  False negatives:  {format_metric(model_metrics.get('fn', 'N/A'))}"
            )

        engine_image_metrics = self.metrics.get("engine_metrics", {}).get(
            "image_metrics", {}
        )
        engine_sequence_metrics = self.metrics.get("engine_metrics", {}).get(
            "sequence_metrics", {}
        )
        if "engine" in subset:
            logging.info("Engine Metrics:")
            logging.info("    Image Metrics:")
            logging.info(
                f"       Precision: {format_metric(engine_image_metrics.get('precision', 'N/A'))}"
            )
            logging.info(
                f"       Recall:    {format_metric(engine_image_metrics.get('recall', 'N/A'))}"
            )
            logging.info(
                f"       F1 Score:  {format_metric(engine_image_metrics.get('f1', 'N/A'))}"
            )
            logging.info("    Sequence Metrics:")
            logging.info(
                f"       Precision: {format_metric(engine_sequence_metrics.get('precision', 'N/A'))}"
            )
            logging.info(
                f"       Recall:    {format_metric(engine_sequence_metrics.get('recall', 'N/A'))}"
            )
            logging.info(
                f"       F1 Score:  {format_metric(engine_sequence_metrics.get('f1', 'N/A'))}"
            )
            logging.info(
                f"       Average Detection Delay:  {format_metric(engine_sequence_metrics.get('avg_detection_delay', 'N/A'))}"
            )
