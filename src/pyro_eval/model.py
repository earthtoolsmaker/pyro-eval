import logging
import os

import numpy as np
import onnxruntime
import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from pyroengine.vision import Classifier
from ultralytics import YOLO

from .data_structures import CustomImage


class Model:
    def __init__(self, model_path, inference_params, device=None):
        self.model_path = model_path

        self.model = self.load_model()
        self.device = self.get_device(device)
        self.model.to(self.device)
        self.format = None
        self.inference_params = self.set_inference_params(inference_params)

    def load_model(self):
        if not self.model_path:
            raise ValueError(
                f"No model provided for evaluation, path needs to be specified."
            )

        logging.info(f"Loading model : {self.model_path}")
        if os.path.isfile(self.model_path):
            # Local file, .onnx format
            if self.model_path.endswith(".onnx"):
                return self.load_onnx()

            # Local file, .pt format
            if self.model_path.endswith(".pt"):
                self.format = "pt"
                return YOLO(self.model_path)

        else:
            # File doesn't exist, check for a HuggingFace repo - TODO : decide how HF models path should be provided
            if "huggingface.co" in self.model_path:
                self.load_HF()

            # File doesn't not exist, but path is not a huggingface path
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def load_onnx(self):
        """
        Loads an onnx model
        Format has to be tracked as model call differs from other formats
        """
        try:
            # This object is created to use the pre-processing and post-processing from the engine
            # Parameters are set to remove any filter of the preds
            model = Classifier(
                model_path=self.model_path,
                format="onnx",
                conf=self.inference_params["conf"],
                max_bbox_size=1
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the ONNX model from {self.model_path}: {str(e)}"
            ) from e

        logging.info(f"ONNX model loaded successfully from {self.model_path}")
        self.format = "onnx"
        return model

    def load_HF(self):
        """
        Loads model from an HuggingFace repo
        """
        token = os.getenv("HF_TOKEN") or HfFolder.get_token()
        repo_id = self.model_path.split("https://huggingface.co/")[-1]
        filename = f"{os.path.basename(repo_id)}.pt"
        if token is None:
            raise ValueError(
                "Error : no Hugging Face token found. Please authenticate with `huggingface-cli login`."
            )
        try:
            hf_hub_download(repo_id=repo_id, filename=filename)
        except HfHubHTTPError as e:
            raise ValueError(f"Access denied to  ({repo_id}): {e}")

        # Check model existence on HuggingFace
        api = HfApi()
        # Remove the first part of the url

        model_info = api.model_info(repo_id, token=token)
        if not model_info:
            raise ValueError(
                f"Error : {self.model_path} doesn't exist or is not accessible."
            )

        self.format = "hf"
        # All checks are correct, return the model
        return YOLO(self.model_path)

    def get_device(self, device):
        """
        Returns proper devide depending on configuration
        """
        if device is not None:
            return torch.device(device)
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def set_inference_params(self, inference_params):
        return {
            "conf": inference_params.get("conf", 0.05),
            "iou": inference_params.get("iou", 0),
            "imgsz": inference_params.get("imgsz", 1024),
        }

    def inference(self, image: CustomImage):
        """
        Reads an image and run the model on it.
        """
        pil_image = image.load()

        if self.format == "onnx":
            try:
                # Returns an array of predicitions with boxes xyxyn and confidence
                prediction = self.model(pil_image) # [[x1, y1, x2, y2, confidence]]
            except Exception as e:
                logging.error(f"Onnx inference failed on {image.path} : {e}")
                prediction = []
        else:
            try:
                results = self.model.predict(
                    source=pil_image,
                    conf=self.inference_params["conf"],
                    iou=self.inference_params["iou"],
                    imgsz=self.inference_params["imgsz"],
                    device=self.device,
                )[0]
                # Format predictions to onnx format : [[boxes.xyxyn, conf]]
                prediction = []
                for box in results.boxes:
                    xyxyn = box.xyxyn.cpu().numpy().flatten()  # [x1, y1, x2, y2]
                    conf = box.conf.cpu().item()
                    prediction.append([*xyxyn, conf])

            except Exception as e:
                logging.error(f"Inference failed on {image.path} : {e}")
                prediction = []

        return prediction
