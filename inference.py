import math
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from openvino.inference_engine import IECore, IENetwork, get_version

__all__ = [
    "Face_Detection",
    "Mask_Detection",
]


COLOR = {"Green": (0, 255, 0), "Red": (0, 0, 255)}


def openvino_version_check():
    version = tuple(map(int, get_version().split(".")))[:2]
    if version != (2, 1):
        msg = (
            f"OpenVINO version: {version!r} not compatible with this library, "
            f"expected version: 2.1.xxx"
        )
        logger.warning(msg)
        raise RuntimeError(msg)


class InvalidModel(Exception):
    pass


class Base(ABC):
    """Model Base Class"""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
    ):
        self.model_weights = f"{model_name}.bin"
        self.model_structure = f"{model_name}.xml"
        assert (
            Path(self.model_weights).absolute().exists()
            and Path(self.model_structure).absolute().exists()
        )
        openvino_version_check()

        self.device = device
        self.threshold = threshold
        self._model_size = os.stat(self.model_weights).st_size / 1024.0 ** 2

        self._ie_core = IECore()
        self.model = self._get_model()

        # Get the input layer
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self._init_image_w = source_width
        self._init_image_h = source_height
        self.exec_network = None
        self.perf_stats = {}
        self.load_model()

    def _get_model(self):
        """Helper function for reading the network."""
        try:
            try:
                model = self._ie_core.read_network(
                    model=self.model_structure, weights=self.model_weights
                )
            except AttributeError:
                logger.warn("Using an old version of OpenVINO, consider updating it!")
                model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception:
            raise ValueError(
                "Could not Initialise the network. "
                "Have you entered the correct model path?"
            )
        else:
            return model

    def load_model(self):
        """Load the model into the plugin"""
        if self.exec_network is None:
            start_time = time.time()
            self.exec_network = self._ie_core.load_network(
                network=self.model, device_name=self.device
            )
            self._model_load_time = (time.time() - start_time) * 1000
            logger.info(
                f"Model: {self.model_structure} took {self._model_load_time:.3f} ms to load."
            )

    def predict(self, image, request_id=0, show_bbox=False, **kwargs):
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed correctly.")

        p_image = self.preprocess_input(image, **kwargs)
        predict_start_time = time.time()
        self.exec_network.start_async(
            request_id=request_id, inputs={self.input_name: p_image}
        )
        status = self.exec_network.requests[request_id].wait(-1)
        if status == 0:
            pred_result = []
            for output_name, data_ptr in self.model.outputs.items():
                pred_result.append(
                    self.exec_network.requests[request_id].outputs[output_name]
                )
            self.perf_stats[output_name] = self.exec_network.requests[
                request_id
            ].get_perf_counts()
            predict_end_time = float(time.time() - predict_start_time) * 1000
            bbox, _ = self.preprocess_output(
                pred_result, image, show_bbox=show_bbox, **kwargs
            )
            return (predict_end_time, bbox)

    @abstractmethod
    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """Draw bounding boxes onto the frame."""
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    @abstractmethod
    def draw_output(image):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def plot_frame(image):
        """Helper function for finding image coordinates/px"""
        img = image[:, :, 0]
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    def add_text(self, text, image, position, font_size=0.75, color=(255, 255, 255)):
        cv2.putText(
            image, text, position, cv2.FONT_HERSHEY_COMPLEX, font_size, color, 1,
        )

    def preprocess_input(self, image, height=None, width=None, **kwargs):
        """Helper function for processing frame"""
        if (height and width) is None:
            height, width = self.input_shape[2:]
        p_frame = cv2.resize(image, (width, height))
        # Change data layout from HWC to CHW
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


class Face_Detection(Base):
    """Class for the Face Detection Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
    ):
        super().__init__(
            model_name, source_width, source_height, device, threshold, extensions,
        )

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """Draw bounding boxes onto the Face Detection frame."""
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        if len(inference_results) == 1:
            inference_results = inference_results[0]

        coords = []
        for box in inference_results[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                coords.append((xmin, ymin, xmax, ymax))
                if show_bbox:
                    self.draw_output(
                        image, xmin, ymin, xmax, ymax, **kwargs
                    )
        return coords, image

    @staticmethod
    def draw_output(
        image,
        xmin,
        ymin,
        xmax,
        ymax,
        label="Person",
        padding_size=(0.05, 0.25),
        scale=2,
        thickness=2,
        **kwargs,
    ):
        _label = None
        if kwargs.get("mask_detected"):
            _label = (
                (f"{label} Wearing Mask", COLOR["Green"])
                if float(kwargs.get("mask_detected")) > kwargs.get("threshold", .1)
                else (f"{label} NOT wearing a Mask!!!", COLOR["Red"])
            )
            # print(_label)
        label = _label if _label is not None else (label, COLOR["Green"])

        cv2.rectangle(
            image, (xmin, ymin), (xmax, ymax,), color=label[1], thickness=thickness,
        )
        ((label_width, label_height), _) = cv2.getTextSize(
            label[0], cv2.FONT_HERSHEY_PLAIN, fontScale=scale, thickness=thickness,
        )

        cv2.putText(
            image,
            label[0],
            org=(image.shape[0] // 3, image.shape[1] // 3),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=scale,
            color=label[1],
            thickness=thickness,
        )


class Mask_Detection(Base):
    """Class for the Mask Detection Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
    ):
        super().__init__(
            model_name, source_width, source_height, device, threshold, extensions,
        )

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        flattened_predictions = np.vstack(inference_results).ravel()
        return flattened_predictions, image

    def draw_output(
        self, image, inference_results, **kwargs,
    ):
        pass
