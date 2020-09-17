import cv2
import numpy as np

from pyvino_utils.models.openvino_base.base_model import Base

__all__ = [
    "FaceDetection",
    "MaskDetection",
]


COLOR = {"Green": (0, 255, 0), "Red": (0, 0, 255)}


class FaceDetection(Base):
    """Class for the Face Detection Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
        **kwargs,
    ):
        super().__init__(
            model_name,
            source_width,
            source_height,
            device,
            threshold,
            extensions,
            **kwargs,
        )

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        """Draw bounding boxes onto the Face Detection frame."""
        results = {}
        if not (self._init_image_w and self._init_image_h):
            raise RuntimeError("Initial image width and height cannot be None.")
        if len(inference_results) == 1:
            inference_results = inference_results[0]

        bbox_coord = []
        for box in inference_results[0][0]:  # Output shape is 1x1xNx7
            conf = box[2]
            if conf >= self.threshold:
                xmin = int(box[3] * self._init_image_w)
                ymin = int(box[4] * self._init_image_h)
                xmax = int(box[5] * self._init_image_w)
                ymax = int(box[6] * self._init_image_h)
                bbox_coord.append((xmin, ymin, xmax, ymax))
                if show_bbox:
                    self.draw_output(image, xmin, ymin, xmax, ymax, **kwargs)

        results["image"] = image
        results["bbox_coord"] = bbox_coord
        return results

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
                if float(kwargs.get("mask_detected")) > kwargs.get("threshold", 0.1)
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


class MaskDetection(Base):
    """Class for the Mask Detection Model."""

    def __init__(
        self,
        model_name,
        source_width=None,
        source_height=None,
        device="CPU",
        threshold=0.60,
        extensions=None,
        **kwargs,
    ):
        super().__init__(
            model_name,
            source_width,
            source_height,
            device,
            threshold,
            extensions,
            **kwargs,
        )

    def preprocess_output(self, inference_results, image, show_bbox=False, **kwargs):
        results = {}
        results["flattened_predictions"] = np.vstack(inference_results).ravel()
        results["image"] = image
        return results

    def draw_output(
        self, image, inference_results, **kwargs,
    ):
        pass
