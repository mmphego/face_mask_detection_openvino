#!/usr/bin/env python3

import mimetypes
import os
import time

from argparse import ArgumentParser

import cv2
import numpy as np

from loguru import logger
from responsive_voice.voices import UKEnglishMale
from tqdm import tqdm

from inference import Face_Detection, Mask_Detection


class FormatNotSupported(Exception):
    pass


class InputFeeder:
    def __init__(self, input_file=None):
        """
        This class can be used to feed input from an image, webcam, or video to your model.

        Parameters
        ----------
        input_file: str
            The file that contains the input image or video file.
            Leave empty for cam input_type.

        Example
        -------
        ```
            feed=InputFeeder(input_file='video.mp4')
            for batch in feed.next_frame():
                do_something(batch)
            feed.close()
        ```
        """
        self.input_file = input_file
        assert isinstance(self.input_file, str)
        self.check_file_exists(self.input_file)
        try:
            self._input_type, _ = mimetypes.guess_type(self.input_file)
            assert isinstance(self._input_type, str)
        except AssertionError:
            self._input_type = ""
        self._progress_bar = None
        self.load_data()

    def load_data(self):
        if "video" in self._input_type:
            self.cap = cv2.VideoCapture(self.input_file)
        elif "image" in self._input_type:
            self.cap = cv2.imread(self.input_file)
        elif "cam" in self.input_file.lower():
            self._input_type = self.input_file
            self.cap = cv2.VideoCapture(0)
        else:
            msg = f"Source: {self.input_file} not supported!"
            logger.warn(msg)
            raise FormatNotSupported(msg)
        logger.info(f"Loaded input source type: {self._input_type}")

    @staticmethod
    def check_file_exists(file):
        if "cam" in file:
            return

        if not os.path.exists(os.path.abspath(file)):
            raise FileNotFoundError(f"{file} does not exist.")

    @property
    def source_width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def source_height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def video_len(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def progress_bar(self):
        if not self._progress_bar:
            self._progress_bar = tqdm(total=int(self.video_len - self.fps + 1))
        return self._progress_bar

    def resize(self, frame, height=None, width=None):
        if (height and width) is None:
            width, height = (self.source_width // 2, self.source_height // 2)
        return cv2.resize(frame, (width, height))

    def show(self, frame, frame_name="video"):
        cv2.imshow(frame_name, frame)

    def write_video(self, output_path=".", filename="output_video.mp4"):
        out_video = cv2.VideoWriter(
            os.path.join(output_path, filename),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (self.source_width, self.source_height),
            True,
        )
        return out_video

    # TODO: Convert to contextmanager class
    def next_frame(self, quit_key="q"):
        """Returns the next image from either a video file or webcam."""
        while self.cap.isOpened():
            self.progress_bar.update(1)
            flag = False
            for _ in range(1):
                flag, frame = self.cap.read()

            if not flag:
                break
            yield frame

            key = cv2.waitKey(1) & 0xFF
            # if `quit_key` was pressed, break from the loop
            if key == ord(quit_key):
                break

    def close(self):
        """Closes the VideoCapture."""
        if "image" in self._input_type:
            self.cap.release()
        if self.progress_bar:
            self.progress_bar.close()
        cv2.destroyAllWindows()
        logger.info("============ CleanUp! ============")


def arg_parser():
    """Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--face-model",
        required=True,
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-m",
        "--mask-model",
        required=True,
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to image or video file or 'cam' for Webcam.",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "--face_prob_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for face detections filtering (Default: 0.8)",
    )
    parser.add_argument(
        "--mask_prob_threshold",
        type=float,
        default=0.3,
        help="Probability threshold for face mask detections filtering"
        "(Default: 0.3)",
    )
    parser.add_argument(
        "--enable-speech", action="store_true", help="Enable speech notification.",
    )
    parser.add_argument(
        "--tts",
        default="Please wear your MASK!!",
        help="Text-to-Speech, used for notification.",
    )
    parser.add_argument(
        "--ffmpeg", action="store_true", help="Flush video to FFMPEG.",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Show bounding box and stats on screen [debugging].",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )

    return parser.parse_args()


def main(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the video stream
    video_feed = InputFeeder(input_file=args.input)
    # Initialise the speech output
    if args.enable_speech:
        # TODO: Add args for selecting language, accent and male/female voice
        engine = UKEnglishMale()
        speak = engine.get_mp3(args.tts)

    # Initialise the class
    face_detection = Face_Detection(
        model_name=args.face_model,
        source_width=video_feed.source_width,
        source_height=video_feed.source_height,
        device=args.device,
        threshold=args.face_prob_threshold,
    )
    mask_detection = Mask_Detection(
        model_name=args.mask_model,
        device=args.device,
        threshold=args.mask_prob_threshold,
    )

    try:
        count = 0
        # TODO: Convert to contextmanager
        for frame in video_feed.next_frame():
            count += 1

            face_detect_infer_time, face_bboxes = face_detection.predict(
                frame, show_bbox=args.show_bbox
            )
            if face_bboxes:
                for face_bbox in face_bboxes:
                    # Useful resource: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

                    # Face bounding box coordinates cropped from the face detection inference
                    # are face_bboxes i.e `xmin, ymin, xmax, ymax`
                    # Therefore the face can be cropped by:
                    # frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

                    # extract the face ROI
                    (x, y, w, h) = face_bbox
                    face = frame[y:h, x:w]
                    (face_height, face_width) = face.shape[:2]
                    # Crop and show face
                    # video_feed.show(frame[y:h, x:w], "face")

                    # ensure the face width and height are sufficiently large
                    if face_height < 20 or face_width < 20:
                        continue

                    mask_detect_infer_time, mask_bboxes = mask_detection.predict(
                        face, show_bbox=args.show_bbox
                    )

                    if int(count) % 200 == 1 and args.enable_speech:
                        engine.play_mp3(mp3_file)

            if args.debug:
                text = f"Face Detection Inference time: {face_detect_infer_time:.3f} ms"
                face_detection.add_text(
                    text, frame, (15, video_feed.source_height - 80)
                )
                text = f"Face Mask Detection Inference time: {mask_detect_infer_time:.3f} ms"
                mask_detection.add_text(
                    text, frame, (15, video_feed.source_height - 60)
                )

                video_feed.show(video_feed.resize(frame))

    finally:
        video_feed.close()


if __name__ == "__main__":
    # Grab command line args
    args = arg_parser()
    # Perform inference on the input stream
    main(args)
