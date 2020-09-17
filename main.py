#!/usr/bin/env python3

from argparse import ArgumentParser

from responsive_voice.voices import UKEnglishMale

from inference import FaceDetection, MaskDetection
from pyvino_utils import InputFeeder


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
        help="Probability threshold for face mask detections filtering" "(Default: 0.3)",
    )
    parser.add_argument(
        "--enable-speech", action="store_true", help="Enable speech notification.",
    )
    parser.add_argument(
        "--tts",
        type=str,
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
    input_feed = InputFeeder(input_feed=args.input)
    # Initialise the speech output
    if args.enable_speech:
        # TODO: Add args for selecting language, accent and male/female voice
        engine = UKEnglishMale()
        speak = engine.get_mp3(args.tts)
    # Initialise the class
    face_detection = FaceDetection(
        model_name=args.face_model,
        device=args.device,
        threshold=args.face_prob_threshold,
        input_feed=input_feed,
    )
    mask_detection = MaskDetection(
        model_name=args.mask_model,
        device=args.device,
        threshold=args.mask_prob_threshold,
    )

    count = 0
    face_detect_infer_time = 0
    mask_detect_infer_time = 0
    mask_detected_prob = -1
    try:
        # TODO: Convert to contextmanager
        for frame in input_feed.next_frame():
            count += 1

            fd_results = face_detection.predict(
                frame, show_bbox=args.show_bbox, mask_detected=mask_detected_prob
            )
            face_bboxes = fd_results["process_output"]["bbox_coord"]
            if face_bboxes:
                for face_bbox in face_bboxes:
                    # Useful resource:
                    # https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

                    # Face bounding box coordinates cropped from the face detection
                    # inference are face_bboxes i.e `xmin, ymin, xmax, ymax`
                    # Therefore the face can be cropped by:
                    # frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

                    # extract the face ROI
                    (x, y, w, h) = face_bbox
                    face = frame[y:h, x:w]
                    (face_height, face_width) = face.shape[:2]
                    # Crop and show face
                    # input_feed.show(frame[y:h, x:w], "face")

                    # ensure the face width and height are sufficiently large
                    if face_height < 20 or face_width < 20:
                        continue

                    md_results = mask_detection.predict(
                        face, show_bbox=args.show_bbox, frame=frame
                    )
                    mask_detected_prob = md_results["process_output"][
                        "flattened_predictions"
                    ]
                    if (
                        int(count) % 200 == 1
                        and args.enable_speech
                        and float(mask_detected_prob) < args.mask_prob_threshold
                    ):
                        engine.play_mp3(speak)

            if args.debug:
                text = f"Face Detection Inference time: {face_detect_infer_time:.3f} ms"
                input_feed.add_text(text, frame, (15, input_feed.source_height - 80))
                text = (
                    f"Face Mask Detection Inference time: {mask_detect_infer_time:.3f} ms"
                )
                input_feed.add_text(text, frame, (15, input_feed.source_height - 60))

                input_feed.show(input_feed.resize(frame))

    finally:
        input_feed.close()


if __name__ == "__main__":
    # Grab command line args
    args = arg_parser()
    # Perform inference on the input stream
    main(args)
