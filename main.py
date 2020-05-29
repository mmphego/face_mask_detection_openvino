#!/usr/bin/env python3

import time

from argparse import ArgumentParser

import cv2
import numpy as np
from responsive_voice.voices import UKEnglishMale


from inference import Network

engine = UKEnglishMale()
mp3_file = engine.get_mp3("Please wear your MASK, or you will become a statistic!!")


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
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for detections filtering" "(0.8 by default)",
    )
    parser.add_argument(
        "--out", action="store_true", help="Write video to file.",
    )
    parser.add_argument(
        "--ffmpeg", action="store_true", help="Flush video to FFMPEG.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )

    return parser.parse_args()


def draw_boxes(frame, f_result, m_result, count, prob_threshold, width, height):
    """Draw bounding boxes onto the frame."""
    loc = 20
    for box in f_result[0][0]:  # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= prob_threshold:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            _y = ymax + loc if ymax + loc > loc else ymax - loc

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            detected_threshold = round(float(m_result.flatten()), 3)
            if detected_threshold > 0.3:
                label = ("Mask", (0, 255, 0))  # Color format: BGR
            else:
                label = ("No Mask", (0, 0, 255))
                if int(count) % 200 == 1:
                    engine.play_mp3(mp3_file)
            cv2.putText(
                frame,
                f"{label[0]}: {detected_threshold :.2f}%",
                (xmin - 2 * loc, _y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=label[1],
                thickness=2,
            )
    return frame


def process_frame(frame, height, width):
    """Helper function for processing frame"""
    p_frame = cv2.resize(frame, (width, height))
    # Change data layout from HWC to CHW
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(1, *p_frame.shape)
    return p_frame


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    face_infer_network = Network()
    mask_infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    try:
        mask_infer_network.load_model(
            model_xml=args.mask_model,
            device=args.device,
        )
        face_infer_network.load_model(
            model_xml=args.face_model,
            device=args.device,
        )
    except Exception:
        raise

    if args.input.lower() == "cam":
        video_file = 0
    else:
        video_file = args.input
        assert os.path.isfile(video_file)

    stream = cv2.VideoCapture(video_file)
    stream.open(video_file)

    # Grab the shape of the input
    orig_width = int(stream.get(3))
    orig_height = int(stream.get(4))
    average_infer_time = []

    _, _, input_height, input_width = face_infer_network.get_input_shape()
    _, _, mask_input_height, mask_input_width = mask_infer_network.get_input_shape()

    if not stream.isOpened():
        msg = "Cannot open video source!!!"
        raise RuntimeError(msg)
    count = 0

    while stream.isOpened():
        # Grab the next stream.
        (grabbed, frame) = stream.read()
        # If the frame was not grabbed, then we might have reached end of steam,
        # then break
        if not grabbed:
            break

        count += 1
        p_frame = process_frame(frame, input_height, input_width)
        m_frame = process_frame(frame, mask_input_height, mask_input_width)

        start_infer = time.time()
        face_infer_network.exec_net(p_frame)
        mask_infer_network.exec_net(m_frame)
        if face_infer_network.wait() == 0 and mask_infer_network.wait() == 0:
            f_result = face_infer_network.get_output()
            m_result = mask_infer_network.get_output()
            end_infer = time.time() - start_infer
            average_infer_time.append(end_infer)
            message = f"Inference time: {end_infer*1000:.2f}ms"
            cv2.putText(
                frame, message, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1
            )

            # Draw the boxes onto the input
            out_frame = draw_boxes(
                frame,
                f_result,
                m_result,
                count,
                prob_threshold,
                orig_width,
                orig_height,
            )

        if args.debug:
            cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Grab command line args
    args = arg_parser()
    # Perform inference on the input stream
    infer_on_stream(args)
