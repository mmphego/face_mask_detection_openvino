# Face Mask Detection using OpenVINO

| Details            |              |
|-----------------------|---------------|
| Programming Language: |  [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) |
| Intel OpenVINO ToolKit: |[![OpenVINO 2020.2](https://img.shields.io/badge/openvino-2020.2-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)|
| Docker (Ubuntu OpenVINO pre-installed): | [mmphego/intel-openvino](https://hub.docker.com/r/mmphego/intel-openvino)|
| Hardware Used: | Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz |
| Device: | CPU |
| Blog Post | [![blogpost](https://img.shields.io/badge/BlogPost-Link-brightgreen)](https://blog.mphomphego.co.za/blog/2020/06/02/Face-Mask-Detection-using-Intel-OpenVINO-and-OpenCV.html) |
| Visitors | ![](https://visitor-badge.laobi.icu/badge?page_id=mmphego.face_mask_detection_openvino)|


Face Mask Detection application uses Deep Learning/Machine Learning to recognize if a user is not wearing a mask and issues an alert.

By utilizing pre-trained models and Intel OpenVINO toolkit with OpenCV. This enables us to use the async API which can improve overall frame-rate of the application, rather than wait for inference to complete, the application can continue operating on the host while accelerator is busy.

This application executes 2 parallel infer requests for the Face Mask Detection and Face Detection networks that run simultaneously.

Using a set of the following pre-trained models:
- [face-detection-adas-0001](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html), which is a primary detection network for finding faces.
- face-mask-detection, which is a pretrained model for detecting a mask.

This application can be improved and then integrated with CCTV or other types cameras to detect and identify people without masks in public areas such as shopping centres and etc. This the ever increasing COVID-19 cases world-wide these application could be useful in controlling the spread of the virus.

![Screenshot from 2020-06-01 22-21-47](https://user-images.githubusercontent.com/7910856/83451683-a8d71780-a457-11ea-8eae-185725fefcc9.png)

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

You might also be interested on reading about [AI At The Edge - An Introduction To Intel OpenVINO Toolkit.](https://blog.mphomphego.co.za/blog/2020/05/25/AI-at-the-Edge-An-introduction-to-Intel-OpenVINO-Toolkit.html)

## Support
If you have found this useful, please donate by clicking on the image below:

[![image](https://user-images.githubusercontent.com/7910856/88235803-e4ce7200-cc7b-11ea-8218-c3c04810052c.png)](https://paypal.me/mmphego)

### Tutorial
#### Blog Post

I have created a detailed blog post on the implementation: https://blog.mphomphego.co.za/blog/2020/06/02/Face-Mask-Detection-using-Intel-OpenVINO-and-OpenCV.html

#### YouTube Tutorial

The first of many...

[![Watch the video](https://user-images.githubusercontent.com/7910856/88237923-a8514500-cc80-11ea-9cc8-0692eb0c4d6e.gif)](https://www.youtube.com/watch?v=6r6foGbCHQ0)

## Hardware Requirement

- Minimum Intel Gen 6 processors


## Installation

- Download the docker images with a pre-installed version of OpenVINO 2020.2
```bash
docker pull mmphego/intel-openvino
```

- Download the facemask detection model.
```bash
wget https://github.com/didi/maskdetection/raw/master/model/face_mask.caffemodel
wget https://raw.githubusercontent.com/didi/maskdetection/master/model/deploy.prototxt
```

- Convert model to OpenVINO's `Intermediate Representations` (IR) using the `Model Optimizer`, which will produce `.xml` and `.bin` files.
```bash
docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
mmphego/intel-openvino \
bash -c "/opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
    --framework caffe \
    --input_model face_mask.caffemodel \
    --input_proto deploy.prototxt"
```

- Download face detection model from the model zoo, which will produce `.xml` and `.bin` files.
```bash
docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
mmphego/intel-openvino \
bash -c "/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py \
    --name face-detection-adas-0001 \
    --precision FP16"
```

## Usage

```bash
$ python main.py -h

usage: main.py [-h] -f FACE_MODEL -m MASK_MODEL -i INPUT [-d DEVICE]
               [--face_prob_threshold FACE_PROB_THRESHOLD]
               [--mask_prob_threshold MASK_PROB_THRESHOLD] [--enable-speech]
               [--tts TTS] [--ffmpeg] [--show-bbox] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -f FACE_MODEL, --face-model FACE_MODEL
                        Path to an xml file with a trained model.
  -m MASK_MODEL, --mask-model MASK_MODEL
                        Path to an xml file with a trained model.
  -i INPUT, --input INPUT
                        Path to image or video file or 'cam' for Webcam.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  --face_prob_threshold FACE_PROB_THRESHOLD
                        Probability threshold for face detections filtering
                        (Default: 0.8)
  --mask_prob_threshold MASK_PROB_THRESHOLD
                        Probability threshold for face mask detections
                        filtering(Default: 0.3)
  --enable-speech       Enable speech notification.
  --tts TTS             Text-to-Speech, used for notification.
  --ffmpeg              Flush video to FFMPEG.
  --show-bbox           Show bounding box and stats on screen [debugging].
  --debug               Show output on screen [debugging].

```

### Example Usage

```bash
xhost +;
docker run --rm -ti \
--volume "$PWD":/app \
--env DISPLAY=$DISPLAY \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--device /dev/snd \
--device /dev/video0 \
mmphego/intel-openvino \
bash -c \
"source /opt/intel/openvino/bin/setupvars.sh && \
python main.py \
    --face-model models/face-detection-adas-0001 \
    --mask-model models/face_mask \
    --debug \
    --show-bbox \
    -i resources/mask.mp4"
xhost -;
```

- `--env DISPLAY=$DISPLAY`: Enables GUI applications
- `--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"`: Enable GUI applications
- `--device /dev/snd`: Enable sound from container
- `--device /dev/video0`: Share webcam with container

![ezgif-4-993045572ebb](https://user-images.githubusercontent.com/7910856/88237923-a8514500-cc80-11ea-9cc8-0692eb0c4d6e.gif)

### Packaging the Application
We can use the [Deployment Manager](https://docs.openvinotoolkit.org/latest/_docs_install_guides_deployment_manager_tool.html) present in OpenVINO to create a runtime package from our application. These packages can be easily sent to other hardware devices to be deployed.

To deploy the application to various devices using the Deployment Manager run the steps below.

Note: Choose from the devices listed below.

```bash
DEVICE='cpu' # or gpu, vpu, gna, hddl
docker run --rm -ti \
--volume "$PWD":/app \
mmphego/intel-openvino bash -c "\
  python /opt/intel/openvino/deployment_tools/tools/deployment_manager/deployment_manager.py \
  --targets ${DEVICE} \
  --user_data /app \
  --output_dir . \
  --archive_name face_mask_detection_${DEVICE}"

```

## Credit

- Face mask detection caffe model: [https://github.com/didi/maskdetection](https://github.com/didi/maskdetection)
- [COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning by Adrian Rosebrock ](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
