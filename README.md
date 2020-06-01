# Face Mask Detection using OpenVINO

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![OpenVINO 2020.2](https://img.shields.io/badge/openvino-2020.2-blue.svg)](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html)

Face Mask Detection Platform uses Artificial Network to recognize if a user is not wearing a mask and issues an alert.

![Screenshot from 2020-06-01 22-21-47](https://user-images.githubusercontent.com/7910856/83451683-a8d71780-a457-11ea-8eae-185725fefcc9.png)

**What is OpenVino?**

OpenVino (OpenVisual Inferencing and Neural Network Optimization) is toolkit to develop Deep Learning Application especially for Computer Vision by Intel. OpenVino Enables deep learning inference at the edge and supports heterogeneous execution across computer vision accelerators—CPU, GPU, Intel® Movidius™ Neural Compute Stick, and FPGA—using a common API. [read more](https://docs.openvinotoolkit.org/)

You might also be interested on reading about [AI At The Edge - An Introduction To Intel OpenVINO Toolkit.](https://blog.mphomphego.co.za/blog/2020/05/25/AI-at-the-Edge-An-introduction-to-Intel-OpenVINO-Toolkit.html)

## Hardware Requirement

- Minimum Intel Gen6 processors


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
    --face-model models/face-detection-adas-0001.xml \
    --mask-model models/face_mask.xml \
    --debug \
    -i resources/mask.mp4"
xhost -;
```

- `--env DISPLAY=$DISPLAY`: Enables GUI applications
- `--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"`: Enable GUI applications
- `--device /dev/snd`: Enable sound from container
- `--device /dev/video0`: Share webcam with container

## Credit

- Face mask detection caffe model: [https://github.com/didi/maskdetection](https://github.com/didi/maskdetection)
