FROM mmphego/intel-openvino
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
