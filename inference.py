#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys

import numpy as np

from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        # Initialize the plugin
        # https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html
        self.ie_core = IECore()
        self.network = None
        self.exec_network = None
        self._input_blob = None
        self._output_blob = None
        self.infer_request = None

    def load_model(
        self, model_xml: str, device: str = "CPU", cpu_extension=None
    ) -> None:
        """Load the model given IR files.
#
        Params
        ======
        model: str
            model xml (Intermediate Representation)
        device: str
            Defaults to CPU as device for use in the workspace.
        cpu_extension: str (optional)
        """
        ### TODO: Load the model ###
        # https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        assert os.path.isfile(model_bin) and os.path.isfile(model_xml)
        self._model_size = os.stat(model_bin).st_size / 1024. ** 2

        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.ie_core.add_extension(cpu_extension, device)

        try:
            self.network = self.ie_core.read_network(model=model_xml, weights=model_bin)
        except AttributeError:
            self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.ie_core.load_network(
            network=self.network, device_name=device
        )

        ### TODO: Check for supported layers ###
        supported_layers = self.ie_core.query_network(
            network=self.network, device_name=device
        )
        # Check for any unsupported layers, and let the user know if anything is missing.
        unsupported_layers = [
            l for l in self.network.layers.keys() if l not in supported_layers
        ]
        if len(unsupported_layers) != 0:
            msg = (
                "Unsupported layers found: {}, Check whether extensions are available "
                "to add to IECore.".format(unsupported_layers)
            )
            raise RuntimeError(msg)

        ### TODO: Add any necessary extensions ###
        # Add a CPU extension, if applicable
        if cpu_extension and "CPU" in device:
            self.ie_core.add_extension(cpu_extension, device)

        # Get the input layer
        self._input_blob = next(iter(self.network.inputs))
        self._output_blob = next(iter(self.network.outputs))

    def get_input_shape(self) -> list:
        """Gets the input shape of the network."""
        return self.network.inputs[self._input_blob].shape

    def exec_net(
        self, image: object, request_id: int = 0,
    ):
        """Makes an asynchronous inference request, given an input image."""
        if not isinstance(image, np.ndarray):
            raise IOError("Image not parsed.")
        self.exec_network.start_async(
            request_id=request_id, inputs={self._input_blob: image}
        )

    def wait(self, request_id: int = 0):
        """Checks the status of the inference request."""
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def get_output(self, request_id: int = 0):
        """Returns a list of the results for the output layer of the network."""
        return self.exec_network.requests[request_id].outputs[self._output_blob]
