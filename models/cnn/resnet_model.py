# Residual networks model.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.cnn.cnn_model import CNNModel
from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig
from models.cnn.resnet_module import ResnetModule
from models.model_factory import RegisterModel

from google.protobuf.text_format import Merge


@RegisterModel('resnet')
class ResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self, config, is_training=True, inp=None, label=None, batch_size=None):
        """
        Resnet constructor.

        :param config:      [object]    Configuration object.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        self._config = config
        self._is_training = is_training
        super(ResnetModel, self).__init__(
            config,
            self._get_resnet_module(),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)

    def _get_resnet_module(self):
        return ResnetModule(self.config.resnet_module_config, is_training=self.is_training)

    @classmethod
    def create_from_file(cls,
                         config_filename,
                         is_training=True,
                         inp=None,
                         label=None,
                         batch_size=None):
        config = ResnetModelConfig()
        with open(config_filename, 'r') as f:
            Merge(f.read(), config)
        return cls(config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)
