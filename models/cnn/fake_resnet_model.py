# A fake resnet model built for gradient testing. Using tf.float64.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from google.protobuf.text_format import Merge

from models.base.nnlib import weight_variable_cpu
from models.cnn.cnn_model import CNNModel
from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig
from models.model_factory import RegisterModel


class FakeResnetModule(object):
    """A fake resnet."""

    def __init__(self, config, is_training=True):
        """
        Resnet module constructor.

        :param config:      [object]    Configuration object, see configs/resnet_module_config.proto.
        """
        self._config = config
        self._data_format = config.data_format
        self._is_training = is_training

    def __call__(self, inp):
        """
        Builds Resnet graph.

        :param inp:         [Tensor]    Input tensor to the Resnet, [B, H, W, C].

        :return:            [Tensor]    Output tensor, [B, Ho, Wo, Co] if not build classifier,
                                        [B, K] if build classifier.
        """
        config = self.config
        assert self.data_format == 'NHWC'
        # Dùng shape động để an toàn trong graph mode
        bsz = tf.shape(inp)[0]
        h = tf.reshape(inp, [bsz, -1])
        if config.build_classifier:
            with tf.variable_scope('logit'):
                h = self._fully_connected(h, config.num_classes)
        return h

    def _fully_connected(self, x, out_dim):
        """
        A FullyConnected layer for final output.
        """
        x_shape = x.get_shape()
        d = x_shape[1]
        w = self._weight_variable(
            [d, out_dim],
            init_method='uniform_scaling',
            init_param={'factor': 1.0},
            weight_decay=self.config.weight_decay,
            dtype=self.dtype,
            name='w')
        b = self._weight_variable(
            [out_dim], init_method='constant', init_param={'val': 0.0}, name='b', dtype=self.dtype)
        return tf.nn.xw_plus_b(x, w, b)

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """
        A wrapper to declare variables on CPU. See nnlib.py:weight_variable_cpu.
        """
        return weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed)

    @property
    def config(self):
        return self._config

    @property
    def data_format(self):
        return self._data_format

    @property
    def dtype(self):
        return tf.float64


@RegisterModel('fake-resnet')
class FakeResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self, config, is_training=True, inp=None, label=None, batch_size=None):
        self._config = config
        self._is_training = is_training
        super(FakeResnetModel, self).__init__(
            config,
            self._get_resnet_module(),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)

    def _get_resnet_module(self):
        return FakeResnetModule(self.config.resnet_module_config, is_training=self.is_training)

    @classmethod
    def create_from_file(cls,
                         config_filename,
                         is_training=True,
                         inp=None,
                         label=None,
                         batch_size=None):
        config = ResnetModelConfig()
        Merge(open(config_filename).read(), config)
        return cls(config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    @property
    def dtype(self):
        return tf.float64


class FakeAssignedWeightsResnetModule(FakeResnetModule):
    def __init__(self, config, weights_dict, is_training=True):
        self._weights_dict = weights_dict
        self._create_new_var = weights_dict is None
        super(FakeAssignedWeightsResnetModule, self).__init__(config, is_training=is_training)

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """If weights_dict is not None, reuse stored variables."""
        var = super(FakeAssignedWeightsResnetModule, self)._weight_variable(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=0)
        if self.create_new_var:
            if self.weights_dict is None:
                self.weights_dict = {}
            self.weights_dict[var.name] = var
            return var
        else:
            return self.weights_dict[var.name]

    def _batch_norm(self, name, x):
        """Simple BN without moving averages (giữ nguyên hành vi test)."""
        if self.data_format == 'NCHW':
            axis = 1
            axes = [0, 2, 3]
        else:
            axis = -1
            axes = [0, 1, 2]
        with tf.variable_scope('BatchNorm'):
            beta = self._weight_variable([int(x.get_shape()[axis])], name='beta', dtype=self.dtype)
        mean, var = tf.nn.moments(x, axes=axes)
        if self.data_format == 'NCHW':
            beta = tf.reshape(beta, [1, -1, 1, 1])
            mean = tf.reshape(mean, [1, -1, 1, 1])
            var = tf.reshape(var, [1, -1, 1, 1])
        return tf.nn.batch_normalization(x, mean, var, beta, None, 0.001)

    @property
    def create_new_var(self):
        return self._create_new_var

    @property
    def weights_dict(self):
        return self._weights_dict

    @weights_dict.setter
    def weights_dict(self, weights_dict):
        self._weights_dict = weights_dict


@RegisterModel('fake-assign-wts-resnet')
class FakeAssignedWeightsResnetModel(CNNModel):
    """Resnet model with externally assigned example weights."""

    def __init__(self,
                 config,
                 weights_dict,
                 is_training=True,
                 inp=None,
                 label=None,
                 ex_wts=None,
                 batch_size=None):
        # Example weights (float64 để khớp dtype model)
        if ex_wts is None:
            w = tf.placeholder(self.dtype, [batch_size], name='w')
        else:
            w = ex_wts
        self._ex_wts = w
        super(FakeAssignedWeightsResnetModel, self).__init__(
            config,
            FakeAssignedWeightsResnetModule(
                config.resnet_module_config, weights_dict, is_training=is_training),
            is_training=is_training,
            inp=inp,
            label=label,
            batch_size=batch_size)

    @classmethod
    def create_from_file(cls,
                         config_filename,
                         is_training=True,
                         inp=None,
                         label=None,
                         batch_size=None):
        config = ResnetModelConfig()
        Merge(open(config_filename).read(), config)
        return cls(config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    def _compute_loss(self, output):
        """Weighted softmax CE in float64."""
        with tf.variable_scope('costs'):
            label = tf.one_hot(self.label, self._cnn_module.config.num_classes, dtype=self.dtype)
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
            xent_avg = tf.reduce_mean(xent, name='xent')
            xent_wt = tf.reduce_sum(xent * self.ex_wts, name='xent_wt')
            cost = xent_wt + self._decay()
            self._cross_ent_avg = xent_avg
            self._cross_ent_wt = xent_wt
        return cost

    @property
    def ex_wts(self):
        return self._ex_wts

    @property
    def cross_ent(self):
        return self._cross_ent_avg

    @property
    def dtype(self):
        return tf.float64
