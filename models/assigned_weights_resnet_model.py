# Residual networks model with externally assigned weights (parameters).

from __future__ import absolute_import, division, print_function, unicode_literals

# Dùng TF1-compat để chạy API TF1 trong môi trường TF2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import logger
from models.cnn.cnn_model import CNNModel
from models.model_factory import RegisterModel
from models.cnn.resnet_module import ResnetModule
from models.base.nnlib import weight_variable_cpu, batch_norm

# (Tuỳ chọn, chỉ dùng nếu gọi create_from_file)
from google.protobuf.text_format import Merge
from models.cnn.configs.resnet_model_config_pb2 import ResnetModelConfig

log = logger.get()


class AssignedWeightsResnetModule(ResnetModule):
    def __init__(self, config, weights_dict, is_training=True):
        """Initialize the module with a weight dictionary.

        :param config:          [object]  A configuration object.
        :param weights_dict:    [dict]    A dictionary that stores all the parameters.
        :param is_training:     [bool]    Whether in training mode.
        """
        self._weights_dict = weights_dict
        self._create_new_var = weights_dict is None
        super(AssignedWeightsResnetModule, self).__init__(config, is_training=is_training)

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """Declare variables; nếu có weights_dict thì dùng shared var từ đó."""
        var = super(AssignedWeightsResnetModule, self)._weight_variable(
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
        """Batch normalization (đơn giản, chỉ dùng beta)."""
        if self.data_format == 'NCHW':
            axis = 1
            axes = [0, 2, 3]
        else:
            axis = -1
            axes = [0, 1, 2]
        with tf.compat.v1.variable_scope('BatchNorm'):
            beta = self._weight_variable([int(x.get_shape()[axis])], name='beta')
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


@RegisterModel('assign-wts-resnet')
class AssignedWeightsResnetModel(CNNModel):
    """Resnet model."""

    def __init__(self,
                 config,
                 weights_dict,
                 is_training=True,
                 inp=None,
                 label=None,
                 ex_wts=None,
                 batch_size=None):
        """
        Resnet constructor.

        :param config:      [object]    Configuration object.
        :param weights_dict [dict]      Dictionary for assigned weights.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        """
        # Example weights.
        if ex_wts is None:
            w = tf.compat.v1.placeholder(self.dtype, [batch_size], name='w')
        else:
            w = ex_wts
        self._ex_wts = w

        super(AssignedWeightsResnetModel, self).__init__(
            config,
            AssignedWeightsResnetModule(
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
        with open(config_filename, 'r') as f:
            Merge(f.read(), config)
        return cls(config, is_training=is_training, inp=inp, label=label, batch_size=batch_size)

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [Tensor]    Output of the network.

        :return                 [Scalar]    Loss value.
        """
        with tf.compat.v1.variable_scope('costs'):
            label = tf.one_hot(self.label, self._cnn_module.config.num_classes)
            xent = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output)
            xent_avg = tf.reduce_mean(xent, name='xent')
            xent_wt = tf.reduce_sum(xent * self.ex_wts, name='xent_wt')
            cost = xent_wt
            cost += self._decay()
            self._cross_ent_avg = xent_avg
            self._cross_ent_wt = xent_wt
        return cost

    @property
    def ex_wts(self):
        return self._ex_wts

    @property
    def cross_ent(self):
        return self._cross_ent_avg
