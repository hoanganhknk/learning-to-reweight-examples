# Residual networks module.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.base.nnlib import concat, weight_variable_cpu  # bỏ batch_norm
from utils.logger import get as get_logger

log = get_logger()


class ResnetModule(object):
    """Resnet module."""

    def __init__(self, config, is_training=True):
        """
        Resnet module constructor.

        :param config:  [object]  See configs/resnet_module_config.proto.
        """
        self._config = config
        self._dtype = tf.float32
        self._data_format = config.data_format
        self._is_training = is_training

    def __call__(self, inp):
        """Builds Resnet graph."""
        config = self.config
        strides = config.strides
        dropout = config.dropout
        if len(config.dilations) == 0:
            dilations = [1] * len(config.strides)
        else:
            dilations = config.dilations
        assert len(config.strides) == len(dilations), 'Need to pass in lists of same size.'
        filters = [ff for ff in config.num_filters]  # copy
        init_filter_size = config.init_filter_size

        if self.data_format == 'NCHW':
            inp = tf.transpose(inp, [0, 3, 1, 2])

        with tf.variable_scope('init'):
            h = self._conv('init_conv', inp, init_filter_size, self.config.num_channels, filters[0],
                           self._stride_arr(config.init_stride), 1)
            h = self._batch_norm('init_bn', h)
            h = self._relu('init_relu', h)

            if config.init_max_pool:
                h = tf.nn.max_pool(
                    h,
                    self._stride_arr(3),
                    self._stride_arr(2),
                    'SAME',
                    data_format=self.data_format)

        if config.use_bottleneck:
            res_func = self._bottleneck_residual
            for ii in range(1, len(filters)):
                filters[ii] *= 4
        else:
            res_func = self._residual

        # One loop over all units
        nlayers = sum(config.num_residual_units)
        ss = 0
        ii = 0
        for _ in range(nlayers):
            if ii == 0:
                no_activation = (ss == 0)
                in_filter = filters[ss]
                stride = self._stride_arr(strides[ss])
            else:
                no_activation = False
                in_filter = filters[ss + 1]
                stride = self._stride_arr(1)

            out_filter = filters[ss + 1]

            if dilations[ss] > 1:
                if config.use_bottleneck:
                    dilation = [dilations[ss] // strides[ss], dilations[ss], dilations[ss]]
                else:
                    dilation = [dilations[ss] // strides[ss], dilations[ss]]
            else:
                dilation = [1, 1, 1] if config.use_bottleneck else [1, 1]

            with tf.variable_scope('unit_{}_{}'.format(ss + 1, ii)):
                h = res_func(
                    h,
                    in_filter,
                    out_filter,
                    stride,
                    dilation,
                    dropout=dropout,
                    no_activation=no_activation)

            if (ii + 1) % config.num_residual_units[ss] == 0:
                ss += 1
                ii = 0
            else:
                ii += 1

        if isinstance(h, tuple):
            h = concat(h, axis=3)

        if config.build_classifier:
            with tf.variable_scope('unit_last'):
                h = self._batch_norm('final_bn', h)
                h = self._relu('final_relu', h)
            h = self._global_avg_pool(h)
            with tf.variable_scope('logit'):
                h = self._fully_connected(h, config.num_classes)

        return h

    def _weight_variable(self,
                         shape,
                         init_method=None,
                         dtype=tf.float32,
                         init_param=None,
                         weight_decay=None,
                         name=None,
                         trainable=True,
                         seed=0):
        """Declare variables on CPU."""
        return weight_variable_cpu(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed)

    def _stride_arr(self, stride):
        """Map scalar stride to 4D stride array."""
        if self.data_format == 'NCHW':
            return [1, 1, stride, stride]
        else:
            return [1, stride, stride, 1]

    def _batch_norm(self, name, x):
        """
        BatchNorm thuần TF (không dùng Keras/tf.layers để tránh lỗi Keras 3).
        - Tự tạo moving_mean/moving_var.
        - Thêm update ops vào tf.GraphKeys.UPDATE_OPS để CNNModel chạy cùng train_op.
        """
        # Trục kênh & axes để tính moments
        if self.data_format == 'NCHW':
            ch_axis = 1
            stats_axes = [0, 2, 3]
            reshape_to = lambda v: tf.reshape(v, [1, -1, 1, 1])
        else:  # NHWC
            ch_axis = -1
            stats_axes = [0, 1, 2]
            reshape_to = lambda v: v  # [C] broadcast được theo NHWC

        with tf.variable_scope(name):
            channels = int(x.get_shape()[ch_axis])
            gamma = tf.get_variable('gamma', shape=[channels],
                                    initializer=tf.ones_initializer(), dtype=self.dtype)
            beta  = tf.get_variable('beta',  shape=[channels],
                                    initializer=tf.zeros_initializer(), dtype=self.dtype)
            moving_mean = tf.get_variable('moving_mean', shape=[channels],
                                          initializer=tf.zeros_initializer(),
                                          trainable=False, dtype=self.dtype)
            moving_var  = tf.get_variable('moving_var',  shape=[channels],
                                          initializer=tf.ones_initializer(),
                                          trainable=False, dtype=self.dtype)

            momentum = 0.997
            eps = 1e-5

            if self.is_training:
                batch_mean, batch_var = tf.nn.moments(x, stats_axes)

                update_mean = tf.assign(moving_mean, moving_mean * momentum + batch_mean * (1.0 - momentum))
                update_var  = tf.assign(moving_var,  moving_var  * momentum + batch_var  * (1.0 - momentum))

                # Cho vào UPDATE_OPS để được chạy khi train
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)

                mean_to_use = batch_mean
                var_to_use  = batch_var
            else:
                mean_to_use = moving_mean
                var_to_use  = moving_var

            # Với NCHW cần reshape tham số về [1,C,1,1]
            scale = reshape_to(gamma)
            offset = reshape_to(beta)
            mean_b = reshape_to(mean_to_use)
            var_b  = reshape_to(var_to_use)

            y = tf.nn.batch_normalization(x, mean_b, var_b, offset, scale, eps)
            return y

    def _possible_downsample(self, x, in_filter, out_filter, stride):
        """Downsample bằng avg_pool + pad kênh khi cần."""
        if stride[2] > 1:
            with tf.variable_scope('downsample'):
                x = tf.nn.avg_pool(x, stride, stride, 'SAME', data_format=self.data_format)

        if in_filter < out_filter:
            with tf.variable_scope('pad'):
                pad_ = [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                if self.data_format == 'NCHW':
                    x = tf.pad(x, [[0, 0], pad_, [0, 0], [0, 0]])
                else:
                    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], pad_])
        return x

    def _residual_inner(self,
                        x,
                        in_filter,
                        out_filter,
                        stride,
                        dilation_rate,
                        dropout=0.0,
                        no_activation=False):
        """Bên trong residual (2 conv)."""
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
                if dropout > 0.0 and self.is_training:
                    log.info('Using dropout with {:d}%'.format(int(dropout * 100)))
                    x = tf.nn.dropout(x, keep_prob=(1.0 - dropout))
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter, out_filter,
                           self._stride_arr(1), dilation_rate[1])
        return x

    def _residual(self,
                  x,
                  in_filter,
                  out_filter,
                  stride,
                  dilation_rate,
                  dropout=0.0,
                  no_activation=False):
        """Residual block 2 conv."""
        orig_x = x
        x = self._residual_inner(
            x, in_filter, out_filter, stride, dilation_rate,
            dropout=dropout, no_activation=no_activation)
        x += self._possible_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _bottleneck_residual_inner(self,
                                   x,
                                   in_filter,
                                   out_filter,
                                   stride,
                                   dilation_rate,
                                   no_activation=False):
        """Bottleneck (3 conv)."""
        with tf.variable_scope('sub1'):
            if not no_activation:
                x = self._batch_norm('bn1', x)
                x = self._relu('relu1', x)
            x = self._conv('conv1', x, 1, in_filter, out_filter // 4, stride, dilation_rate[0])
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu('relu2', x)
            x = self._conv('conv2', x, 3, out_filter // 4, out_filter // 4,
                           self._stride_arr(1), dilation_rate[1])
        with tf.variable_scope('sub3'):
            x = self._batch_norm('bn3', x)
            x = self._relu('relu3', x)
            x = self._conv('conv3', x, 1, out_filter // 4, out_filter,
                           self._stride_arr(1), dilation_rate[2])
        return x

    def _possible_bottleneck_downsample(self, x, in_filter, out_filter, stride):
        """Projection khi stride>1 hoặc thay đổi số kênh."""
        if stride[1] > 1 or in_filter != out_filter:
            x = self._conv('project', x, 1, in_filter, out_filter, stride, 1)
        return x

    def _bottleneck_residual(self,
                             x,
                             in_filter,
                             out_filter,
                             stride,
                             dilation_rate,
                             no_activation=False):
        """Residual kiểu bottleneck."""
        orig_x = x
        x = self._bottleneck_residual_inner(
            x, in_filter, out_filter, stride, dilation_rate, no_activation=no_activation)
        x += self._possible_bottleneck_downsample(orig_x, in_filter, out_filter, stride)
        return x

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, dilation_rate):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            init_method = 'truncated_normal'
            init_param = {'mean': 0, 'stddev': np.sqrt(2.0 / n)}
            kernel = self._weight_variable(
                [filter_size, filter_size, in_filters, out_filters],
                init_method=init_method,
                init_param=init_param,
                weight_decay=self.config.weight_decay,
                dtype=self.dtype,
                name='w')
            if dilation_rate == 1:
                return tf.nn.conv2d(
                    x, kernel, strides, padding='SAME', data_format=self.data_format)
            elif dilation_rate > 1:
                # atrous_conv2d chỉ hỗ trợ NHWC trong TF1
                assert self.data_format == 'NHWC', 'Dilated convolution needs NHWC format.'
                assert all([strides[ss] == 1 for ss in range(len(strides))]), 'Strides need to be 1'
                return tf.nn.atrous_conv2d(x, kernel, dilation_rate, padding='SAME')

    def _relu(self, name, x):
        return tf.nn.relu(x, name=name)

    def _fully_connected(self, x, out_dim):
        x_shape = x.get_shape()
        d = int(x_shape[1])
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

    def _global_avg_pool(self, x):
        if self.data_format == 'NCHW':
            return tf.reduce_mean(x, [2, 3])
        else:
            return tf.reduce_mean(x, [1, 2])

    @property
    def config(self):
        return self._config

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_format(self):
        return self._data_format

    @property
    def is_training(self):
        return self._is_training
