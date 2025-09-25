# Basic neural networks using TensorFlow (TF2 in TF1-compat mode).
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import logger

log = logger.get()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    weight_decay=None,
                    name=None,
                    trainable=True,
                    seed=0):
    """
    Declares a variable.

    :param shape:         [list]     Shape of the weights.
    :param init_method:   [string]   One of 'constant', 'truncated_normal', 'uniform_scaling', 'xavier'.
    :param dtype          [dtype]    TensorFlow dtype.
    :param init_param:    [dict]     Init parameters.
    :param weight_decay:  [float]    L2 weight decay.
    :param name:          [string]   Variable name.
    :param trainable:     [bool]     Trainable flag.
    :param seed:          [int]      Random seed.

    :return:              [tf.Variable]
    """
    if dtype != tf.float32:
        log.warning('Not using float32, currently using {}'.format(dtype))

    if init_method is None:
        initializer = tf.zeros_initializer(dtype=dtype)
    elif init_method == 'truncated_normal':
        mean = 0.0 if not init_param or 'mean' not in init_param else init_param['mean']
        stddev = 0.1 if not init_param or 'stddev' not in init_param else init_param['stddev']
        log.info('Normal initialization std {:.3e}'.format(stddev))
        initializer = tf.truncated_normal_initializer(mean=mean, stddev=stddev, seed=seed, dtype=dtype)
    elif init_method == 'uniform_scaling':
        factor = 1.0 if not init_param or 'factor' not in init_param else init_param['factor']
        log.info('Uniform initialization scale {:.3e}'.format(factor))
        initializer = tf.uniform_unit_scaling_initializer(factor=factor, seed=seed, dtype=dtype)
    elif init_method == 'constant':
        value = 0.0 if not init_param or 'val' not in init_param else init_param['val']
        initializer = tf.constant_initializer(value=value, dtype=dtype)
    elif init_method == 'xavier':
        # tf.contrib.layers.xavier_initializer is removed in TF2
        # Use Glorot (Xavier) from Keras initializers. Original code used uniform=False => normal.
        initializer = tf.compat.v1.keras.initializers.glorot_normal()
    else:
        raise ValueError('Non supported initialization method!')

    try:
        shape_int = [int(ss) for ss in shape]
        log.info('Weight shape {}'.format(shape_int))
    except Exception:
        pass

    if weight_decay is not None and weight_decay > 0.0:
        def _reg(x):
            return tf.multiply(tf.nn.l2_loss(x), weight_decay)
        reg = _reg
        log.info('Weight decay {}'.format(weight_decay))
    else:
        reg = None

    var = tf.get_variable(
        name, shape, initializer=initializer, regularizer=reg, dtype=dtype, trainable=trainable
    )

    if not weight_decay:
        log.warning('No weight decay for {}'.format(var.name))

    log.info('Initialized weight {}'.format(var.name))
    return var


def weight_variable_cpu(shape,
                        init_method=None,
                        dtype=tf.float32,
                        init_param=None,
                        weight_decay=None,
                        name=None,
                        trainable=True,
                        seed=0):
    """
    Declares variables on CPU. See weight_variable for usage.
    """
    with tf.device('/cpu:0'):
        return weight_variable(
            shape,
            init_method=init_method,
            dtype=dtype,
            init_param=init_param,
            weight_decay=weight_decay,
            name=name,
            trainable=trainable,
            seed=seed
        )


def concat(x, axis):
    """Concatenates a list of tensors along axis."""
    return tf.concat(x, axis=axis)


def split(x, num, axis):
    """Splits a tensor into `num` parts along axis."""
    return tf.split(x, num_or_size_splits=num, axis=axis)


def stack(x):
    """Stacks a list of tensors along a new axis (axis=0)."""
    return tf.stack(x)


def cnn(x,
        filter_size,
        strides,
        pool_fn,
        pool_size,
        pool_strides,
        act_fn,
        dtype=tf.float32,
        add_bias=True,
        weight_decay=None,
        scope='cnn',
        trainable=True):
    """
    Builds a convolutional neural network.
    """
    num_layer = len(filter_size)
    h = x
    with tf.variable_scope(scope):
        for ii in range(num_layer):
            with tf.variable_scope('layer_{}'.format(ii)):
                w = weight_variable_cpu(
                    filter_size[ii],
                    init_method='xavier',
                    dtype=dtype,
                    weight_decay=weight_decay,
                    name='w',
                    trainable=trainable)
                h = tf.nn.conv2d(h, w, strides=strides[ii], padding='SAME', name='conv')
                if add_bias:
                    b = weight_variable_cpu(
                        [filter_size[ii][3]],
                        init_method='constant',
                        dtype=dtype,
                        init_param={'val': 0},
                        name='b',
                        trainable=trainable)
                    h = tf.add(h, b, name='conv_bias')
                if act_fn[ii] is not None:
                    h = act_fn[ii](h, name='act')
                if pool_fn[ii] is not None:
                    h = pool_fn[ii](h, pool_size[ii], strides=pool_strides[ii], padding='SAME', name='pool')
    return h


def mlp(x,
        dims,
        is_training=True,
        act_fn=None,
        dtype=tf.float32,
        add_bias=True,
        weight_decay=None,
        scope='mlp',
        dropout=None,
        trainable=True):
    """
    Builds a multi-layer perceptron.
    """
    num_layer = len(dims) - 1
    h = x
    with tf.variable_scope(scope):
        for ii in range(num_layer):
            with tf.variable_scope('layer_{}'.format(ii)):
                dim_in = dims[ii]
                dim_out = dims[ii + 1]
                w = weight_variable_cpu(
                    [dim_in, dim_out],
                    init_method='xavier',
                    dtype=dtype,
                    weight_decay=weight_decay,
                    name='w',
                    trainable=trainable)
                h = tf.matmul(h, w, name='linear')
                if add_bias:
                    b = weight_variable_cpu(
                        [dim_out],
                        init_method='constant',
                        dtype=dtype,
                        init_param={'val': 0.0},
                        name='b',
                        trainable=trainable)
                    h = tf.add(h, b, name='linear_bias')
                if act_fn and act_fn[ii] is not None:
                    h = act_fn[ii](h)
                if dropout is not None and dropout[ii]:
                    log.info('Apply dropout 0.5')
                    keep_prob = 0.5 if is_training else 1.0
                    h = tf.nn.dropout(h, keep_prob=keep_prob)
    return h


def batch_norm(x,
               is_training,
               gamma=None,
               beta=None,
               eps=1e-10,
               name='bn_out',
               decay=0.99,
               dtype=tf.float32,
               data_format='NHWC'):
    """
    Applies batch normalization.
    Returns (normed, update_ops_or_None)
    """
    if data_format == 'NHWC':
        n_out = x.get_shape()[-1]
        axes = [0, 1, 2]
    elif data_format == 'NCHW':
        n_out = x.get_shape()[1]
        axes = [0, 2, 3]
    else:
        raise ValueError('Unsupported data_format: {}'.format(data_format))

    try:
        n_out = int(n_out)
        shape = [n_out]
    except Exception:
        shape = None

    emean = tf.get_variable(
        'ema_mean',
        shape=shape,
        trainable=False,
        dtype=dtype,
        initializer=tf.constant_initializer(0.0, dtype=dtype))
    evar = tf.get_variable(
        'ema_var',
        shape=shape,
        trainable=False,
        dtype=dtype,
        initializer=tf.constant_initializer(1.0, dtype=dtype))

    if is_training:
        mean, var = tf.nn.moments(x, axes, name='moments')
        ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
        ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps, name=name)
        return normed, [ema_mean_op, ema_var_op]
    else:
        normed = tf.nn.batch_normalization(x, emean, evar, beta, gamma, eps, name=name)
        return normed, None
