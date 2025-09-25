# A general CNN model.

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.optim.optimizer_factory import get_optimizer
from utils import logger

log = logger.get()


class CNNModel(object):
    """CNN model."""

    def __init__(self, config, cnn_module, is_training=True, inp=None, label=None, batch_size=None):
        """
        CNN constructor.
        """
        self._config = config
        self._bn_update_ops = None
        self._is_training = is_training
        self._batch_size = batch_size

        # Input.
        if inp is None:
            x = tf.placeholder(
                self.dtype,
                [batch_size, config.input_height, config.input_width, config.num_channels],
                name='x'
            )
        else:
            x = inp

        if label is None:
            y = tf.placeholder(tf.int32, [batch_size], name='y')
        else:
            y = label
        self._input = x
        self._label = y
        self._cnn_module = cnn_module

        logits = self._build_graph(x)
        cost = self._compute_loss(logits)
        self._cost = cost
        self._output = logits
        self._correct = self._compute_correct()

        if not is_training:
            return

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0),
            trainable=False,
            dtype=tf.int64
        )
        learn_rate = tf.get_variable(
            'learn_rate', [],
            initializer=tf.constant_initializer(0.0),
            trainable=False,
            dtype=self.dtype
        )
        self._learn_rate = learn_rate
        self._grads_and_vars = self._compute_gradients(cost)
        log.info('BN update ops:')
        [log.info(op) for op in self.bn_update_ops]
        log.info('Total number of BN updates: {}'.format(len(self.bn_update_ops)))
        self._train_op = self._apply_gradients(
            self._grads_and_vars, global_step=global_step, name='train_step'
        )
        self._global_step = global_step
        self._new_learn_rate = tf.placeholder(self.dtype, shape=[], name='new_learning_rate')
        self._learn_rate_update = tf.assign(self._learn_rate, self._new_learn_rate)

    def _build_graph(self, inp):
        """Builds core computation graph from inputs to outputs."""
        return self._cnn_module(inp)

    def _compute_loss(self, output):
        """Computes the total loss function."""
        with tf.variable_scope('costs'):
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.label)
            xent = tf.reduce_mean(xent, name='xent')
            cost = xent + self._decay()
            self._cross_ent = xent
        return cost

    def _compute_correct(self):
        """Computes number of correct predictions."""
        output_idx = tf.cast(tf.argmax(self.output, axis=1), self.label.dtype)
        return tf.cast(tf.equal(output_idx, self.label), tf.float32)

    def assign_learn_rate(self, session, learn_rate_value):
        """Assigns new learning rate."""
        log.info('Adjusting learning rate to {}'.format(learn_rate_value))
        session.run(self._learn_rate_update, feed_dict={self._new_learn_rate: learn_rate_value})

    def _apply_gradients(self, grads_and_vars, global_step=None, name='train_step'):
        """
        Applies the gradients globally.
        """
        # opt = get_optimizer(self.config.optimizer_config.type)(
        #     self.learn_rate, self.config.optimizer_config.momentum)
        opt = tf.train.MomentumOptimizer(self.learn_rate, 0.9)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step, name=name)
        return train_op

    def _compute_gradients(self, cost, var_list=None):
        """
        Computes the gradients to variables.
        """
        if var_list is None:
            var_list = tf.trainable_variables()
        grads = tf.gradients(cost, var_list)
        return list(zip(grads, var_list))

    def _decay(self):
        """
        Applies L2 weight decay loss.
        """
        weight_decay_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        log.info('Weight decay variables')
        [log.info(x) for x in weight_decay_losses]
        log.info('Total length: {}'.format(len(weight_decay_losses)))
        if len(weight_decay_losses) > 0:
            return tf.add_n(weight_decay_losses)
        else:
            log.warning('No weight decay variables!')
            return tf.constant(0.0, dtype=self.dtype)

    def _get_feed_dict(self, inp=None, label=None):
        """Generates feed dict."""
        if inp is None and label is None:
            return None
        feed_data = {}
        if inp is not None:
            feed_data[self.input] = inp
        if label is not None:
            feed_data[self.label] = label
        return feed_data

    def infer_step(self, sess, inp=None):
        """Runs one inference step."""
        return sess.run(self.output, feed_dict=self._get_feed_dict(inp=inp))

    def eval_step(self, sess, inp=None, label=None):
        """Runs one eval step."""
        return sess.run(
            [self.correct, self.cross_ent], feed_dict=self._get_feed_dict(inp=inp, label=label)
        )

    def train_step(self, sess, inp=None, label=None):
        """Runs one training step."""
        results = sess.run(
            [self.cross_ent, self.train_op] + self.bn_update_ops,
            feed_dict=self._get_feed_dict(inp=inp, label=label)
        )
        return results[0]

    @property
    def cost(self):
        return self._cost

    @property
    def train_op(self):
        return self._train_op

    @property
    def bn_update_ops(self):
        if self._bn_update_ops is None:
            self._bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        return self._bn_update_ops

    @property
    def config(self):
        return self._config

    @property
    def learn_rate(self):
        return self._learn_rate

    @property
    def dtype(self):
        return tf.float32

    @property
    def is_training(self):
        return self._is_training

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

    @property
    def correct(self):
        if self._correct is None:
            self._correct = self._compute_correct()
        return self._correct

    @property
    def label(self):
        return self._label

    @property
    def cross_ent(self):
        return self._cross_ent

    @property
    def global_step(self):
        return self._global_step

    @property
    def grads_and_vars(self):
        return self._grads_and_vars
