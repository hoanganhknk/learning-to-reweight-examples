# Multi-tower CNN model for training CNN on multiple towers (multiple GPUs).

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.cnn.cnn_model import CNNModel
from utils import logger

log = logger.get()


class MultiTowerCNNModel(CNNModel):
    """Multi Tower CNN Model."""

    def __init__(self,
                 config,
                 cnn_module,
                 is_training=True,
                 inp=None,
                 label=None,
                 batch_size=None,
                 num_replica=2):
        """
        Multi Tower CNN constructor.

        :param config:      [object]    Configuration object.
        :param cnn_module:  [object]    A CNN module which builds the main graph.
        :param is_training: [bool]      Whether in training mode, default True.
        :param inp:         [Tensor]    Inputs to the network, optional, default placeholder.
        :param label:       [Tensor]    Labels for training, optional, default placeholder.
        :param batch_size:  [int]       Number of examples in batch dimension (optional).
        :param num_replica: [int]       Number of in-graph replicas (number of GPUs).
        """
        with tf.device(self._get_device("cpu", 0)):
            self._num_replica = num_replica
            self._avg_cost = None
            self._avg_cross_ent = None
            self._stack_output = None
            super(MultiTowerCNNModel, self).__init__(
                config,
                cnn_module,
                is_training=is_training,
                inp=inp,
                label=label,
                batch_size=batch_size)

    def _build_graph(self, inp):
        """
        Builds core computation graph from inputs to outputs.

        :param inp:            [Tensor]     4D float tensor, inputs to the network.

        :return                [Tensor]     output tensor.
        """
        inputs = tf.split(inp, self.num_replica, axis=0)
        outputs = []
        for ii in range(self.num_replica):
            _device = self._get_replica_device(ii)
            with tf.device(_device):
                with tf.name_scope("replica_{}".format(ii)):
                    outputs.append(self._cnn_module(inputs[ii]))
                    log.info("Replica {} forward built on {}".format(ii, _device))
                    # bật reuse cho các tower sau tower đầu
                    tf.compat.v1.get_variable_scope().reuse_variables()
        # Reset reuse flag (API private, giữ hành vi gốc)
        tf.compat.v1.get_variable_scope()._reuse = None
        return outputs

    def _compute_loss(self, output):
        """
        Computes the total loss function.

        :param output:          [list]      Outputs of the network from each tower.

        :return                 [list]      Loss value from each tower.
        """
        labels = tf.split(self.label, self.num_replica, axis=0)
        xent_list = []
        cost_list = []
        for ii, (_output, _label) in enumerate(zip(output, labels)):
            with tf.device(self._get_replica_device(ii)):
                with tf.name_scope("replica_{}".format(ii)):
                    _xent = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=_output, labels=_label))
                    xent_list.append(_xent)
                    _cost = _xent
                    if self.is_training:
                        _cost += self._decay()
                    cost_list.append(_cost)
        self._cross_ent = xent_list
        return cost_list

    def _compute_gradients(self, cost, var_list=None):
        """
        :params cost            [list]      List of loss values from each tower.

        :return                 [list]      List of pairs of (gradient, variable) where the gradient
        has been averaged across all towers.
        """
        grads_and_vars = []
        for ii, _cost in enumerate(cost):
            with tf.device(self._get_replica_device(ii)):
                with tf.name_scope("replica_{}".format(ii)):
                    var_list = tf.trainable_variables()
                    grads = tf.gradients(_cost, var_list)
                    grads_and_vars.append(list(zip(grads, var_list)))
        avg_grads_and_vars = self._average_gradients(grads_and_vars)
        return avg_grads_and_vars

    def _average_gradients(self, tower_grads):
        """
        Calculates the average gradient for each shared variable across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, v in grad_and_vars:
                if g is None:
                    log.warning('No gradient for variable "{}"'.format(v.name))
                    grads.append(None)
                    break
                else:
                    grads.append(tf.expand_dims(g, 0))

            if grads[0] is None:
                grad = None
            else:
                grad = tf.concat(grads, axis=0)
                grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]  # shared variable
            average_grads.append((grad, v))
        return average_grads

    def _get_device(self, device_name="cpu", device_id=0):
        return "/{}:{:d}".format(device_name, device_id)

    def _get_replica_device(self, replica_id):
        return self._get_device("gpu", replica_id)

    @property
    def cost(self):
        if self._avg_cost is None:
            self._avg_cost = tf.reduce_mean(tf.stack(self._cost))
        return self._avg_cost

    @property
    def cross_ent(self):
        if self._avg_cross_ent is None:
            self._avg_cross_ent = tf.reduce_mean(tf.stack(self._cross_ent))
        return self._avg_cross_ent

    @property
    def output(self):
        if self._stack_output is None:
            self._stack_output = tf.concat(self._output, axis=0)
        return self._stack_output

    @property
    def num_replica(self):
        return self._num_replica
