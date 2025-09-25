# Unit tests for multi-tower CNN model.

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from models.cnn.multi_tower_resnet_model import MultiTowerResnetModel  # NOQA
from models.cnn.resnet_model import ResnetModel  # NOQA
from models.model_factory import get_model_from_file
from utils.test_utils import check_two_dict


class MultiTowerModelTests(tf.test.TestCase):
    def test_fw(self):
        """Tests the forward computation is the same."""
        with tf.Graph().as_default(), self.test_session() as sess:
            config_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'configs/resnet-test.prototxt')
            np.random.seed(0)
            xval = np.random.uniform(-1.0, 1.0, [10, 32, 32, 3]).astype(np.float32)
            x = tf.constant(xval)
            x1 = x[:5, :, :, :]
            x2 = x[5:, :, :, :]
            # Tách hai lượt chạy riêng (BN nhạy batch)
            with tf.compat.v1.variable_scope("Model", reuse=None):
                m11 = get_model_from_file("resnet", config_file, inp=x1)
            with tf.compat.v1.variable_scope("Model", reuse=True):
                m12 = get_model_from_file("resnet", config_file, inp=x2)
            with tf.compat.v1.variable_scope("Model", reuse=True):
                m2 = get_model_from_file("multi-tower-resnet", config_file, num_replica=2, inp=x)
            sess.run(tf.global_variables_initializer())
            y11, y12, y2 = sess.run([m11.output, m12.output, m2.output])
            np.testing.assert_allclose(y11, y2[:5, :], rtol=1e-5)
            np.testing.assert_allclose(y12, y2[5:, :], rtol=1e-5)

    def test_bk(self):
        """Tests the backward computation is the same."""
        with tf.Graph().as_default(), self.test_session() as sess:
            config_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'configs/resnet-test.prototxt')
            np.random.seed(0)
            xval = np.random.uniform(-1.0, 1.0, [10, 32, 32, 3]).astype(np.float32)
            yval = np.floor(np.random.uniform(0, 9.9, [10])).astype(np.int32)
            x = tf.constant(xval)
            y = tf.constant(yval)
            x1, x2 = x[:5, :, :, :], x[5:, :, :, :]
            y1, y2 = y[:5], y[5:]
            with tf.compat.v1.variable_scope("Model", reuse=None):
                m11 = get_model_from_file("resnet", config_file, inp=x1, label=y1)
            with tf.compat.v1.variable_scope("Model", reuse=True):
                m12 = get_model_from_file("resnet", config_file, inp=x2, label=y2)
            with tf.compat.v1.variable_scope("Model", reuse=True):
                m2 = get_model_from_file("multi-tower-resnet", config_file, num_replica=2, inp=x, label=y)
            sess.run(tf.global_variables_initializer())

            # Lấy tên biến và gradient (dùng list(map(...)) cho an toàn)
            name_list11 = list(map(lambda t: t[1].name, m11.grads_and_vars))
            grads11     = list(map(lambda t: t[0],       m11.grads_and_vars))
            g11         = sess.run(grads11)
            gdict11     = dict(zip(name_list11, g11))

            name_list12 = list(map(lambda t: t[1].name, m12.grads_and_vars))
            grads12     = list(map(lambda t: t[0],       m12.grads_and_vars))
            g12         = sess.run(grads12)
            gdict12     = dict(zip(name_list12, g12))

            # Trung bình 2 tower đơn và so với multi-tower
            name_list2 = list(map(lambda t: t[1].name, m2.grads_and_vars))
            grads2     = list(map(lambda t: t[0],       m2.grads_and_vars))
            g2         = sess.run(grads2)
            gdict2     = dict(zip(name_list2, g2))

            name_list1 = name_list11
            g1 = [(gdict11[k] + gdict12[k]) / 2.0 for k in name_list1]
            gdict1 = dict(zip(name_list1, g1))
            check_two_dict(gdict1, gdict2, tol=1e-1)


if __name__ == "__main__":
    tf.test.main()
