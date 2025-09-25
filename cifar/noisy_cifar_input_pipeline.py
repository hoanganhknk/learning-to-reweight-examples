from __future__ import division, print_function

# Dùng TF1-compat để có FixedLenFeature/parse_single_example/...
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from datasets.cifar.cifar_input_pipeline import CifarInputPipeline
from datasets.data_factory import RegisterInputPipeline


@RegisterInputPipeline('cifar-noisy')
class NoisyCifarInputPipeline(CifarInputPipeline):
    def parse_example_proto(self, example_serialized):
        feature_map = {
            'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'label': tf.FixedLenFeature([], dtype=tf.int64,  default_value=-1),
            'clean': tf.FixedLenFeature([], dtype=tf.int64,  default_value=-1),
            'index': tf.FixedLenFeature([], dtype=tf.int64,  default_value=-1),
        }
        features = tf.parse_single_example(example_serialized, feature_map)

        image_size = 32
        img = tf.reshape(tf.decode_raw(features['image'], tf.uint8),
                         [image_size, image_size, 3])

        data = {
            'image': img,
            'label': features['label'],
            'clean': features['clean'],
            'index': features['index'],
        }
        return data

    def preprocess_example(self, example, is_training, thread_id=0):
        data = super(NoisyCifarInputPipeline, self).preprocess_example(
            example, is_training, thread_id=thread_id)
        data['clean'] = example['clean']
        data['index'] = example['index']
        return data
