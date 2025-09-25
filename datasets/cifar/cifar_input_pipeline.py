# CIFAR input pipeline.
from __future__ import absolute_import, division, print_function

# Dùng TF1-compat để có FixedLenFeature/parse_single_example/...
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from datasets.base.input_pipeline import InputPipeline
from datasets.data_factory import RegisterInputPipeline
from utils import logger

log = logger.get()


@RegisterInputPipeline('cifar')
class CifarInputPipeline(InputPipeline):
    """CIFAR input pipeline."""

    def parse_example_proto(self, example_serialized):
        feature_map = {
            'image': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'label': tf.FixedLenFeature([], dtype=tf.int64,  default_value=-1),
        }
        features = tf.parse_single_example(example_serialized, feature_map)
        image_size = 32
        return {
            'image': tf.reshape(tf.decode_raw(features['image'], tf.uint8),
                                [image_size, image_size, 3]),
            'label': features['label'],
        }

    def distort_image(self, image):
        """Applies random distortion on the image for training."""
        image_size = 32
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        log.info("Apply random cropping")
        # dùng API mới tương thích TF1/TF2
        image = tf.image.resize_with_crop_or_pad(image, image_size + 4, image_size + 4)
        image = tf.image.random_crop(image, [image_size, image_size, 3])
        log.info("Apply random flipping")
        image = tf.image.random_flip_left_right(image)
        return image

    def eval_image(self, image):
        """Prepares the image for testing."""
        return tf.image.convert_image_dtype(image, dtype=tf.float32)

    def preprocess_example(self, example, is_training, thread_id=0):
        image = example['image']
        if is_training:
            image = self.distort_image(image)
        else:
            image = self.eval_image(image)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return {'image': image, 'label': example['label']}
