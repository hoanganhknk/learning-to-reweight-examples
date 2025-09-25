# Modifications Copyright (c) 2019 Uber Technologies, Inc.
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Converts CIFAR datasets to TFRecord format and manually add flip-label noise.
# WARNING: the generated data is corrupted. Use it only for noisy label problem research.
#
# Output TFRecord in the following location:
# [output folder]/train-00000-of-00005
# ...
# [output folder]/validation-00000-of-00001
#
# Usage:
# python generate_noisy_cifar_data.py --dataset cifar-10 --data_folder /path/to/raw
#   --seed 0 --noise_ratio 0.4 --num_clean 100 --num_val 5000
#
from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import six
import pickle as pkl
import tensorflow as tf  # TF2.x ok for tf.train.Feature & tf.io.TFRecordWriter

# Use tf.io.gfile (with fallback) for all file I/O
try:
    from tensorflow.io import gfile  # TF2
except Exception:
    from tensorflow.compat.v1 import gfile  # fallback

# -------------------------
# TFRecord helper features
# -------------------------
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(image, label, clean, idx):
    """Convert one sample to tf.train.Example."""
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'clean': _int64_feature(clean),
        'index': _int64_feature(idx),
    }))
    return example

# -------------------------
# Pickle (Py2 -> Py3) helpers
# -------------------------
def _load_pickle(file_path):
    """Load CIFAR pickle robustly on Py3 (decode Py2 pickles)."""
    with gfile.GFile(file_path, 'rb') as f:
        try:
            # Prefer decoding bytes -> str keys
            return pkl.load(f, encoding='latin1')
        except TypeError:
            # Fallback if encoding arg unsupported
            return pkl.load(f)

def _get_key(d, name):
    """Access key whether stored as str or bytes."""
    if name in d:
        return d[name]
    bname = name.encode('utf-8')
    return d[bname] if bname in d else None

# -------------------------
# Utilities
# -------------------------
def _split(num, seed, partitions):
    """Randomly split indices into partitions with sizes given by `partitions`."""
    all_idx = np.arange(num)
    rnd = np.random.RandomState(seed)
    rnd.shuffle(all_idx)
    siz = 0
    results = []
    for pp in partitions:
        results.append(all_idx[siz:siz + pp])
        siz += pp
    return results

def serialize_to_tf_record(basename, num_shard, images, labels, mask=None):
    """Serialize to sharded TFRecords using TF2 writer."""
    output_filename = basename + '-{:05d}-of-{:05d}'
    num_example = images.shape[0]
    num_example_per_shard = int(np.ceil(num_example / float(num_shard)))
    for ii in six.moves.xrange(num_shard):
        _filename = output_filename.format(ii, num_shard)
        # TF2: use tf.io.TFRecordWriter (tf.python_io.* removed)
        with tf.io.TFRecordWriter(_filename) as writer:
            start = num_example_per_shard * ii
            end = min(num_example_per_shard * (ii + 1), num_example)
            for jj in six.moves.xrange(start, end):
                _mask = 1 if mask is None else int(mask[jj])
                _example = _convert_to_example(images[jj].tobytes(), int(labels[jj]), _mask, int(jj))
                writer.write(_example.SerializeToString())
            # Context manager closes properly in TF2.

# -------------------------
# CIFAR readers
# -------------------------
def read_cifar_10(data_folder):
    """Read CIFAR-10 (python) robustly for Py3."""
    train_file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_file_list = ['test_batch']

    for file_list, name in zip([train_file_list, test_file_list], ['train', 'validation']):
        img_list, label_list = [], []
        for fname in file_list:
            batch = _load_pickle(os.path.join(data_folder, 'cifar-10-batches-py', fname))
            _img = _get_key(batch, 'data')
            _lbl = _get_key(batch, 'labels')
            if _img is None or _lbl is None:
                raise KeyError("CIFAR-10 pickle missing 'data'/'labels' keys (str/bytes).")
            _img = np.asarray(_img, dtype=np.uint8).reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            img_list.append(_img)
            label_list.append(np.asarray(_lbl, dtype=np.int64))
        img = np.concatenate(img_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        if name == 'train':
            train_img, train_label = img, label
        else:
            test_img, test_label = img, label
    return train_img, train_label, test_img, test_label

def read_cifar_100(data_folder):
    """Read CIFAR-100 (python) robustly for Py3 (handles fine_labels)."""
    base = os.path.join(data_folder, 'cifar-100-python')
    train_dict = _load_pickle(os.path.join(base, 'train'))
    test_dict  = _load_pickle(os.path.join(base, 'test'))

    train_data   = _get_key(train_dict, 'data')
    train_labels = _get_key(train_dict, 'fine_labels') or _get_key(train_dict, 'labels')
    test_data    = _get_key(test_dict,  'data')
    test_labels  = _get_key(test_dict,  'fine_labels') or _get_key(test_dict, 'labels')

    if train_data is None or test_data is None:
        raise KeyError("CIFAR-100 pickle missing 'data' key (str/bytes).")
    if train_labels is None or test_labels is None:
        raise KeyError("CIFAR-100 pickle missing 'fine_labels'/'labels' keys (str/bytes).")

    train_img = np.asarray(train_data, dtype=np.uint8).reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
    test_img  = np.asarray(test_data,  dtype=np.uint8).reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
    train_label = np.asarray(train_labels, dtype=np.int64)
    test_label  = np.asarray(test_labels,  dtype=np.int64)

    return train_img, train_label, test_img, test_label

# -------------------------
# Dataset splits & noise
# -------------------------
def trainval_split(img, label, num_val, seed):
    """Split training set into train/val."""
    assert img.shape[0] == label.shape[0], 'Images and labels dimension must match.'
    num = img.shape[0]
    trainval_partition = [num - num_val, num_val]
    idx = _split(num, seed, trainval_partition)
    return img[idx[0]], label[idx[0]], img[idx[1]], label[idx[1]]

def _flip_data(img, label, noise_ratio, num_classes, seed):
    """Uniformly reassign first K labels (K = noise_ratio * N) to random classes."""
    num = len(label)
    assert len(img) == len(label)
    num_noise = int(num * noise_ratio)
    rnd = np.random.RandomState(seed + 1)

    # Randomly re-assign labels (keep original semantics: not guaranteeing flip)
    new_label = np.floor(rnd.uniform(0, num_classes - 1, size=[num_noise])).astype(np.int64)
    new_noise_label_ = new_label

    label = np.concatenate([new_noise_label_, label[num_noise:]])
    noise_mask = np.concatenate([np.zeros([num_noise]), np.ones([num - num_noise])]).astype(np.int64)

    # Shuffle to mix noisy/clean
    idx = np.arange(num)
    rnd.shuffle(idx)
    return img[idx], label[idx], noise_mask[idx]

def _flip_data_background(img, label, noise_ratio, num_classes, seed):
    """Flip first K labels to the same random class (background noise style)."""
    num = len(label)
    assert len(img) == len(label)
    num_noise = int(num * noise_ratio)
    rnd = np.random.RandomState(seed + 1)

    new_label = int(np.floor(rnd.uniform(0, num_classes - 1)))
    print('Random new label:', new_label)
    new_noise_label_ = np.full([num_noise], new_label, dtype=np.int64)
    noise_mask0 = (new_label == label[:num_noise]).astype(np.int64)

    label = np.concatenate([new_noise_label_, label[num_noise:]])
    noise_mask = np.concatenate([noise_mask0, np.ones([num - num_noise])]).astype(np.int64)

    idx = np.arange(num)
    rnd.shuffle(idx)
    return img[idx], label[idx], noise_mask[idx]

def generate_data(img, label, noise_ratio, num_clean, num_classes, seed, background=False):
    """Generate noisy/clean split from training set."""
    num = img.shape[0]
    noise_img, noise_label, clean_img, clean_label = trainval_split(img, label, num_clean, seed)
    if background:
        noise_img, noise_label, noise_mask = _flip_data_background(
            noise_img, noise_label, noise_ratio, num_classes, seed)
    else:
        noise_img, noise_label, noise_mask = _flip_data(
            noise_img, noise_label, noise_ratio, num_classes, seed)
    return noise_img, noise_label, noise_mask, clean_img, clean_label

# -------------------------
# Main generator
# -------------------------
def generate_noisy_cifar(dataset,
                         data_folder,
                         num_val,
                         noise_ratio,
                         num_clean,
                         output_folder,
                         seed,
                         background=False):
    """Generate noisy CIFAR and write TFRecords."""
    # Read raw dataset
    if dataset == 'cifar-10':
        train_img, train_label, test_img, test_label = read_cifar_10(data_folder)
        num_classes = 10
    elif dataset == 'cifar-100':
        train_img, train_label, test_img, test_label = read_cifar_100(data_folder)
        num_classes = 100
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset))

    # Split train/val
    train_img, train_label, val_img, val_label = trainval_split(train_img, train_label, num_val, seed)

    # Generate noisy & clean partitions
    noise_img, noise_label, noise_mask, clean_img, clean_label = generate_data(
        train_img, train_label, noise_ratio, num_clean, num_classes, seed, background=background)

    # Ensure output folder (use gfile)
    if not gfile.exists(output_folder):
        gfile.makedirs(output_folder)

    # Write TFRecords
    serialize_to_tf_record(os.path.join(output_folder, 'train_noisy'), 4, noise_img, noise_label, noise_mask)
    serialize_to_tf_record(os.path.join(output_folder, 'train_clean'), 1, clean_img, clean_label)
    serialize_to_tf_record(os.path.join(output_folder, 'validation'), 1, val_img, val_label)

    # Noisy validation
    if background:
        noise_val_img, noise_val_label, noise_val_mask = _flip_data_background(
            val_img, val_label, noise_ratio, num_classes, seed)
    else:
        noise_val_img, noise_val_label, noise_val_mask = _flip_data(
            val_img, val_label, noise_ratio, num_classes, seed)
    serialize_to_tf_record(os.path.join(output_folder, 'validation_noisy'), 1,
                           noise_val_img, noise_val_label, noise_val_mask)

    # Test set
    serialize_to_tf_record(os.path.join(output_folder, 'test'), 1, test_img, test_label)

# -------------------------
# CLI (optional use)
# -------------------------
def _main():
    # Dùng argparse để tránh phụ thuộc tf.flags (đời cũ)
    import argparse
    parser = argparse.ArgumentParser(description="Generate noisy CIFAR TFRecords.")
    parser.add_argument('--noise_ratio', type=float, default=0.4)
    parser.add_argument('--num_clean', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_folder', type=str, default='./data/cifar-10',
                        help='Folder containing cifar-10-batches-py/ or cifar-100-python/')
    parser.add_argument('--dataset', type=str, default='cifar-10', choices=['cifar-10','cifar-100'])
    parser.add_argument('--output_folder', type=str, default='./data/cifar-10-noisy')
    parser.add_argument('--background', action='store_true', help='Use background label flipping')
    args = parser.parse_args()

    generate_noisy_cifar(args.dataset, args.data_folder, args.num_val, args.noise_ratio,
                         args.num_clean, args.output_folder, args.seed, background=args.background)

if __name__ == '__main__':
    _main()
