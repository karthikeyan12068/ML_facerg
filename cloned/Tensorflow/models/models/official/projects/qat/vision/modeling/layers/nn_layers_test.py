# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for nn_layers."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.projects.qat.vision.modeling.layers import nn_layers


class NNLayersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('deeplabv3plus', 1, 128, 128),
      ('deeplabv3plus', 2, 128, 128),
      ('deeplabv3', 1, 128, 64),
      ('deeplabv3', 2, 128, 64),
      ('deeplabv3plus_sum_to_merge', 1, 64, 128),
      ('deeplabv3plus_sum_to_merge', 2, 64, 128),
  )
  def test_segmentation_head_creation(self, feature_fusion, upsample_factor,
                                      low_level_num_filters, expected_shape):
    input_size = 128
    decoder_outupt_size = input_size // 2

    decoder_output = tf.random.uniform(
        (2, decoder_outupt_size, decoder_outupt_size, 64), dtype=tf.float32)
    backbone_output = tf.random.uniform((2, input_size, input_size, 32),
                                        dtype=tf.float32)
    segmentation_head = nn_layers.SegmentationHeadQuantized(
        num_classes=5,
        level=4,
        upsample_factor=upsample_factor,
        low_level=2,
        low_level_num_filters=low_level_num_filters,
        feature_fusion=feature_fusion)

    features = segmentation_head((backbone_output, decoder_output))

    self.assertAllEqual([
        2, expected_shape * upsample_factor, expected_shape * upsample_factor, 5
    ], features.shape.as_list())

  @parameterized.parameters(
      (None, []),
      (None, [6, 12, 18]),
      ([32, 32], [6, 12, 18]),
  )
  def test_spatial_pyramid_pooling_creation(self, pool_kernel_size,
                                            dilation_rates):
    inputs = tf.keras.Input(shape=(64, 64, 128), dtype=tf.float32)
    layer = nn_layers.SpatialPyramidPoolingQuantized(
        output_channels=256,
        dilation_rates=dilation_rates,
        pool_kernel_size=pool_kernel_size)
    output = layer(inputs)
    self.assertAllEqual([None, 64, 64, 256], output.shape)

  @parameterized.parameters(
      (3, [6, 12, 18, 24], 128),
      (3, [6, 12, 18], 128),
      (3, [6, 12], 256),
      (4, [], 128),
      (4, [6, 12, 18], 128),
      (4, [], 256),
  )
  def test_aspp_creation(self, level, dilation_rates, num_filters):
    input_size = 128 // 2**level
    tf.keras.backend.set_image_data_format('channels_last')
    endpoints = tf.random.uniform(
        shape=(2, input_size, input_size, 64), dtype=tf.float32)

    network = nn_layers.ASPPQuantized(
        level=level, dilation_rates=dilation_rates, num_filters=num_filters)

    feats = network(endpoints)

    self.assertAllEqual([2, input_size, input_size, num_filters],
                        feats.shape.as_list())


if __name__ == '__main__':
  tf.test.main()
