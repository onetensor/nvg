from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_slim as slim


tf.compat.v1.disable_eager_execution()


def lrelu(value: tf.Tensor) -> tf.Tensor:
    return tf.maximum(value * 0.2, value)


def upsample_and_concat(x1: tf.Tensor, x2: tf.Tensor, output_channels: int, input_channels: int) -> tf.Tensor:
    kernel = tf.Variable(tf.random.truncated_normal([2, 2, output_channels, input_channels], stddev=0.02))
    deconvolved = tf.nn.conv2d_transpose(x1, kernel, tf.shape(x2), strides=[1, 2, 2, 1])
    result = tf.concat([deconvolved, x2], 3)
    result.set_shape([None, None, None, output_channels * 2])
    return result


def network(inputs: tf.Tensor) -> tf.Tensor:
    conv1 = slim.conv2d(inputs, 32, [3, 3], activation_fn=lrelu, scope="g_conv1_1")
    conv1 = slim.conv2d(conv1, 32, [3, 3], activation_fn=lrelu, scope="g_conv1_2")
    conv2 = slim.conv2d(slim.max_pool2d(conv1, [2, 2], padding="SAME"), 64, [3, 3], activation_fn=lrelu, scope="g_conv2_1")
    conv2 = slim.conv2d(conv2, 64, [3, 3], activation_fn=lrelu, scope="g_conv2_2")
    conv3 = slim.conv2d(slim.max_pool2d(conv2, [2, 2], padding="SAME"), 128, [3, 3], activation_fn=lrelu, scope="g_conv3_1")
    conv3 = slim.conv2d(conv3, 128, [3, 3], activation_fn=lrelu, scope="g_conv3_2")
    conv4 = slim.conv2d(slim.max_pool2d(conv3, [2, 2], padding="SAME"), 256, [3, 3], activation_fn=lrelu, scope="g_conv4_1")
    conv4 = slim.conv2d(conv4, 256, [3, 3], activation_fn=lrelu, scope="g_conv4_2")
    conv5 = slim.conv2d(slim.max_pool2d(conv4, [2, 2], padding="SAME"), 512, [3, 3], activation_fn=lrelu, scope="g_conv5_1")
    conv5 = slim.conv2d(conv5, 512, [3, 3], activation_fn=lrelu, scope="g_conv5_2")
    conv6 = slim.conv2d(upsample_and_concat(conv5, conv4, 256, 512), 256, [3, 3], activation_fn=lrelu, scope="g_conv6_1")
    conv6 = slim.conv2d(conv6, 256, [3, 3], activation_fn=lrelu, scope="g_conv6_2")
    conv7 = slim.conv2d(upsample_and_concat(conv6, conv3, 128, 256), 128, [3, 3], activation_fn=lrelu, scope="g_conv7_1")
    conv7 = slim.conv2d(conv7, 128, [3, 3], activation_fn=lrelu, scope="g_conv7_2")
    conv8 = slim.conv2d(upsample_and_concat(conv7, conv2, 64, 128), 64, [3, 3], activation_fn=lrelu, scope="g_conv8_1")
    conv8 = slim.conv2d(conv8, 64, [3, 3], activation_fn=lrelu, scope="g_conv8_2")
    conv9 = slim.conv2d(upsample_and_concat(conv8, conv1, 32, 64), 32, [3, 3], activation_fn=lrelu, scope="g_conv9_1")
    conv9 = slim.conv2d(conv9, 32, [3, 3], activation_fn=lrelu, scope="g_conv9_2")
    return tf.nn.depth_to_space(slim.conv2d(conv9, 12, [1, 1], activation_fn=None, scope="g_conv10"), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ containing NHWC key 'input'")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sample = np.load(args.input)["input"].astype(np.float32)
    inputs = tf.compat.v1.placeholder(tf.float32, [None, None, None, 4], name="input")
    output = network(inputs)
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as session:
        saver.restore(session, args.checkpoint)
        reference = session.run(output, feed_dict={inputs: sample})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, input=sample, output=reference)


if __name__ == "__main__":
    main()
