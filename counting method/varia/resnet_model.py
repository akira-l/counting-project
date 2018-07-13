import os
import tensorflow as tf
import numpy as np

import resnet_utils 

class resnet:
    def __init__(self):
        load_path = "/home/yuanzhi/figure/mexp/version2/resnet-pretrained/"
        self.xs = tf.placeholder(tf.float32, shape = [None, image_size_x, image_size_y, 3])
        
        self.ys = tf.placeholder(tf.float32, shape = [None, ground_size_x, ground_size_y])
        
        resnet_arg_scope = resnet_utils.resnet_arg_scope
        slim = tf.contrib.slim
        
        
    def build(self):
        net_x = resnet_v1_50(self.xs,is_training = False)
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        saver.restore(sess,"/home/yuanzhi/figure/mexp/version2/resnet_v1_101.ckpt")
        net_y = resnet_v1_50(self.ys,is_training = False, reuse = True)
        
        return net_x, net_y
        





@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
               
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
      depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth, [1, 1],
                stride=stride,
                activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')

        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def resnet_v1(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
 outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
              net = inputs
              if include_root_block:
                if output_stride is not None:
                  if output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiple of 4.')
                  output_stride /= 4
                net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
              net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
              # Convert end_points_collection into a dictionary of end_points.
              end_points = slim.utils.convert_collection_to_dict(end_points_collection)

              if global_pool:
                # Global average pooling.
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                end_points['global_pool'] = net
              if num_classes:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                end_points[sc.name + '/logits'] = net
                if spatial_squeeze:
                  net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                  end_points[sc.name + '/spatial_squeeze'] = net
                end_points['predictions'] = slim.softmax(net, scope='predictions')
              return net, end_points
resnet_v1.default_image_size = 224








def resnet_v1_block(scope, base_depth, num_units, stride):
  """Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
  """
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_101'):
  """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=23, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v1_101.default_image_size = resnet_v1.default_image_size


def resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_152'):
  """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v1_152.default_image_size = resnet_v1.default_image_size


def resnet_v1_200(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='resnet_v1_200'):
  """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
      resnet_v1_block('block2', base_depth=128, num_units=24, stride=2),
      resnet_v1_block('block3', base_depth=256, num_units=36, stride=2),
      resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
  ]
  return resnet_v1(inputs, blocks, num_classes, is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)
resnet_v1_200.default_image_size = resnet_v1.default_image_size

