
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
   

def transition_layer(inputs, factor, theta, scope=None):
   
    with tf.variable_scope(scope):
        if factor == 1:
            return inputs
        else:
            current_depth = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            net = slim.batch_norm(inputs, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, theta * current_depth, [1, 1], scope='conv1x1',
                              activation_fn=None, normalizer_fn=None)

            net = slim.avg_pool2d(net, [2, 2], scope='avg_pool', stride=factor)
            return net


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None,
                       outputs_collections=None):
  
    current_stride = 1

    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
               
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, i, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, i, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)

            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def densenet_arg_scope(weight_decay=0.0001,
                       batch_norm_decay=0.997,
                       batch_norm_epsilon=1e-5,
                       batch_norm_scale=True,
                       activation_fn=tf.nn.relu,
                       use_batch_norm=True):
    
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm if use_batch_norm else None,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.avg_pool2d, slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
