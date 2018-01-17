import nasnet
import tensorflow as tf
from nasnet import build_nasnet_base_my
import nasnet_utils

from tensorflow.python.framework import ops
arg_scope = tf.contrib.framework.arg_scope
slim = tf.contrib.slim


gradname = 'ResizeBicubicCustom'
@ops.RegisterGradient('{}grad'.format(gradname))
def frop_grad(op, grad):
    x = op.inputs[0]
    x_shape = x.get_shape().as_list()
    feat_map_size = tf.reduce_prod(x_shape[1:3])
    target_size = tf.reduce_prod(grad.get_shape().as_list()[1:3])
    gradient_factor = tf.cast(tf.divide(feat_map_size, target_size), tf.float32)
    y = tf.divide(tf.image.resize_bicubic(grad, x_shape[1:3]), gradient_factor)
    return y, None


def _large_imagenet_config(is_training=True):
    drop_path_keep_prob = 1.0 if not is_training else 0.7
    return tf.contrib.training.HParams(
        stem_multiplier=3.0,
        dense_dropout_keep_prob=0.5,
        num_cells=18,
        filter_scaling_rate=2.0,
        num_conv_filters=168,
        drop_path_keep_prob=drop_path_keep_prob,
        use_aux_head=1,
        num_reduction_layers=2,
        data_format='NHWC',
        skip_reduction_layer_input=1,
        total_training_steps=250000,
    )

def build_nasnet_large(images,
                       is_training=True,
                       final_endpoint=None):
    """Build NASNet Large model for the ImageNet Dataset."""
    hparams = _large_imagenet_config(is_training=is_training)


    if hparams.data_format == 'NCHW':
      images = tf.transpose(images, [0, 3, 1, 2])

    # Calculate the total number of cells in the network
    # Add 2 for the reduction cells
    total_num_cells = hparams.num_cells + 2
    # If ImageNet, then add an additional two for the stem cells
    total_num_cells += 2

    normal_cell = nasnet_utils.NasNetANormalCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)
    reduction_cell = nasnet_utils.NasNetAReductionCell(
        hparams.num_conv_filters, hparams.drop_path_keep_prob,
        total_num_cells, hparams.total_training_steps)
    with arg_scope([slim.dropout, nasnet_utils.drop_path, slim.batch_norm],
                   is_training=is_training):
        with arg_scope([slim.avg_pool2d,
                    slim.max_pool2d,
                    slim.conv2d,
                    slim.batch_norm,
                    slim.separable_conv2d,
                    nasnet_utils.factorized_reduction,
                    nasnet_utils.global_avg_pool,
                    nasnet_utils.get_channel_index,
                    nasnet_utils.get_channel_dim],
                   data_format=hparams.data_format):
            netout, end_points = build_nasnet_base_my(images,
                                normal_cell=normal_cell,
                                reduction_cell=reduction_cell,
                                hparams=hparams,
                                final_endpoint=final_endpoint)
            # dicts = end_points.keys()
            # for n in range(len(dicts)):
            #     print '%s:' % dicts[n], end_points[dicts[n]].shape.as_list()
            return netout, end_points, dict({4: end_points['Cell_17'], 3: end_points['Cell_11'],
                                             2: end_points['Cell_5'], 1: end_points['Stem_4']})



def rcu_block(x, channels, n_blocks=2, n_stages=2, is_training=True, stddev=0.015, keep_prob=1.0, name='', weight_decay=0.0005):
    """RCU block.

    Args:
      x : input
      channels : number of filters
      n_blocks : number of blocks
      n_stages : number of repetitions (relu x conv)
      stddev : stddev of a normal distribution for initialisation

    Returns:
      Result of RCU block.
    """
    stages_suffixes = {0 : '_conv',
                       1 : '_conv_relu_varout_dimred'}
    for i in xrange(n_blocks):
        top = tf.identity(x)
        for j in xrange(n_stages):
            top = tf.nn.relu(top)
            with tf.variable_scope('{}{}{}'.format(name, str(i + 1), stages_suffixes[j])):
                w = tf.get_variable('Conv/weights', shape=[3, 3, shape(top)[-1], channels],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                top = tf.nn.conv2d(tf.pad(top, make_pad(1)),
                                   w,
                                   strides=make_stride(1),
                                   padding='VALID')
                if j == 0:
                    b = tf.get_variable('Conv/biases', shape=[shape(top)[-1],],
                                        initializer=tf.zeros_initializer())
                    top = tf.nn.bias_add(top, b)
        x = top + x
    return x

def crp_block(x, channels, n_stages=2, stddev=0.015, keep_prob=1.0, name='', weight_decay=0.0005):
    """CRP block.

    Args:
      x : input
      channels : number of filters
      n_stages : number of repetitions (pool x conv)
      stddev : stddev of a normal distribution for initialisation.

    Returns:
      Result of CRP block.
    """
    top = tf.identity(x)
    for i in xrange(n_stages):
        top = tf.nn.max_pool(tf.pad(top, make_pad(2)),
                             ksize=make_stride(5),
                             strides=make_stride(1),
                             padding='VALID')
        with tf.variable_scope('{}{}_{}'.format(name, str(i + 1), 'outvar_dimred')):
            w = tf.get_variable('Conv/weights', shape=[3, 3, shape(top)[-1], channels],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            top = tf.nn.conv2d(tf.pad(top, make_pad(1)),
                                w,
                                strides=make_stride(1),
                                padding='VALID')
        x = top + x
    return x




def shape(x):
    st_shape = x.get_shape().as_list()
    dy_shape = tf.unstack(tf.shape(x))
    return [s[1] if s[0] is None else s[0] for s in zip(st_shape, dy_shape)]

def make_pad(p):
    return [[0, 0], [p, p], [p, p], [0, 0]]

def make_stride(s):
    return [1, s, s, 1]


def build_refine_net(nas_dict, num_classes, stddev=0.015,
                     keep_prob=1.0, weights_decay=0.0005):
    refinet_blocks = [4, 3, 2, 1]
    prev = None
    g = tf.get_default_graph()
    branches = []
    for order_idx, idx in enumerate(refinet_blocks):
        top = nas_dict[idx]
        top = tf.nn.relu(top)
        if idx == 4:
            channels = 512
            top = tf.nn.dropout(top, keep_prob=keep_prob)
        elif idx == 3:
            channels = 256
            top = tf.nn.dropout(top, keep_prob=keep_prob)
        else:
            channels = 256
        with tf.variable_scope('p_ims1d2_outl{}_dimred'.format(str(order_idx + 1))):
            w = tf.get_variable('Conv/weights', shape=[3, 3, shape(top)[-1], channels],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            top = tf.nn.conv2d(tf.pad(top, make_pad(1)),
                               w,
                               strides=make_stride(1),
                               padding='VALID')
        top = rcu_block(top, channels,
                         n_blocks=2, n_stages=2,
                         stddev=stddev,
                         name='adapt_stage{}_b'.format(str(order_idx + 1)))
        if prev is not None:
            H, W = tf.unstack(shape(top)[1:3])
            h, w = tf.unstack(shape(prev)[1:3])
            with tf.variable_scope('mflow_conv_g{}_b3_joint_varout_dimred'.format(str(order_idx))):
                w = tf.get_variable('Conv/weights', shape=[3, 3, shape(prev)[-1], channels],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                prev = tf.nn.conv2d(tf.pad(prev, make_pad(1)),
                                    w,
                                    strides=make_stride(1),
                                    padding='VALID')
            with tf.variable_scope('adapt_stage{}_b2_joint_varout_dimred'.format(str(order_idx + 1))):
                w = tf.get_variable('Conv/weights', shape=[3, 3, shape(top)[-1], channels],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                top = tf.nn.conv2d(tf.pad(top, make_pad(1)),
                                   w,
                                   strides=make_stride(1),
                                   padding='VALID')
            with g.gradient_override_map({'ResizeBicubic': '{}grad'.format(gradname)}):
                prev = tf.cond(tf.less(h, H),
                               lambda: tf.image.resize_bicubic(prev, shape(top)[1:3]),
                               lambda: prev)
            top = top + prev
        top = tf.nn.relu(top)
        top = crp_block(top, channels, n_stages=4, stddev=stddev,
                         name='mflow_conv_g{}_pool'.format(str(order_idx + 1)))
        prev = rcu_block(top, channels, n_blocks=3, n_stages=2,
                          stddev=stddev, name='mflow_conv_g{}_b'.format(str(order_idx + 1)))

        branches.append(top)

    prev = tf.nn.dropout(prev, keep_prob=keep_prob)
    with tf.variable_scope('clf_conv_{}'.format(str(num_classes))):
        w = tf.get_variable('Conv/weights', shape=[3, 3, shape(prev)[-1], num_classes],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('Conv/biases', shape=[num_classes, ],
                            initializer=tf.zeros_initializer())
        prev = tf.nn.conv2d(tf.pad(prev, make_pad(1)),
                            w,
                            strides=make_stride(1),
                            padding='VALID')
        prev = tf.nn.bias_add(prev, b)
    return prev, branches



class nas_refine_model(object):
    def __init__(self, num_cls, is_training, ohem=True, mining_ratio=0.3):
        self.num_cls = num_cls
        self.is_training = is_training
        self.ratio = 0.25
        if self.is_training:
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1

        self.ohem = ohem
        self.mining_ratio = mining_ratio

    def infer(self, inputs):
        images = inputs['images']
        with tf.variable_scope(tf.get_variable_scope()):
            with slim.arg_scope(nasnet.nasnet_large_arg_scope()):
                 _, _, nas_dict = build_nasnet_large(images, is_training=self.is_training)
            with tf.variable_scope('refinenet'):
                pred, _ = build_refine_net(nas_dict, num_classes=self.num_cls, keep_prob=self.keep_prob)

        return {'logits': pred}

    def loss_fn(self, infers, targets):
        logits = infers['logits']
        labels = tf.cast(
            targets['labels'],
            dtype=tf.int64
        )

        raw_prediction = tf.reshape(logits, [-1, self.num_cls])
        raw_gt = tf.reshape(labels, [-1, ])
        indices = tf.squeeze(tf.where(tf.greater_equal(raw_gt, 0)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)

        if self.ohem:
            keep_num = tf.cast(tf.maximum(tf.cast(tf.size(softmax_loss), dtype=tf.float32) * self.mining_ratio, 1),
                               dtype=tf.int32)
        else:
            keep_num = tf.cast(tf.size(softmax_loss), dtype=tf.int32)

        classification_loss = tf.reduce_mean(tf.nn.top_k(softmax_loss, keep_num)[0], name='cross_entropy_mean')
        tf.add_to_collection(
            name=tf.GraphKeys.LOSSES,
            value=classification_loss
        )

        return {'tower_loss': tf.add_n(
            inputs=tf.get_collection(
                key=tf.GraphKeys.LOSSES,
            )
        )}


    def eval_metric(self, infers, targets):

        raw_output = infers['logits']
        label_gt = targets['labels']
        raw_gt = tf.reshape(label_gt, [-1, ])
        indices = tf.squeeze(tf.where(tf.greater_equal(raw_gt, 0)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        raw_output_val = tf.argmax(raw_output, dimension=3)
        pred_val = tf.expand_dims(raw_output_val, dim=3)  # Create 4-d tensor.
        pred_val = tf.reshape(pred_val, [-1, ])
        pred_val = tf.gather(pred_val, indices)


        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_val, gt, num_classes=self.num_cls)
        return {'miou': mIoU, 'op': update_op}






