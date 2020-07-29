import tensorflow as tf
import math
import tensorflow.contrib.slim as slim
from Attention_NetVLAD.pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_sa_module_msg
import Attention_NetVLAD.pcan_tf_util as tf_util

def pixel_attention(feature_map, weight_decay=0.00004, reuse=None):
	_, H, W, C = tuple([int(x) for x in feature_map.get_shape()])
	w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
												dtype=tf.float32,
												initializer=tf.initializers.orthogonal,
												regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
	b_s = tf.get_variable("SpatialAttention_b_s", [1],
												dtype=tf.float32,
												initializer=tf.initializers.zeros)
	spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) + b_s
	spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))
	attention = tf.reshape(spatial_attention_fm, [-1, H, W, 1])
	#attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
	#attended_fm = attention * feature_map
	return attention

def point_attention(xyz,reshaped_input,is_training,bn_decay=None):
	print(reshaped_input)
	input = tf.reshape(reshaped_input, [-1,4096, 1024])
	print(input)
	
	#msg grouping
	l1_xyz, l1_points = pointnet_sa_module_msg(xyz, input, 256, [0.1, 0.2, 0.4], [16, 32, 64],
                                               [[16, 16, 32], [32, 32, 64], [32, 64, 64]], is_training,
                                               bn_decay,
                                               scope='layer1', use_nchw=True)
	l2_xyz, l2_points, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None,
                                              mlp=[256, 512], mlp2=None, group_all=True, is_training=is_training,
                                              bn_decay=bn_decay, scope='layer3')
	
	print('l2_points:', l2_points)
	l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
	l0_points = pointnet_fp_module(xyz, l1_xyz, tf.concat([xyz,input],axis=-1), l1_points, [128,128], is_training, bn_decay, scope='fa_layer3')
	print('l0_points shape', l0_points)
	

	net = tf_util.conv1d(l0_points, 1, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
	
	net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
	net = tf_util.conv1d(net, 1, 1, padding='VALID', activation_fn=None, scope='fc2')
	
	m = tf.reshape(net, [-1, 1])
	print('m:', m)
	
	#constrain weights to [0, 1]
	m = tf.nn.sigmoid(m)
	print(m)
	return m




def Att_NetVLAD_forward(weights, reshaped_input, feature_size=1024, max_samples=4096, cluster_size=64,
                output_dim=256, gating=True, add_batch_norm=True,
                is_training=True, bn_decay=None):
    """Forward pass of a NetVLAD block.

    Args:
    reshaped_input: If your input is in that form:
    'batch_size' x 'max_samples' x 'feature_size'
    It should be reshaped in the following form:
    'batch_size*max_samples' x 'feature_size'
    by performing:
    reshaped_input = tf.reshape(input, [-1, features_size])

    Returns:
    vlad: the pooled vector of size: 'batch_size' x 'output_dim'
    """
    
    m = weights
    
    print(weights)
    m = tf.tile(m, [1, cluster_size])
    print(m)

    print('m:', m)

    cluster_weights = tf.get_variable("cluster_weights",
                                      [feature_size, cluster_size],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(feature_size)))

    activation = tf.matmul(reshaped_input, cluster_weights)

    # activation = tf.contrib.layers.batch_norm(activation,
    #         center=True, scale=True,
    #         is_training=self.is_training,
    #         scope='cluster_bn')

    # activation = slim.batch_norm(
    #       activation,
    #       center=True,
    #       scale=True,
    #       is_training=self.is_training,
    #       scope="cluster_bn")

    if add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="cluster_bn", fused=False)
    else:
        cluster_biases = tf.get_variable("cluster_biases",
                                         [cluster_size],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(feature_size)))
        activation = activation + cluster_biases

    activation = tf.nn.softmax(activation)

    activation_crn = tf.multiply(activation, m)

    activation = tf.reshape(activation_crn,
                            [-1, max_samples, cluster_size])

    a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
                                       [1, feature_size, cluster_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(feature_size)))

    a = tf.multiply(a_sum, cluster_weights2)

    activation = tf.transpose(activation, perm=[0, 2, 1])

    reshaped_input = tf.reshape(reshaped_input, [-1,
                                                 max_samples, feature_size])

    vlad = tf.matmul(activation, reshaped_input)
    vlad = tf.transpose(vlad, perm=[0, 2, 1])
    vlad = tf.subtract(vlad, a)

    vlad = tf.nn.l2_normalize(vlad, 1)

    vlad = tf.reshape(vlad, [-1, cluster_size * feature_size])
    vlad = tf.nn.l2_normalize(vlad, 1)

    hidden1_weights = tf.get_variable("hidden1_weights",
                                      [cluster_size * feature_size, output_dim],
                                      initializer=tf.random_normal_initializer(
                                          stddev=1 / math.sqrt(cluster_size)))

    ##Tried using dropout
    # vlad=tf.layers.dropout(vlad,rate=0.5,training=self.is_training)

    vlad = tf.matmul(vlad, hidden1_weights)

    ##Added a batch norm
    vlad = tf.contrib.layers.batch_norm(vlad,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope='bn')

    if gating:
        vlad = context_gating(vlad, add_batch_norm, is_training)

    return vlad, weights

def context_gating(input_layer, add_batch_norm=True, is_training=True):
    """Context Gating

    Args:
    input_layer: Input layer in the following shape:
    'batch_size' x 'number_of_activation'

    Returns:
    activation: gated layer in the following shape:
    'batch_size' x 'number_of_activation'
    """

    input_dim = input_layer.get_shape().as_list()[1]

    gating_weights = tf.get_variable("gating_weights",
                                     [input_dim, input_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(input_dim)))

    gates = tf.matmul(input_layer, gating_weights)

    if add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=is_training,
            scope="gating_bn")
    else:
        gating_biases = tf.get_variable("gating_biases",
                                        [input_dim],
                                        initializer=tf.random_normal(stddev=1 / math.sqrt(input_dim)))
        gates = gates + gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(input_layer, gates)

    return activation
