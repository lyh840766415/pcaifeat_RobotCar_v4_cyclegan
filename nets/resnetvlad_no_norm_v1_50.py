import tensorflow as tf
from nets_v3.resnet_v1 import resnet_v1_50, resnet_arg_scope
import pointnetvlad_v3.loupe_no_norm as lp

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image, is_training):
    if image.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, 3]')

    #image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))

    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
        _, endpoints = resnet_v1_50(image, num_classes=None, is_training=is_training, global_pool=True)
    
    resnet_out_shape = endpoints['img_var/resnet_v1_50/block4'].get_shape()
    num_feature = resnet_out_shape[1]*resnet_out_shape[2]
    NetVLAD = lp.NetVLAD(feature_size=2048, max_samples=num_feature, cluster_size=16, output_dim=1000, gating=True, add_batch_norm=True, is_training=is_training)
    
    endpoints['vlad_in'] = tf.reshape(endpoints['img_var/resnet_v1_50/block4'],[-1,2048])
    endpoints['vlad_out'] = NetVLAD.forward(endpoints['vlad_in'])
    
    return endpoints['vlad_out'], 'resnet_v1_50'
