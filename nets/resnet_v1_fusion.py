import tensorflow as tf
from nets_v3.resnet_v1 import resnet_v1_fusion, resnet_arg_scope

def endpoints(pcai, is_training):
	
    with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
    	_, endpoints = resnet_v1_fusion(pcai, num_classes=None, is_training=is_training, global_pool=False)  

    #print(endpoints['fusion_var/resnet_v1_fusion/block1'])
    
    endpoints['final_pooling'] = tf.reduce_mean(endpoints['fusion_var/resnet_v1_fusion/block1'], [1, 2], name='pool5')
    endpoints['model_output'] = tf.layers.dense(endpoints['final_pooling'],1024)
    return endpoints, 'resnet_v1_50'

