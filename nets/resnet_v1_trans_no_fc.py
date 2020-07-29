import tensorflow as tf

from nets_v3.resnet_v1 import resnet_v1_50, resnet_arg_scope

_RGB_MEAN = [123.68, 116.78, 103.94]

def endpoints(image,pc_trans_feat,is_training):
	if image.get_shape().ndims != 4:
		raise ValueError('Input must be of size [batch, height, width, 3]')
	
	image = image - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1,1,1,3))
	
	with tf.contrib.slim.arg_scope(resnet_arg_scope(batch_norm_decay=0.9, weight_decay=0.0)):
		_, endpoints = resnet_v1_50(image, num_classes=None, is_training=is_training, global_pool=True)
	
	if not pc_trans_feat is None:
		concat_feat = tf.concat((endpoints['img_var/resnet_v1_50/block4'],pc_trans_feat),axis=3)
		conv_feat = tf.layers.conv2d(concat_feat,2048,3,activation=tf.nn.relu)
		img_pc_feat = tf.reduce_mean(conv_feat, [1, 2], name='pool6')
		img_pc_feat = tf.nn.l2_normalize(img_pc_feat,1)
	else:
		img_pc_feat = None
	
	img_feat = tf.reduce_mean(endpoints['img_var/resnet_v1_50/block4'], [1, 2], name='pool5')
	img_feat = tf.nn.l2_normalize(img_feat,1)
	
	
	return img_feat,img_pc_feat 
