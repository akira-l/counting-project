import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages 
from tensorflow.python.training.moving_averages import assign_moving_average


import cfg

class resnet_mcnn(object):
    def __init__(self):
        self.batch_size = cfg.batch_size
        self.test_x = cfg.size_x
        self.test_y = cfg.size_y
        self.train_x = cfg.train_size_x
        self.train_y = cfg.train_size_y
        self.is_training = True
        self.BN_DECAY = 0.9997
        self.UPDATE_OPS_COLLECTION = 'resnet_update_ops'
        self.BN_EPSILON = 0.001
        self.RESNET_VARIABLES = 'resnet_variables'
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1, name="weight")
        return tf.Variable(initial)
        
    def biases_variable(self, shape):
        initial = tf.constant(0.1, shape=shape,name="bais")
        return tf.Variable(initial)
        
    def prelu(self, x_):
        alpha = tf.get_variable('alpha', x_.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype=tf.float32)
        pos = tf.nn.relu(x_)
        neg = alpha * (x_ - abs(x_))
        return pos+neg
        
    def conv(self, x_input, shape ,name):
        with tf.variable_scope(name):
            weight = self.weight_variable(shape)
            bias = self.weight_variable([shape[-1]])
            result = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, weight, strides=[1,1,1,1], padding='SAME'), bias))
            return result
    
    def conv_p(self, x_input, shape, name):
        with tf.variable_scope(name):
            weight = self.weight_variable(shape)
            bias = self.weight_variable([shape[-1]])
            result = self.prelu(tf.nn.bias_add(tf.nn.conv2d(x_input, weight, strides=[1,1,1,1], padding='SAME'), bias))
            return result
        
    def fc_layer(self, in_feature, out_shape, name):
        with tf.variable_scope(name+"_fc"):
            in_shape = 1
            for d in in_feature.get_shape().as_list()[1:]:
                in_shape *= d
            weight = tf.get_variable(initializer=tf.truncated_normal([in_shape, out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_weight')
            bias = tf.get_variable(initializer=tf.truncated_normal([out_shape], 0, 1), dtype=tf.float32, name=name+'_fc_bias')
            result = tf.reshape(in_feature, [-1, in_shape])
            result = tf.nn.xw_plus_b(result, weight, bias, name=name+'_fc_do')
            return result
            
    def batch_norm_layer(self, x, is_training, name):
        with tf.variable_scope(name):
            beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
            axises = np.arange(len(x.shape) - 1)
            batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
            def mean_var_with_update(self):
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(is_training, mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
    




    def batch_norm(self, x, is_training, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = x.get_shape().as_list()[-1:]#tf.shape(x)[-1:]
            moving_mean = tf.get_variable('mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
            moving_variance = tf.get_variable('variance', params_shape, initializer=tf.ones_initializer, trainable=False)
            def mean_var_with_update():
                #mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
                mean, variance = tf.nn.moments(x, x.get_shape().as_list()[-1:], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay), assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)
            mean, variance = tf.cond(tf.cast(is_training, tf.bool), mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x
    
    
    def Batch_Normalization(self, x, is_training, scope):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
            return tf.cond(is_training,
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=is_training, reuse=True))
    
    
    def bn(self, x, is_training):
        x_shape = x.get_shape()  
        params_shape = x_shape[-1:]  
        axis = list(range(len(x_shape) - 1))  
        beta = self._get_variable('beta', params_shape, initializer=tf.zeros_initializer())  
        gamma = self._get_variable('gamma', params_shape, initializer=tf.ones_initializer())  
        moving_mean = self._get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
        moving_variance = self._get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
        # These ops will only be preformed when training.  
        mean, variance = tf.nn.moments(x, axis)  
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.BN_DECAY)  
        update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, self.BN_DECAY)  
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_mean)  
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_variance)  
        mean, variance = control_flow_ops.cond(  
            is_training, lambda: (mean, variance),  
            lambda: (moving_mean, moving_variance))  
        return tf.nn.batch_normalization(x, mean, variance, beta, gamma, self.BN_EPSILON) 
    
    
    
    def _get_variable(self,name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.VARIABLES, self.RESNET_VARIABLES]
        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)
    
    
    
            
    def model(self, x_input1, x_input2, x_input3):
        with tf.name_scope('resmcnn1'):
            c1_conv1 = self.conv(x_input1, [7, 7, 64, 32], 'conv1')
            c1_pool1 = tf.nn.max_pool(c1_conv1, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name='pool1')
            c1_conv2 = self.conv(c1_pool1, [7,7,32,8],'conv2')
            c1_conv3 = self.conv(c1_conv2, [5,5,8,1], 'conv3')
        with tf.name_scope('resmcnn2'):
            c2_conv1 = self.conv(x_input2, [3,3,256,64], 'conv1')
            c2_pool1 = tf.nn.max_pool(c2_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            c2_conv2 = self.conv(c2_pool1, [5,5,64,8], 'conv2')
            c2_conv3 = self.conv(c2_conv2, [5,5,8,1], 'conv3')
        with tf.name_scope('resmcnn3'):
            c3_conv1 = self.conv(x_input3, [3,3,512,64], 'conv1')
            c3_conv2 = self.conv(c3_conv1, [5,5,64,8], 'conv2')
            c3_conv3 = self.conv(c3_conv2, [5,5,8,1], 'conv3')
            
        with tf.name_scope('merge'):
            m_node = []
            m_node.append(c1_conv3)
            m_node.append(c2_conv3)
            m_node.append(c3_conv3)
            merge = tf.concat(m_node, 3)
            merge_conv1 = self.conv(merge, [3,3,3,3], 'conv1')
            merge_conv2 = self.conv(merge_conv1, [5,5,3,1], 'conv2')
            #merge_out = self.batch_norm(merge_conv2, is_training=True)
            merge_out = tf.layers.batch_normalization(merge_conv2, training=True)
        return merge_out
        
        
    def loss_layer(self, pred, gt, stage='train'):
        if stage=='test':
            ground = tf.reshape(gt, [self.test_x//8, self.test_y//8, 1])
        else:
            ground = tf.reshape(gt, [-1, self.train_x//8, self.train_y//8, 1])
        loss = 10000*tf.reduce_sum(tf.square(tf.subtract(pred, ground)))
        return loss
        
        
        
        
