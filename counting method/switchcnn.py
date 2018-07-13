import tensorflow as tf
import cfg
from tflearn.layers.conv import global_avg_pool

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils


class switchcnn(object):
    def __init__(self):
        self.vgg = vgg16.Vgg16()
        self.output_shape_1 = tf.placeholder(dtype=tf.int32, shape=[4])
        self.output_shape_2 = tf.placeholder(dtype=tf.int32, shape=[4])
        self.batch = cfg.batch_size
        self.sub_graph1 = tf.Graph()
        self.sub_graph2 = tf.Graph()
        self.sub_graph3 = tf.Graph()
        self.switch_graph = tf.Graph()
        
        
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
        
    def Global_Average_Pooling(self, x):
        return global_avg_pool(x, name='Global_avg_pooling')
        
        
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
            
    def model(self, x_input):
        with switch_graph.as_default():
            with tf.name_scope("switch_net"):
                self.vgg.build(x_input)
                gap = self.Global_Average_Pooling(self.vgg.conv5_3)
                
                switch_fc512 = self.fc_layer(gap, 521, 'switch_fc512')
                switch_fc3 = self.fc_layer(switch_fc512,3, 'switch_fc3')
                switch_prob = tf.nn.softmax(switch_fc3, name='prob')
                self.switch = switch_prob
            
        with sub_graph1.as_default():
            with tf.name_scope('sub_net1'):
                s1_conv1 = self.conv(x_input,[9,9,3,16],'s1_conv1'
                s1_pool1 = tf.nn.max_pool(s1_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s1_pool1')
                s1_conv2 = self.conv(s1_pool1, [7,7,16,32],'s1_conv2')
                s1_pool2 = tf.nn.max_pool(s1_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s1_pool2')
                s1_conv3 = self.conv(s1_pool2, [7,7,32,16], 's1_conv3')
                s1_conv4 = self.conv(s1_conv3, [7,7,16,8], name='s1_conv4')
                s1_conv5 = self.conv(s1_conv4, [1,1,8,1], name='s1_conv5')
                self.result1 = s1_conv5
        
        with sub_graph2.as_default():
            with tf.name_scope('sub_net2'):
                s2_conv1 = self.conv(x_input, [7,7,3,20], 's2_conv1')
                s2_pool1 = tf.nn.max_pool(s2_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s2_pool1')
                s2_conv2 = self.conv(s2_pool1, [5,5,20,40], 's2_conv2')
                s2_pool2 = tf.nn.max_pool(s2_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s2_pool2')
                s2_conv3 = self.conv(s2_pool2, [5,5,40,20], 's2_conv3')
                s2_conv4 = self.conv(s2_conv2, [5,5,20,10], 's2_conv4')
                s2_conv5 = self.conv(s2_conv4, [1,1,10,1], 's2_conv5')
                self.result2 = s2_conv5

        with sub_graph3.as_default():
            with tf.name_scope('sub_net3'):
                s3_conv1 = self.conv(x_input, [5,5,3,24], 's3_conv1')
                s3_pool1 = tf.nn.max_pool(s3_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s3_conv1')
                s3_conv2 = self.conv(s3_pool1, [3,3,24,48], 's3_conv2')
                s3_pool2 = tf.nn.max_pool(s3_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='s3_pool2')
                s3_conv3 = self.conv(s3_pool2, [3,3,48,24], 's3_conv3')
                s3_conv4 = self.conv(s3_conv3, [3,3,24,12], 's3_conv4')
                s3_conv5 = self.conv(s3_conv4, [1,1,12,1], 's3_conv5')
                self.result3 = s3_conv5
                
    def loss_layer(self, pred, gt):
        ground = tf.reshape(gt, [-1, 75, 75, 1])
        loss = tf.reduce_sum(tf.square(tf.subtract(pred, ground)))/90000
        
        return loss


