import tensorflow as tf
import cfg

class mcnn_model(object):
    def __init__(self):
        self.batch_size = cfg.batch_size
        self.test_x = cfg.size_x
        self.test_y = cfg.size_y
        self.train_x = cfg.train_size_x
        self.train_y = cfg.train_size_y
    
    
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
            
            
    def model(self, x_input):
        with tf.name_scope('column1'):
            c1_conv1 = self.conv(x_input, [5,5,3,24], 'conv1')
            c1_pool1 = tf.nn.max_pool(c1_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            c1_conv2 = self.conv(c1_pool1, [3,3,24,48], 'conv2')
            c1_pool2 = tf.nn.max_pool(c1_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            c1_conv3 = self.conv(c1_pool2, [3,3,48,24], 'conv3')
            c1_conv4 = self.conv(c1_conv3, [3,3,24,12], 'conv4')
        with tf.name_scope('column2'):
            c2_conv1 = self.conv(x_input, [7,7,3,20], 'conv1')
            c2_pool1 = tf.nn.max_pool(c2_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            c2_conv2 = self.conv(c2_pool1, [5,5,20,40], 'conv2')
            c2_pool2 = tf.nn.max_pool(c2_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            c2_conv3 = self.conv(c2_pool2, [5,5,40,20], 'conv3')
            c2_conv4 = self.conv(c2_conv3, [5,5,20,10], 'conv4')
        with tf.name_scope('column3'):
            c3_conv1 = self.conv(x_input, [9,9,3,16], 'conv1')
            c3_pool1 = tf.nn.max_pool(c3_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            c3_conv2 = self.conv(c3_pool1, [7,7,16,32], 'conv2')
            c3_pool2 = tf.nn.max_pool(c3_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            c3_conv3 = self.conv(c3_pool2, [7,7,32,16], 'conv3')
            c3_conv4 = self.conv(c3_conv3, [7,7,16,8], 'conv4')
        with tf.name_scope('merge'):
            m_node = []
            m_node.append(c1_conv4)
            m_node.append(c2_conv4)
            m_node.append(c3_conv4)
            merge = tf.concat(m_node, 3)
            
            merge_conv = self.conv(merge, [1,1,30,1], 'conv')
            
        return merge_conv
        
    def loss_layer(self, pred, gt, stage='train'):
        if stage=='test':
            ground = tf.reshape(gt, [self.test_x//4, self.test_y//4, 1])
        else:
            ground = tf.reshape(gt, [-1, self.train_x//4, self.train_y//4, 1])
        loss = tf.reduce_sum(tf.square(tf.subtract(pred, ground)))
        return loss
        
        
