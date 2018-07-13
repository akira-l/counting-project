import tensorflow as tf
import cfg

import tensorflow_vgg.vgg16 as vgg16
import tensorflow_vgg.utils as utils

class cpcnn(object):
    def __init__(self, stage='train'):
        self.vgg = vgg16.Vgg16()
        self.output_shape_1 = tf.placeholder(dtype=tf.int32, shape=[4])
        self.output_shape_2 = tf.placeholder(dtype=tf.int32, shape=[4])
        self.batch = cfg.batch_size
        if stage=='test':
            self.batch = 1
        self.stage = stage
        self.test_x = 300#cfg.size_x
        self.test_y = 300#cfg.size_y
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
        with tf.name_scope("dme1"):
            h_conv1 = self.conv(x_input,[11,11,3,16],'conv1')
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            h_conv2 = self.conv(h_pool1,[9,9,16,24],'conv2')
            h_conv3 = self.conv(h_conv2, [7,7,24,16], 'conv3')
            h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            h_conv4 = self.conv(h_pool2, [7,7,16,16],'conv4')
            h_conv5 = self.conv(h_conv4,[11,11,16,8],'conv5')
            
        with tf.name_scope("dme2"):
            h2_conv1 = self.conv(x_input,[9,9,3,16],'conv1')
            h2_pool1 = tf.nn.max_pool(h2_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            h2_conv2 = self.conv(h2_pool1,[7,7,16,24],'conv2')
            h2_conv3 = self.conv(h2_conv2, [5,5,24,32], 'conv3')
            h2_pool2 = tf.nn.max_pool(h2_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            h2_conv4 = self.conv(h2_pool2, [5,5,32,32],'conv4')
            h2_conv5 = self.conv(h2_conv4,[3,3,32,16],'conv5')
            
        with tf.name_scope("dme3"):
            h3_conv1 = self.conv(x_input,[7,7,3,16],'conv1')
            h3_pool1 = tf.nn.max_pool(h3_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            h3_conv2 = self.conv(h3_pool1,[5,5,16,24],'conv2')
            h3_conv3 = self.conv(h3_conv2, [3,3,24,48], 'conv3')
            h3_pool2 = tf.nn.max_pool(h3_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            h3_conv4 = self.conv(h3_pool2, [3,3,48,48],'conv4')
            h3_conv5 = self.conv(h3_conv4,[3,3,48,24],'conv5')
            
        with tf.name_scope("dme"):
            node = []
            node.append(h_conv5)
            node.append(h2_conv5)
            node.append(h3_conv5)
            dme_out = tf.concat(node, 3)
            
        with tf.name_scope("lce"):
            lce_conv1 = self.conv(x_input, [5,5,3,8], 'conv1')
            lce_conv2 = self.conv(lce_conv1, [3,3,8,64], 'conv2')
            lce_pool1 = tf.nn.max_pool(lce_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
            lce_conv3 = self.conv(lce_pool1, [3,3,64,128],'conv3')
            lce_conv4 = self.conv(lce_conv3, [3,3,128,128],'conv4')
            lce_pool2 = tf.nn.max_pool(lce_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
            lce_conv5 = self.conv(lce_pool2, [3,3,128,64],'conv5')
            lce_conv6 = self.conv(lce_conv5, [3,3,64,16],'conv6')
            lce_pool3 = tf.nn.max_pool(lce_conv6, ksize=[1,4,4,1], strides=[1,2,2,1], padding='SAME', name='pool3')
            #lce_fc1024 = self.fc_layer(lce_pool3, 1024, 'lce-fc1024')
            lce_fc512 = self.fc_layer(lce_pool3, 128, 'lce-fc512')
            lce_fc5 = self.fc_layer(lce_fc512, 5, 'lce-fc5')
            print('lce_fc5',lce_fc5.get_shape().as_list())
            
            lce_re = tf.reshape(lce_fc5, [self.batch,1,1,5])
            #size_ = h_conv5.get_shape().as_list()
            if self.stage=='test':
                ones_init_lce = tf.ones([self.batch, self.test_x//4, self.test_y//4, 5], name='lce_ones')
            if self.stage=='train': 
                ones_init_lce = tf.ones([self.batch, self.train_x//4, self.train_y//4, 5], name='lce_ones')
            lce_out = lce_re*ones_init_lce
            print('lce_out', lce_out.get_shape().as_list())
            
            
        with tf.name_scope("content_vgg"):
            self.vgg.build(x_input)
        
        with tf.name_scope("gce"):
            fc512 = self.fc_layer(self.vgg.pool5, 256, 'gce-fc512')
            fc256 = self.fc_layer(fc512, 128, 'gce-fc256')
            gce_fc5 = self.fc_layer(fc256, 5, 'gce-fc5')
            print('gce_fc5', gce_fc5.get_shape().as_list())
            if self.stage=='test':
                ones_init_gce = tf.ones([self.batch, self.test_x//4, self.test_y//4, 5], name='gce_ones')
            if self.stage=='train':
                ones_init_gce = tf.ones([self.batch, self.train_x//4, self.train_y//4, 5], name='gce_ones')
            gce_re = tf.reshape(gce_fc5, [-1,1,1,5])
            #gce_out = gce_re*tf.ones([size_[0],size_[1],size_[2], 5])
            gce_out = gce_re*ones_init_gce
            print('gce_out', gce_out.get_shape().as_list())
        print("monitoring one_init shape:", self.stage)
            
        with tf.name_scope("fcnn"):
            fusion_node = []
            print('h_conv5', h_conv5.get_shape().as_list())
            print('h2_conv5', h2_conv5.get_shape().as_list())
            print('h3_conv5', h3_conv5.get_shape().as_list())
            print('lce_out', lce_out.get_shape().as_list())
            print('gce_out', gce_out.get_shape().as_list())

            fusion_node.append(h_conv5)
            fusion_node.append(h2_conv5)
            fusion_node.append(h3_conv5)
            fusion_node.append(lce_out)
            fusion_node.append(gce_out)
            fusion = tf.concat(fusion_node,3)

            print('fusion',fusion.get_shape().as_list())
            
            f_conv1 = self.conv(fusion, [9,9,58,64], 'conv1')
            f_conv2 = self.conv(f_conv1, [7,7,64,32], 'conv2')
            
            #output_shape_1 = tf.placeholder(dtype=tf.int32, shape=[4])
            W_trans1 = self.weight_variable([3,3,32,32])
            f_tconv1 = tf.nn.conv2d_transpose(f_conv2, filter=W_trans1, output_shape=self.output_shape_1, strides=[1,2,2,1],padding='SAME', name='hconv1')
            
            f_conv3 = self.conv(f_tconv1, [5,5,32,16],'conv3')
            
            #output_shape_2 = tf.placeholder(dtype=tf.int32, shape[4])
            W_trans2 = self.weight_variable([3,3,16,16])
            f_tconv2 = tf.nn.conv2d_transpose(f_conv3, filter=W_trans2, output_shape=self.output_shape_2, strides=[1,2,2,1], padding='SAME', name='hconv2')
            
            f_conv4 = self.conv(f_tconv2, [1,1,16,1], 'conv4')
            if self.stage=='train':
                f_out = tf.layers.batch_normalization(f_conv4, training=True)
            if self.stage == 'test':
                f_out = tf.layers.batch_normalization(f_conv4, training=False)
        
        with tf.name_scope("discriminator"):
            dis_node = []
            dis_node.append(h_conv5)
            dis_node.append(h2_conv5)
            dis_node.append(h3_conv5)
            dis_input = tf.concat(dis_node,3)
            
            dis_convp1 = self.conv_p(dis_input, [3,3,48,64], 'convp1')
            dis_convp2 = self.conv_p(dis_convp1, [3,3,64,128], 'convp2')
            dis_pool1 = tf.nn.max_pool(dis_convp2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='dis_pool1')
            dis_convp3 = self.conv_p(dis_pool1, [3,3,128,256], 'convp3')
            dis_pool2 = tf.nn.max_pool(dis_convp3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='dis_pool2')
            dis_convp4 = self.conv_p(dis_pool2, [3,3,256,256], 'convp4')
            dis_convp5 = self.conv_p(dis_convp4, [3,3,256,256], 'convp5')
            dis_pool3 = tf.nn.max_pool(dis_convp5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='dis_pool3')
            dis_conv6 = self.conv(dis_pool3, [3,3,256,1], 'dis_conv6')
            
            dis_out = tf.reduce_sum(tf.nn.sigmoid(dis_conv6))
        print('!!!!!load is done here!!!!!')
            
        return f_conv4, dis_out
            
    def loss_layer(self, pred, gt, dis_out):
        ground = tf.reshape(gt, [-1, 300, 300, 1])
        loss = tf.reduce_sum(tf.square(tf.subtract(pred, ground)))/90000 + dis_out
        
        return loss
            
            
