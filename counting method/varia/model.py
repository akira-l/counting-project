import tensorflow as tf
import config as cfg
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  

class Model_net(object):
    def __init__(self):
        self.BN_DECAY = 0.9997
        self.UPDATE_OPS_COLLECTION = 'resnet_update_ops'
        self.BN_EPSILON = 0.001
        self.RESNET_VARIABLES = 'resnet_variables'
        self.image_size_x = cfg.image_size_x
        self.image_size_y = cfg.image_size_y
        self.ground_size_x = cfg.ground_size_x
        self.ground_size_y = cfg.ground_size_y
        self.xs = tf.placeholder(tf.float32,shape=[None,self.image_size_x,self.image_size_y,3])
        self.x_image = tf.reshape(self.xs,[-1,self.image_size_x,self.image_size_y,3])
        self.training_flag = tf.placeholder(tf.bool)
        self.ys = tf.placeholder(tf.float32,shape=[None,self.ground_size_x,self.ground_size_y])
        self.y_image = tf.reshape(self.ys,[-1,self.ground_size_x,self.ground_size_y,1])
        
        self.prediction = self.model(self.x_image)
        self.loss_sum = self.loss_layer(self.prediction,self.y_image)
        
    #################################
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    ###################################
    def biases_variable(self,shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
    
    ###################################
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
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               collections=collections,
                               trainable=trainable)

    #####################
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
    #################################
    def model(self, x_image):
        with tf.variable_scope('model1'):
            #####8-9-26-27-28
            #model 1
            W_conv1 = self.weight_variable([5,5,3,36])
            b_conv1 = self.biases_variable([36])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W_conv2 = self.weight_variable([7,7,36,72])
            b_conv2 = self.biases_variable([72])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
            #h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W_conv3 = self.weight_variable([13,13,72,36])
            b_conv3 = self.biases_variable([36])
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
            #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W_conv4 = self.weight_variable([11,11,36,1])
            b_conv4 = self.biases_variable([1])
            h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)

            #h_conv4 = bn(h_conv4,training_flag)

        with tf.variable_scope('model2'):
            #####6-7-21-22-23
            #model2
            W2_conv1 = self.weight_variable([3,3,3,24])
            b2_conv1 = self.biases_variable([24])
            h2_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W2_conv1,strides=[1,1,1,1],padding='SAME')+b2_conv1)
            h2_pool1 = tf.nn.max_pool(h2_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W2_conv2 = self.weight_variable([5,5,24,48])
            b2_conv2 = self.biases_variable([48])
            h2_conv2 = tf.nn.relu(tf.nn.conv2d(h2_pool1,W2_conv2,strides=[1,1,1,1],padding='SAME')+b2_conv2)
            #h2_pool2 = tf.nn.max_pool(h2_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W2_conv3 = self.weight_variable([15,15,48,24])
            b2_conv3 = self.biases_variable([24])
            h2_conv3 = tf.nn.relu(tf.nn.conv2d(h2_conv2,W2_conv3,strides=[1,1,1,1],padding='SAME')+b2_conv3)
            #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W2_conv4 = self.weight_variable([11,11,24,1])
            b2_conv4 = self.biases_variable([1])
            h2_conv4 = tf.nn.relu(tf.nn.conv2d(h2_conv3,W2_conv4,strides=[1,1,1,1],padding='SAME')+b2_conv4)

            #h2_conv4 = bn(h2_conv4,training_flag)

        with tf.variable_scope('model3'):
            #####4-5-16-17-18
            #model3
            W3_conv1 = self.weight_variable([3,3,3,24])
            b3_conv1 = self.biases_variable([24])
            h3_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W3_conv1,strides=[1,1,1,1],padding='SAME')+b3_conv1)
            h3_pool1 = tf.nn.max_pool(h3_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W3_conv2 = self.weight_variable([5,5,24,48])
            b3_conv2 = self.biases_variable([48])
            h3_conv2 = tf.nn.relu(tf.nn.conv2d(h3_pool1,W3_conv2,strides=[1,1,1,1],padding='SAME')+b3_conv2)
            #h3_pool2 = tf.nn.max_pool(h3_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W3_conv3 = self.weight_variable([11,11,48,24])
            b3_conv3 = self.biases_variable([24])
            h3_conv3 = tf.nn.relu(tf.nn.conv2d(h3_conv2,W3_conv3,strides=[1,1,1,1],padding='SAME')+b3_conv3)
            #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W3_conv4 = self.weight_variable([9,9,24,1])
            b3_conv4 = self.biases_variable([1])
            h3_conv4 = tf.nn.relu(tf.nn.conv2d(h3_conv3,W3_conv4,strides=[1,1,1,1],padding='SAME')+b3_conv4)

            #h3_conv4 = bn(h3_conv4,training_flag)

        with tf.variable_scope('model4'):
            #####2-3-13-14
            #model4
            W4_conv1 = self.weight_variable([7,7,3,18])
            b4_conv1 = self.biases_variable([18])
            #h4_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(x_image,W4_conv1,rate=2,padding='SAME')+b4_conv1)
            h4_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W4_conv1,strides=[1,1,1,1],padding='SAME')+b4_conv1)    
            h4_pool1 = tf.nn.max_pool(h4_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W4_conv2 = self.weight_variable([9,9,18,36])
            b4_conv2 = self.biases_variable([36])
            h4_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h4_pool1,W4_conv2,rate=2,padding='SAME')+b4_conv2)
            #h4_conv2 = tf.nn.relu(tf.nn.conv2d(h4_pool1,W4_conv2,strides=[1,1,1,1],padding='SAME')+b4_conv2)  
            #h4_pool2 = tf.nn.max_pool(h4_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W4_conv3 = self.weight_variable([9,9,36,6])
            b4_conv3 = self.biases_variable([6])
            h4_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv2,W4_conv3,rate=2,padding='SAME')+b4_conv3)
            #h4_conv3 = tf.nn.relu(tf.nn.conv2d(h4_conv2,W4_conv3,strides=[1,1,1,1],padding='SAME')+b4_conv3)  
            #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W4_conv4 = self.weight_variable([7,7,6,1])
            b4_conv4 = self.biases_variable([1])
            #h4_conv4 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv3,W4_conv4,rate=2,padding='SAME')+b4_conv4)
            h4_conv4 = tf.nn.relu(tf.nn.conv2d(h4_conv3,W4_conv4,strides=[1,1,1,1],padding='SAME')+b4_conv4)  

            #h4_conv4 = bn(h4_conv4,training_flag)
        '''
        with tf.variable_scope('model5'):
            #####1-2
            #model5
            W5_conv1 = self.weight_variable([3,3,3,21])
            b5_conv1 = self.biases_variable([21])
            h5_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(self.x_image,W5_conv1,rate=2,padding='SAME')+b5_conv1)
            h5_pool1 = tf.nn.max_pool(h5_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W5_conv2 = self.weight_variable([7,7,21,42])
            b5_conv2 = self.biases_variable([42])
            h5_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h5_pool1,W5_conv2,rate=2,padding='SAME')+b5_conv2)
            #h5_pool2 = tf.nn.max_pool(h5_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W5_conv3 = self.weight_variable([7,7,42,7])
            b5_conv3 = self.biases_variable([7])
            h5_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h5_conv2,W5_conv3,rate=2,padding='SAME')+b5_conv3)
            #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W5_conv4 = self.weight_variable([5,5,7,1])
            b5_conv4 = self.biases_variable([1])
            h5_conv4 = tf.nn.relu(tf.nn.atrous_conv2d(h5_conv3,W5_conv4,rate=2,padding='SAME')+b5_conv4)

            #h5_conv4 = bn(h5_conv4,training_flag)
        '''
        with tf.variable_scope('merge_model'):
            #merged_layer = tf.nn.sigmoid(tf.add(tf.add(tf.add(tf.add(h_conv4,h2_conv4),h3_conv4),h4_conv4),h5_conv4))
            node = []
            node.append(h_conv4)
            node.append(h2_conv4)
            node.append(h3_conv4)
            node.append(h4_conv4)
            #node.append(h5_conv4)
            merged_layer = tf.concat(node,3)

            merge_weight = self.weight_variable([1,1,4,1])
            merge_bias = self.biases_variable([1])
            merge_conv = tf.nn.conv2d(merged_layer,merge_weight,strides=[1,1,1,1],padding='SAME')+merge_bias
            merge_conv = self.bn(merge_conv,self.training_flag)

            #Wm_conv1 = self.weight_variable([])
            return merge_conv
            
            
            
    def loss_layer(self,net,ground_truth):
        prediction = tf.reshape(tensor=net,shape=(-1,2))
        y_image_ = tf.nn.max_pool(ground_truth,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        y_image_ = tf.reshape(tensor=y_image_,shape=(-1,2))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_image_)
        loss_sum = tf.reduce_sum(loss)
        return loss_sum
            


