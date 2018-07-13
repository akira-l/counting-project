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
        self.in1_size_x = self.image_size_x//2
        self.in1_size_y = self.image_size_y//2
        self.in1_size_h = 64
        self.in2_size_x = self.image_size_x//4
        self.in2_size_y = self.image_size_y//4
        self.in2_size_h = 256
        self.in3_size_x = self.image_size_x//8
        self.in3_size_y = self.image_size_y//8
        self.in3_size_h = 512
        
        self.ys = tf.placeholder(tf.float32,shape=[None,self.ground_size_x,self.ground_size_y])
        self.y_image = tf.reshape(self.ys,[-1,self.ground_size_x,self.ground_size_y,1])
        self.training_flag = tf.placeholder(tf.bool)
        
        self.x_1 = tf.placeholder(tf.float32,shape=[None, self.in1_size_x, self.in1_size_y, self.in1_size_h])
        self.x_input1 = tf.reshape(self.x_1, [-1, self.in1_size_x, self.in1_size_y, self.in1_size_h])
        
        self.x_2 = tf.placeholder(tf.float32,shape=[None, self.in2_size_x, self.in2_size_y, self.in2_size_h])
        self.x_input2 = tf.reshape(self.x_2, [-1, self.in2_size_x, self.in2_size_y, self.in2_size_h])
        
        self.x_3 = tf.placeholder(tf.float32,shape=[None, self.in3_size_x, self.in3_size_y, self.in3_size_h])
        self.x_input3 = tf.reshape(self.x_3, [-1, self.in3_size_x, self.in3_size_y, self.in3_size_h])
        
        self.dframe1 = tf.placeholder(tf.float32, shape=[None, self.image_size_x, self.image_size_y, 3])
        self.dframe2 = tf.placeholder(tf.float32, shape=[None, self.image_size_x, self.image_size_y, 3])
        dframe1_ = tf.reshape(self.dframe1, [-1, self.image_size_x, self.image_size_y, 3])
        dframe2_ = tf.reshape(self.dframe2, [-1, self.image_size_x, self.image_size_y, 3])
        node = []
        node.append(dframe1_)
        node.append(dframe2_)
        self.merge_frame = tf.concat(node,3)
        
        
        self.prediction = self.model(self.x_input1, self.x_input2, self.x_input3, self.merge_frame)
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
    def model(self, x_input1, x_input2, x_input3, x_frame):
        
        with tf.variable_scope('model1'):
            #model 1
            W_conv1 = self.weight_variable([7,7,64,32])
            b_conv1 = self.biases_variable([32])
            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_input1, W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            
            W_conv2 = self.weight_variable([7,7,32,8])
            b_conv2 = self.biases_variable([8])
            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
            #h1_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            
            W_conv3 = self.weight_variable([5,5,8,1])
            b_conv3 = self.biases_variable([1])
            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
            
            
        with tf.variable_scope('model2'):
            #model2
            W2_conv1 = self.weight_variable([3,3,256,32])
            b2_conv1 = self.biases_variable([32])
            h2_conv1 = tf.nn.relu(tf.nn.conv2d(x_input2, W2_conv1,strides=[1,1,1,1],padding='SAME')+b2_conv1)
            #h2_pool1 = tf.nn.max_pool(h2_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            W2_conv2 = self.weight_variable([5,5,32,4])
            b2_conv2 = self.biases_variable([4])
            h2_conv2 = tf.nn.relu(tf.nn.conv2d(h2_conv1,W2_conv2,strides=[1,1,1,1],padding='SAME')+b2_conv2)

            W2_conv3 = self.weight_variable([5,5,4,1])
            b2_conv3 = self.biases_variable([1])
            h2_conv3 = tf.nn.relu(tf.nn.conv2d(h2_conv2,W2_conv3,strides=[1,1,1,1],padding='SAME')+b2_conv3)
            print('-----print shape--',tf.shape(h2_conv3))

        with tf.variable_scope('model3'):
            #model3
            W3_conv1 = self.weight_variable([3,3,512,64])
            b3_conv1 = self.biases_variable([64])
            h3_conv1 = tf.nn.relu(tf.nn.conv2d(x_input3, W3_conv1,strides=[1,1,1,1],padding='SAME')+b3_conv1)

            W3_conv2 = self.weight_variable([5,5,64,8])
            b3_conv2 = self.biases_variable([8])
            h3_conv2 = tf.nn.relu(tf.nn.conv2d(h3_conv1,W3_conv2,strides=[1,1,1,1],padding='SAME')+b3_conv2)
            
            W3_conv3 = self.weight_variable([5,5,8,1])
            b3_conv3 = self.biases_variable([1])
            h3_conv3 = tf.nn.relu(tf.nn.conv2d(h3_conv2,W3_conv3,strides=[1,1,1,1],padding='SAME')+b3_conv3)
            
            tmp_shape = tf.shape(h2_conv3)
            W_trans = self.weight_variable([3,3,1,1])
            h_trans = tf.nn.conv2d_transpose(h3_conv3, W_trans, tmp_shape, strides=[1,2,2,1],padding='SAME')
            
            print(tf.shape(h_trans))
        
        with tf.variable_scope('delta_frame'):
            WF_conv1 = self.weight_variable([3,3,6,1])
            bF_conv1 = self.biases_variable([1])
            hF_conv1 = tf.nn.relu(tf.nn.conv2d(x_frame, WF_conv1, strides=[1,1,1,1], padding='SAME')+bF_conv1)
            hF_conv1 = tf.nn.max_pool(hF_conv1,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')


        with tf.variable_scope('merge_model'):
            #merged_layer = tf.nn.sigmoid(tf.add(tf.add(tf.add(tf.add(h_conv4,h2_conv4),h3_conv4),h4_conv4),h5_conv4))
            node = []
            node.append(h_conv3)
            node.append(h2_conv3)
            node.append(h_trans)
            node.append(hF_conv1)
            merged_layer = tf.concat(node,3)

            merge_weight = self.weight_variable([1,1,4,1])
            merge_bias = self.biases_variable([1])
            merge_conv = tf.nn.conv2d(merged_layer,merge_weight,strides=[1,1,1,1],padding='SAME')+merge_bias
            merge_conv = self.bn(merge_conv,self.training_flag)

            #Wm_conv1 = self.weight_variable([])
            return merge_conv
            
            
            
    def loss_layer(self,net,ground_truth):
        prediction = tf.reshape(tensor=net,shape=(-1,2))
        print("check loss layer:-------")
        print(prediction.shape)
        #y_image_ = tf.nn.max_pool(ground_truth,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
        y_image_ = tf.nn.max_pool(ground_truth,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
        #y_image_ = tf.nn.max_pool(y_image_,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        
        print(tf.shape(y_image_))
        y_image_ = tf.reshape(tensor=y_image_,shape=(-1,2))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y_image_)
        loss_sum = tf.reduce_sum(loss)
        return loss_sum
            


