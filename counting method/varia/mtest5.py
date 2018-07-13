import os
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import random
import cv2
import datetime
import scipy.io as scio
from tensorflow.python.ops import control_flow_ops  
from tensorflow.python.training import moving_averages  

from data import Source_Data
import config as cfg


image_size_y = cfg.image_size_y
image_size_x = cfg.image_size_x
ground_size_y = cfg.ground_size_y
ground_size_x = cfg.ground_size_x

total_result = []
train_loss_reco = []
lr_record = []
ts_num = cfg.train_step_times
batch_change = cfg.batch_change
starting_learning_rate = cfg.starting_learning_rate_
get_test_loss = cfg.times4get_test_loss
record_test_map = cfg.times4record_test_map
learning_rate_decay = cfg.learning_rate_decay_rate
save_times = cfg.time4save_para
para_save_path = cfg.para_save_path_
save_test_path = cfg.save_test_path_
sample_for_train = cfg.sample_for_train_
test_path = cfg.test_path_


BN_DECAY = 0.9997
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
BN_EPSILON = 0.001
RESNET_VARIABLES = 'resnet_variables'

decay_step_ = cfg.DECAY_STEP

#####
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#####
def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
#####
def _get_variable(name,
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
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

#####
def bn(x, is_training):
    x_shape = x.get_shape()  
    params_shape = x_shape[-1:]  
    axis = list(range(len(x_shape) - 1))  
    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())  
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())  
    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
    # These ops will only be preformed when training.  
    mean, variance = tf.nn.moments(x, axis)  
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)  
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)  
    mean, variance = control_flow_ops.cond(  
        is_training, lambda: (mean, variance),  
        lambda: (moving_mean, moving_variance))  
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON) 
##########################
def stitching(test_image,trainging_times):
    area = cfg.area
    s_img = np.zeros([720,1280],np.uint8)
    for i in range(0,28):
        img = test_image[i]
        h,w = img.shape[:2]
        tmp = cv2.resize(img,(2*w,2*h),interpolation=cv2.INTER_CUBIC)
        pa,pb,pc,pd = area[i]
        for j in range(0,ground_size_x):
            for k in range(0,ground_size_y):
                if s_img[pa+j,pc+k]>tmp[j,k]:
                    s_img[pa+j,pc+k] = tmp[j,k]
    cv2.imwrite(save_test_path+datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_with-t-'+str(trainging_times)+'.jpg',s_img)
    return s_img
                    
####################################
def test_subregion(test_name,num,arg_return):
    tmp = scio.loadmat(test_path+test_name)
    img = tmp.get('img'+str(num))
    ground = np.double(tmp.get('ground'+str(num)))
    area = cfg.area
    test_data = []
    if arg_return == 0:
        #just return image
        for i in range(28):
            test_data.append(img[area[i][0]:area[i][1],area[i][2]:area[i][3],:])
        return test_data
    if arg_return == 1:
        #return image and ground
        ground_data = []
        for j in range(28):
            test_data.append(img[area[j][0]:area[j][1],area[j][2]:area[j][3],:])
            ground_data.append(ground[area[j][0]:area[j][1],area[j][2]:area[j][3]])
        return test_data,ground_data
        
########################################################


trainf = open(r'train_loss.txt','a')
testf_per = open(r'test_loss_per_area.txt','a')


sequence = random.sample(range(1,sample_for_train+1),sample_for_train)
batch_times = 1
Data = Source_Data(batch_times,sequence)
train_batch,ground_batch = Data.get_train_data()


test_batch = scio.loadmat(test_path+'data201.mat')

xs = tf.placeholder(tf.float32,shape=[None,image_size_x,image_size_y,3])
x_image = tf.reshape(xs,[-1,image_size_x,image_size_y,3])
ys = tf.placeholder(tf.float32,shape=[None,ground_size_x,ground_size_y])
y_image = tf.reshape(ys,[-1,ground_size_x,ground_size_y,1])
training_flag = tf.placeholder(tf.bool)

with tf.variable_scope('model1'):
    #####8-9-26-27-28
    #model 1
    W_conv1 = weight_variable([5,5,3,36])
    b_conv1 = biases_variable([36])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_conv2 = weight_variable([7,7,36,72])
    b_conv2 = biases_variable([72])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
    #h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_conv3 = weight_variable([13,13,72,36])
    b_conv3 = biases_variable([36])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
    #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W_conv4 = weight_variable([11,11,36,1])
    b_conv4 = biases_variable([1])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)

    #h_conv4 = bn(h_conv4,training_flag)

with tf.variable_scope('model2'):
    #####6-7-21-22-23
    #model2
    W2_conv1 = weight_variable([3,3,3,24])
    b2_conv1 = biases_variable([24])
    h2_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W2_conv1,strides=[1,1,1,1],padding='SAME')+b2_conv1)
    h2_pool1 = tf.nn.max_pool(h2_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W2_conv2 = weight_variable([5,5,24,48])
    b2_conv2 = biases_variable([48])
    h2_conv2 = tf.nn.relu(tf.nn.conv2d(h2_pool1,W2_conv2,strides=[1,1,1,1],padding='SAME')+b2_conv2)
    #h2_pool2 = tf.nn.max_pool(h2_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W2_conv3 = weight_variable([15,15,48,24])
    b2_conv3 = biases_variable([24])
    h2_conv3 = tf.nn.relu(tf.nn.conv2d(h2_conv2,W2_conv3,strides=[1,1,1,1],padding='SAME')+b2_conv3)
    #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W2_conv4 = weight_variable([11,11,24,1])
    b2_conv4 = biases_variable([1])
    h2_conv4 = tf.nn.relu(tf.nn.conv2d(h2_conv3,W2_conv4,strides=[1,1,1,1],padding='SAME')+b2_conv4)

    #h2_conv4 = bn(h2_conv4,training_flag)

with tf.variable_scope('model3'):
    #####4-5-16-17-18
    #model3
    W3_conv1 = weight_variable([3,3,3,24])
    b3_conv1 = biases_variable([24])
    h3_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W3_conv1,strides=[1,1,1,1],padding='SAME')+b3_conv1)
    h3_pool1 = tf.nn.max_pool(h3_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W3_conv2 = weight_variable([5,5,24,48])
    b3_conv2 = biases_variable([48])
    h3_conv2 = tf.nn.relu(tf.nn.conv2d(h3_pool1,W3_conv2,strides=[1,1,1,1],padding='SAME')+b3_conv2)
    #h3_pool2 = tf.nn.max_pool(h3_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W3_conv3 = weight_variable([11,11,48,24])
    b3_conv3 = biases_variable([24])
    h3_conv3 = tf.nn.relu(tf.nn.conv2d(h3_conv2,W3_conv3,strides=[1,1,1,1],padding='SAME')+b3_conv3)
    #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W3_conv4 = weight_variable([9,9,24,1])
    b3_conv4 = biases_variable([1])
    h3_conv4 = tf.nn.relu(tf.nn.conv2d(h3_conv3,W3_conv4,strides=[1,1,1,1],padding='SAME')+b3_conv4)

    #h3_conv4 = bn(h3_conv4,training_flag)

with tf.variable_scope('model4'):
    #####2-3-13-14
    #model4
    W4_conv1 = weight_variable([7,7,3,18])
    b4_conv1 = biases_variable([18])
    #h4_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(x_image,W4_conv1,rate=2,padding='SAME')+b4_conv1)
    h4_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W4_conv1,strides=[1,1,1,1],padding='SAME')+b4_conv1)    
    h4_pool1 = tf.nn.max_pool(h4_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W4_conv2 = weight_variable([9,9,18,36])
    b4_conv2 = biases_variable([36])
    h4_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h4_pool1,W4_conv2,rate=2,padding='SAME')+b4_conv2)
    #h4_conv2 = tf.nn.relu(tf.nn.conv2d(h4_pool1,W4_conv2,strides=[1,1,1,1],padding='SAME')+b4_conv2)  
    #h4_pool2 = tf.nn.max_pool(h4_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W4_conv3 = weight_variable([9,9,36,6])
    b4_conv3 = biases_variable([6])
    h4_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv2,W4_conv3,rate=2,padding='SAME')+b4_conv3)
    #h4_conv3 = tf.nn.relu(tf.nn.conv2d(h4_conv2,W4_conv3,strides=[1,1,1,1],padding='SAME')+b4_conv3)  
    #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W4_conv4 = weight_variable([7,7,6,1])
    b4_conv4 = biases_variable([1])
    #h4_conv4 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv3,W4_conv4,rate=2,padding='SAME')+b4_conv4)
    h4_conv4 = tf.nn.relu(tf.nn.conv2d(h4_conv3,W4_conv4,strides=[1,1,1,1],padding='SAME')+b4_conv4)  

    #h4_conv4 = bn(h4_conv4,training_flag)
'''
with tf.variable_scope('model5'):
    #####1-2
    #model5
    W5_conv1 = weight_variable([3,3,3,21])
    b5_conv1 = biases_variable([21])
    h5_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(x_image,W5_conv1,rate=2,padding='SAME')+b5_conv1)
    h5_pool1 = tf.nn.max_pool(h5_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W5_conv2 = weight_variable([7,7,21,42])
    b5_conv2 = biases_variable([42])
    h5_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h5_pool1,W5_conv2,rate=2,padding='SAME')+b5_conv2)
    #h5_pool2 = tf.nn.max_pool(h5_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W5_conv3 = weight_variable([7,7,42,7])
    b5_conv3 = biases_variable([7])
    h5_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h5_conv2,W5_conv3,rate=2,padding='SAME')+b5_conv3)
    #h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    W5_conv4 = weight_variable([5,5,7,1])
    b5_conv4 = biases_variable([1])
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

    merge_weight = weight_variable([1,1,4,1])
    merge_bias = biases_variable([1])
    merge_conv = tf.nn.conv2d(merged_layer,merge_weight,strides=[1,1,1,1],padding='SAME')+merge_bias
    
    merge_conv = bn(merge_conv,training_flag)

    #Wm_conv1 = weight_variable([])

#prediction = tf.reshape(merge_conv,[-1,ground_size_x//2,ground_size_y//2,1])
prediction_img = tf.reshape(merge_conv,[-1,ground_size_x//2,ground_size_y//2,1])
prediction = tf.reshape(tensor=merge_conv,shape=(-1,2))
y_image = tf.nn.max_pool(y_image,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
y_image = tf.reshape(tensor=y_image,shape=(-1,2))
#loss = tf.losses.mean_squared_error(y_image,prediction)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                               labels=y_image)
loss_sum = tf.reduce_sum(loss)

gloabal_step_ = tf.Variable(0,trainable=False)

#learning_rate = tf.train.exponential_decay(starting_learning_rate,global_step=gloabal_step_,decay_steps=10,decay_rate=learning_rate_decay,staircase=True)
learning_rate = tf.train.inverse_time_decay(starting_learning_rate,global_step=gloabal_step_,decay_steps=decay_step_,decay_rate=learning_rate_decay,staircase=False)

train_opt = tf.train.AdamOptimizer(learning_rate)
add_global = gloabal_step_.assign_add(1)
with tf.control_dependencies([add_global]):
    train_step = train_opt.minimize(loss_sum)#,global_step=ts_num)

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())


print("-----this expriment train  "+str(ts_num)+" times----- \n")

for i in range(1,ts_num+1):
    print('training step: '+str(i))
    _,train_loss_record = sess.run([train_step,loss_sum],feed_dict={xs:train_batch,ys:ground_batch,training_flag:True})
    train_loss_reco.append(train_loss_record)
    trainf.write(str(train_loss_record)+'\n')
    ##test accuracy
    if i%get_test_loss==0:
        test_,test_ground = test_subregion('data201',201,1)
        test_loss,lr_record_ = sess.run([loss_sum,learning_rate],feed_dict={xs:test_,ys:test_ground,training_flag:False})
        lr_record.append(lr_record_)
        testf_per.write(str(test_loss)+'\n')
        total_result.append(test_loss)
        print('test_loss:  '+str(test_loss)+'\n')
        #print(sess.run(y_image,feed_dict={xs:test_,ys:test_ground,training_flag:False}))

        if i%record_test_map==0:
            test_prediction = sess.run(prediction_img,feed_dict={xs:test_,training_flag:False})
            test_img = stitching(test_prediction,i)
        '''
        plt.figure("ground map")
        plt.imshow(test_img)
        plt.pause(3)
        plt.close()
        '''
        
    if i%batch_change==0:
        batch_times += 1
        #print batch_times
        if batch_times>200:
            sequence = random.sample(range(1,sample_for_train+1),sample_for_train)
            batch_times = 1
        Data = Source_Data(batch_times,sequence)
        trpara_save_pathain_batch,ground_batch = Data.get_train_data()

    if i>500 and i%save_times==0:
        cur_save_path = para_save_path+datetime.datetime.now().strftime('%m_%d_%H_%M')+'_'+str(i)+'/'
        os.makedirs(cur_save_path)
        saver.save(sess,cur_save_path+'t-'+str(i)+'_l-'+str(test_loss)+'.ckpt')


plt.figure(0)
plt.subplot(131)
plt.title('test loss')
plt.plot(total_result)
plt.subplot(132)
plt.title('train loss')
plt.plot(train_loss_reco)
plt.subplot(133)
plt.title('learning rate')
plt.plot(lr_record)
plt.savefig(save_test_path+'result-'+datetime.datetime.now().strftime('%m_%d_%H_%M')+'.jpg')





testf_per.close()
trainf.close()

