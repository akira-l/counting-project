import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import cv2
import datetime
import scipy.io as scio
from data import Source_Data
import config as cfg


image_size_y = 200
image_size_x = 160
ground_size_y = 200
ground_size_x = 160

batch_times = 1
total_result = []


#####
def compute_accuracy(v_xs,v_ys,i):
    global prediction
    y_pre = tf.reshape(prediction,[-1,ground_size_x,ground_size_y,1])
    y_data = sess.run(y_pre,feed_dict={xs:v_xs,ys:v_ys})
    y_datashow = tf.reshape(y_data,[ground_size_x,ground_size_y])
    '''
    if ((i<200)and(i%50==0))or(i%2000==0):    
        img_show = sess.run(y_datashow)
        im = Image.fromarray(np.uint8(img_show))
        im.show()
    '''
    y_data = tf.to_float(tf.reshape(v_ys,[ground_size_x,ground_size_y]))
    result_error = tf.reduce_sum(tf.square(y_datashow-y_data))/2
    return sess.run(result_error)
#####
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#####
def biases_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
##########################
def stitching(test_image):
    area = cfg.area
    s_img = np.zeros([720,1280],np.uint8)
    for i in range(0,28):
        img = test_image[i]
        h,w = img.shape[:2]
        tmp = cv2.resize(img,(2*w,2*h),interpolation=cv2.INTER_CUBIC)
        pa,pb,pc,pd = area[i]
        for j in range(0,160):
            for k in range(0,200):
                if s_img[pa+j,pc+k]>tmp[j,k]:
                    s_img[pa+j,pc+k] = tmp[j,k]
    cv2.imwrite(cfg.save_test_path+datetime.datetime.now().strftime('%m_%d_%H_%M')+'.jpg',s_img)
    return s_img
                    
####################################
def test_subregion(test_name,num,arg_return):
    tmp = scio.loadmat(cfg.test_path+test_name)
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
        


ts_num = 50
batch_change = 2


#trainf = open(r'train_loss.txt','a')
#testf_per = open(r'test_loss_per_area.txt','a')
#testf = open(r'test_loss.txt','a')


sequence = random.sample(range(1,cfg.sample_for_train+1),cfg.sample_for_train)
batch_times = 1
Data = Source_Data(batch_times,sequence)
train_batch,ground_batch = Data.get_train_data()


test_batch = scio.loadmat(cfg.test_path+'data201.mat')

xs = tf.placeholder(tf.float32,shape=[None,image_size_x,image_size_y,3])
x_image = tf.reshape(xs,[-1,image_size_x,image_size_y,3])
ys = tf.placeholder(tf.float32,shape=[None,ground_size_x,ground_size_y])
y_image = tf.reshape(ys,[-1,ground_size_x,ground_size_y,1])




#####8-9-26-27-28
#model 1
W_conv1 = weight_variable([5,5,3,12])
b_conv1 = biases_variable([12])
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv2 = weight_variable([7,7,12,24])
b_conv2 = biases_variable([24])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
#h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv3 = weight_variable([13,13,24,12])
b_conv3 = biases_variable([12])
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
#h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv4 = weight_variable([11,11,12,1])
b_conv4 = biases_variable([1])
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)


#####6-7-21-22-23
#model2
W2_conv1 = weight_variable([3,3,3,9])
b2_conv1 = biases_variable([9])
h2_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W2_conv1,strides=[1,1,1,1],padding='SAME')+b2_conv1)
h2_pool1 = tf.nn.max_pool(h2_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W2_conv2 = weight_variable([5,5,9,18])
b2_conv2 = biases_variable([18])
h2_conv2 = tf.nn.relu(tf.nn.conv2d(h2_pool1,W2_conv2,strides=[1,1,1,1],padding='SAME')+b2_conv2)
#h2_pool2 = tf.nn.max_pool(h2_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W2_conv3 = weight_variable([15,15,18,9])
b2_conv3 = biases_variable([9])
h2_conv3 = tf.nn.relu(tf.nn.conv2d(h2_conv2,W2_conv3,strides=[1,1,1,1],padding='SAME')+b2_conv3)
#h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W2_conv4 = weight_variable([11,11,9,1])
b2_conv4 = biases_variable([1])
h2_conv4 = tf.nn.relu(tf.nn.conv2d(h2_conv3,W2_conv4,strides=[1,1,1,1],padding='SAME')+b2_conv4)


#####4-5-16-17-18
#model3
W3_conv1 = weight_variable([3,3,3,9])
b3_conv1 = biases_variable([9])
h3_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W3_conv1,strides=[1,1,1,1],padding='SAME')+b3_conv1)
h3_pool1 = tf.nn.max_pool(h3_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W3_conv2 = weight_variable([5,5,9,18])
b3_conv2 = biases_variable([18])
h3_conv2 = tf.nn.relu(tf.nn.conv2d(h3_pool1,W3_conv2,strides=[1,1,1,1],padding='SAME')+b3_conv2)
#h3_pool2 = tf.nn.max_pool(h3_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W3_conv3 = weight_variable([11,11,18,6])
b3_conv3 = biases_variable([6])
h3_conv3 = tf.nn.relu(tf.nn.conv2d(h3_conv2,W3_conv3,strides=[1,1,1,1],padding='SAME')+b3_conv3)
#h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W3_conv4 = weight_variable([9,9,6,1])
b3_conv4 = biases_variable([1])
h3_conv4 = tf.nn.relu(tf.nn.conv2d(h3_conv3,W3_conv4,strides=[1,1,1,1],padding='SAME')+b3_conv4)


#####2-3-13-14
#model4
W4_conv1 = weight_variable([3,3,3,9])
b4_conv1 = biases_variable([9])
h4_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(x_image,W4_conv1,rate=2,padding='SAME')+b4_conv1)
h4_pool1 = tf.nn.max_pool(h4_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W4_conv2 = weight_variable([9,9,9,18])
b4_conv2 = biases_variable([18])
h4_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h4_pool1,W4_conv2,rate=2,padding='SAME')+b4_conv2)
#h4_pool2 = tf.nn.max_pool(h4_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W4_conv3 = weight_variable([9,9,18,9])
b4_conv3 = biases_variable([9])
h4_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv2,W4_conv3,rate=2,padding='SAME')+b4_conv3)
#h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W4_conv4 = weight_variable([7,7,9,1])
b4_conv4 = biases_variable([1])
h4_conv4 = tf.nn.relu(tf.nn.atrous_conv2d(h4_conv3,W4_conv4,rate=2,padding='SAME')+b4_conv4)


#####1-2
#model5
W5_conv1 = weight_variable([3,3,3,9])
b5_conv1 = biases_variable([9])
h5_conv1 = tf.nn.relu(tf.nn.atrous_conv2d(x_image,W5_conv1,rate=4,padding='SAME')+b5_conv1)
#h5_pool1 = tf.nn.max_pool(h5_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W5_conv2 = weight_variable([5,5,9,18])
b5_conv2 = biases_variable([18])
h5_conv2 = tf.nn.relu(tf.nn.atrous_conv2d(h5_conv1,W5_conv2,rate=4,padding='SAME')+b5_conv2)
h5_pool2 = tf.nn.max_pool(h5_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W5_conv3 = weight_variable([5,5,18,9])
b5_conv3 = biases_variable([9])
h5_conv3 = tf.nn.relu(tf.nn.atrous_conv2d(h5_pool2,W5_conv3,rate=4,padding='SAME')+b5_conv3)
#h_pool3 = tf.nn.max_pool(h_conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W5_conv4 = weight_variable([3,3,9,1])
b5_conv4 = biases_variable([1])
h5_conv4 = tf.nn.relu(tf.nn.atrous_conv2d(h5_conv3,W5_conv4,rate=4,padding='SAME')+b5_conv4)

merged_layer = tf.nn.sigmoid(tf.add(tf.add(tf.add(tf.add(h_conv4,h2_conv4),h3_conv4),h4_conv4),h5_conv4))


prediction = tf.reshape(merged_layer,[-1,ground_size_x/2,ground_size_y/2,1])
y_image = tf.nn.max_pool(y_image,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
loss = tf.reduce_sum(tf.square(tf.subtract(prediction,y_image)))/2
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(1,ts_num+1):
    print('training step: ',i)
    _,train_loss_record = sess.run([train_step,loss],feed_dict={xs:train_batch,ys:ground_batch})
    
    #trainf.write(str(sum(train_loss_record))+'\n')
    ##test accuracy
    if i%8==0:
        test_,test_ground = test_subregion('data201',201,1)
        test_loss = sess.run(loss,feed_dict={xs:test_,ys:test_ground})
        #testf_per.write(str(test_loss)+'\n')
        total_result.append(test_loss)
        #testf.write(str(sum(test_loss))+'\n')
        print('sum for test_loss:  ',test_loss,'\n')
        
    ##test for map, show result
    if i%10==0:
        test_ = test_subregion('data201',201,0)
        test_prediction = sess.run(prediction,feed_dict={xs:test_})
        test_img = stitching(test_prediction)
        '''
        plt.figure("ground map")
        plt.imshow(test_img)
        plt.pause(5)
        plt.close()
        '''
        
    if i%batch_change==0:
        batch_times += 1
        #print batch_times
        if batch_times>200:
            sequence = random.sample(range(1,cfg.sample_for_train+1),cfg.sample_for_train)
            batch_times = 1
        Data = Source_Data(batch_times,sequence)
        train_batch,ground_batch = Data.get_train_data()
    '''
    time_record = ts_num-i
    if (time_record<500)and(time_record%100==0):
        saver.save(sess,'/home/yuanzhi/figure/exp3/overall-test/result/result'+str(i)+'.ckpt')
    '''
plt.plot(total_result)
plt.show()

#testf.close()
#testf_per.close()
#trainf.close()

