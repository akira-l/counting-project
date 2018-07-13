from __future__ import division
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import random
import cv2
import datetime
import scipy.io as scio


from data import Source_Data
import config as cfg
from model7 import Model_net
from fetch_resnet import get_resnet


image_size_y = cfg.image_size_y
image_size_x = cfg.image_size_x
ground_size_y = cfg.ground_size_y
ground_size_x = cfg.ground_size_x

total_result = []
train_loss_reco = []
lr_record = []
set_frame_record = []

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
data_cap_map = cfg.area_

decay_step_ = cfg.DECAY_STEP

set_frame = 1.0


##########################
def stitching(test_image,trainging_times,flag):
    s_img = np.zeros([720,1280],np.uint8)
    for i in range(len(data_cap_map)):
        img = test_image[i]
        h,w = img.shape[:2]
        tmp = cv2.resize(img,(8*w,8*h),interpolation=cv2.INTER_CUBIC)
        pc,pd,pa,pb = data_cap_map[i]
        for j in range(0,ground_size_x):
            for k in range(0,ground_size_y):
                s_img[pa+j,pc+k] = tmp[j,k]
    cv2.imwrite(save_test_path+flag+datetime.datetime.now().strftime('%m_%d_%H_%M_%S')+'_with-t-'+str(trainging_times)+'.jpg',s_img)
    #return s_img
                    
####################################
def get_test_data(name):
    data_baggage = scio.loadmat(test_path + 'data' + str(name) + '.mat')
    img = data_baggage.get('img'+str(name))
    ground = data_baggage.get('ground'+str(name))
    frame_1 = data_baggage.get('delta1_frame'+str(name))
    frame_2 = data_baggage.get('delta2_frame'+str(name))
    
    tbs = len(data_cap_map)
    test_batch = np.zeros([tbs,320,400,3])
    ground_batch = np.zeros([tbs,320,400])
    frame1_batch = np.zeros([tbs,320,400,3])
    frame2_batch = np.zeros([tbs,320,400,3])

    for i in range(tbs):
        area = data_cap_map[i]
        test_batch[i,:,:,:] = img[area[2]:area[3],area[0]:area[1],:]
        ground_batch[i,:,:] = ground[area[2]:area[3],area[0]:area[1]]
        frame1_batch[i,:,:,:] = frame_1[area[2]:area[3],area[0]:area[1],:]
        frame2_batch[i,:,:,:] = frame_2[area[2]:area[3],area[0]:area[1],:]
    return test_batch, ground_batch, frame1_batch, frame2_batch
########################################################



sequence = random.sample(range(1,sample_for_train+1),sample_for_train)
batch_times = 1
Data = Source_Data(batch_times,sequence)
train_batch,ground_batch,frame1_batch, frame2_batch = Data.get_train_data()


resnet_ = get_resnet()

data_x1, data_x2, data_x3 = resnet_.get_resnet_output(train_batch)


with tf.Graph().as_default() as g2:
    net = Model_net()

    prediction_img = net.prediction#tf.reshape(net.prediction,[-1,ground_size_x//2,ground_size_y//2,1])

    gloabal_step_ = tf.Variable(0,trainable=False)

    learning_rate = tf.train.inverse_time_decay(starting_learning_rate,global_step=gloabal_step_,decay_steps=decay_step_,decay_rate=learning_rate_decay,staircase=False)

    train_opt = tf.train.AdamOptimizer(learning_rate)
    add_global = gloabal_step_.assign_add(1)
    with tf.control_dependencies([add_global]):
        train_step = train_opt.minimize(net.loss_sum)
    saver2 = tf.train.Saver(max_to_keep=20)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    print("-----this expriment train  "+str(ts_num)+" times----- \n")

    for i in range(1,ts_num+1):
        print('training step: '+str(i))
        if i%100==0:
            set_frame = 1-i/(ts_num-3000)
            if set_frame <= 0:
                set_frame = 0
        #feed_dict={net.xs:train_batch,net.ys:ground_batch,net.training_flag:True}
        feed_dict={net.x_1:data_x1, net.x_2:data_x2, net.x_3:data_x3, net.dframe1:frame1_batch, net.dframe2:frame2_batch, net.ys:ground_batch, net.training_flag:True, net.frame_para:set_frame}
        _,train_loss_record = sess.run([train_step,net.loss_sum],feed_dict=feed_dict)
        train_loss_reco.append(train_loss_record)
        set_frame_record.append(set_frame)
        ##test accuracy
        if i%get_test_loss==0:
            test_, test_ground, test_frame1, test_frame2 = get_test_data(254)
            tdata_x1, tdata_x2, tdata_x3 = resnet_.get_resnet_output(test_)
            feed_dict = {net.x_1:tdata_x1, net.x_2:tdata_x2, net.x_3:tdata_x3, net.dframe1:test_frame1, net.dframe2:test_frame2, net.ys:test_ground, net.training_flag:False, net.frame_para:0}
            test_loss,lr_record_ = sess.run([net.loss_sum,learning_rate],feed_dict=feed_dict)
            print('test_loss single:  '+str(test_loss)+'\n')

            if i%record_test_map==0:
                feed_dict = {net.x_1:tdata_x1, net.x_2:tdata_x2, net.x_3:tdata_x3, net.dframe1:test_frame1, net.dframe2:test_frame2, net.training_flag:False, net.frame_para:0}
                test_prediction = sess.run(prediction_img,feed_dict=feed_dict)
                stitching(test_prediction,i,'single')

            test_, test_ground, test_frame1, test_frame2 = get_test_data(256)
            tdata_x1, tdata_x2, tdata_x3 = resnet_.get_resnet_output(test_)
            feed_dict = {net.x_1:tdata_x1, net.x_2:tdata_x2, net.x_3:tdata_x3, net.dframe1:test_frame1, net.dframe2:test_frame2, net.ys:test_ground, net.training_flag:False, net.frame_para:0}
            test_loss,lr_record_ = sess.run([net.loss_sum,learning_rate],feed_dict=feed_dict)
            lr_record.append(lr_record_)
            total_result.append(test_loss)
            print('test_loss multi:  '+str(test_loss)+'\n')

            if i%record_test_map==0:
                feed_dict = {net.x_1:tdata_x1, net.x_2:tdata_x2, net.x_3:tdata_x3, net.dframe1:test_frame1, net.dframe2:test_frame2, net.training_flag:False, net.frame_para:0}
                test_prediction = sess.run(prediction_img,feed_dict=feed_dict)
                stitching(test_prediction,i,'multi')
            
            '''
            plt.figure("ground map")
            plt.imshow(test_img)
            plt.pause(3)
            plt.close()
            '''
        if i%batch_change==0:
            batch_times += 1
            #print batch_times
            if batch_times > sample_for_train//cfg.need_img:
                sequence = random.sample(range(1,sample_for_train+1),sample_for_train)
                batch_times = 1
            Data = Source_Data(batch_times,sequence)
            train_batch, ground_batch, frame1_batch, frame2_batch = Data.get_train_data()
            data_x1, data_x2, data_x3 = resnet_.get_resnet_output(train_batch)

        if i>500 and i%save_times==0:
            cur_save_path = para_save_path+datetime.datetime.now().strftime('%m_%d_%H_%M')+'_'+str(i)+'/'
            os.makedirs(cur_save_path)
            saver2.save(sess,cur_save_path+'t-'+str(i)+'_l-'+str(test_loss)+'.ckpt')
            save_res_name = cur_save_path+'t-'+str(i)+'-resnet'+'.ckpt'
            resnet_.save_resnet(save_res_name)


plt.figure(0)
plt.subplot(221)
plt.title('test loss')
plt.plot(total_result)
plt.subplot(222)
plt.title('train loss')
plt.plot(train_loss_reco)
plt.subplot(223)
plt.title('learning rate')
plt.plot(lr_record)
plt.subplot(224)
plt.title('set frame')
plt.plot(set_frame_record)
plt.savefig(save_test_path+'result-'+datetime.datetime.now().strftime('%m_%d_%H_%M')+'.jpg')




