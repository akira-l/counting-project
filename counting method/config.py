#configure
import os
import datetime

image_size_y = 400
image_size_x = 320
ground_size_y = 400
ground_size_x = 320


partition = [12,3]#positive / nagetive
batch_size = sum(partition)
data_get_from_one_img = 5
need_img = (batch_size//data_get_from_one_img)+1

#data_path = "/home/yuanzhi/figure/mexp/version2/data/"
#test_path = "/home/yuanzhi/figure/mexp/version2/data/"
#save_test_path = "/home/yuanzhi/figure/mexp/version2/test map"
data_path = os.getcwd()+'/data2/'
test_path_ = os.getcwd()+'/data2/'
discription = "data250 setframe-1000 repeat set2"
save_test_path_ = os.getcwd()+"/test map 1-23 "+ discription+"/"
para_save_path_ = os.getcwd()+"/parameter_save 1-23"+ discription+"/"
if not os.path.exists(para_save_path_):
    os.makedirs(para_save_path_)
if not os.path.exists(save_test_path_):
    os.makedirs(save_test_path_)

sample_for_train_ = 250
sample_for_test = 2
random_val4box = 30
#specific_test = True
times4epoch = sample_for_train_//batch_size
epoch = 140
batch_change = 4
train_step_times = epoch*times4epoch*batch_change
starting_learning_rate_ = 0.05
learning_rate_decay_rate = 0.2
DECAY_STEP = 200
time4save_para = 100

times4get_test_loss = 25
times4record_test_map = 50


test_sample = []

area_ = [[0,400,400,720],[880,1280,400,720],[71,471,21,341],[596,996,31,351],[487,887,360,680]]
#area = [[25,185,271,471],[121,281,138,338],[121,281,272,472],[279,439,71,271],[279,439,271,471],[441,601,7,207],[442,602,207,407],[560,720,2,202],[560,720,169,369],[35,195,471,671],[35,195,671,871],[158,318,471,671],[158,318,611,811],[158,318,811,1011],[279,439,471,671],[279,439,625,825],[279,439,805,1005],[279,439,987,1187],[439,599,437,637],[439,599,637,837],[439,599,707,907],[439,599,907,1107],[439,599,1080,1280],[560,720,369,569],[560,720,501,701],[560,720,701,901],[560,720,880,1080],[560,720,1080,1280]]
