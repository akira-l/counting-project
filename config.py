#configure
import os
import datetime

image_size_y = 400
image_size_x = 320
ground_size_y = 400
ground_size_x = 320


partition = [12,3]#positive / nagetive
batch_size = sum(partition)
need_amount = int(1.5*batch_size//2)

data_path = os.getcwd()+'/data2/'
test_path_ = os.getcwd()+'/data2/'
discription = "data250 setframe-1000 repeat set2"
save_test_path_ = os.getcwd()+"/test map N2feature"+ discription+"/"
para_save_path_ = os.getcwd()+"/parameter_save N2feature"+ discription+"/"
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
