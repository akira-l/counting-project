#configure
import os

image_size_y = 200
image_size_x = 160
ground_size_y = 100
ground_size_x = 80

batch_size = 2
partition = [1,1]#positive / nagetive

#data_path = "/home/yuanzhi/figure/mexp/version2/data/"
#test_path = "/home/yuanzhi/figure/mexp/version2/data/"
#save_test_path = "/home/yuanzhi/figure/mexp/version2/test map"
data_path = os.getcwd()+'/data/'
test_path = os.getcwd()+'/data/'
save_test_path = os.getcwd()+"/test map/"
if not os.path.exists(save_test_path):
    os.makedirs(save_test_path)

data_bag_total_num = 236
sample_for_train = 200
sample_for_test = 2
specific_test = True
test_sample = []

area = [[25,185,271,471],[121,281,138,338],[121,281,272,472],[279,439,71,271],[279,439,271,471],[441,601,7,207],[442,602,207,407],[560,720,2,202],[560,720,169,369],[35,195,471,671],[35,195,671,871],[158,318,471,671],[158,318,611,811],[158,318,811,1011],[279,439,471,671],[279,439,625,825],[279,439,805,1005],[279,439,987,1187],[439,599,437,637],[439,599,637,837],[439,599,707,907],[439,599,907,1107],[439,599,1080,1280],[560,720,369,569],[560,720,501,701],[560,720,701,901],[560,720,880,1080],[560,720,1080,1280]]
