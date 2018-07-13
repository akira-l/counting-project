import os

size_x = 480
size_y = 640
train_size_x = 480
train_size_y = 640
train_data_path = "./train data"
ground_data_path = "./ground data"
total_data_num = 1200#400
batch_size = 4 
epoch = 30
train_times = epoch*(total_data_num//batch_size)



starting_learning_rate_ = 0.05
learning_rate_decay_rate = 0.2
DECAY_STEP = 200


extra_dataset1a_train = "./dataset/ShanghaiTech/part_A/train_data/images"
extra_dataset1a_ground = "./dataset/ShanghaiTech/part_A/train_data/ground-truth"
extra_dataset1b_train = "./dataset/ShanghaiTech/part_B/train_data/images"
extra_dataset1b_ground = "./dataset/ShanghaiTech/part_B/train_data/ground-truth"
extra_dataset2 = "./dataset/mall_dataset/"
