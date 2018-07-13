import os

size_x = 720
size_y = 1280
train_size_x = 300
train_size_y = 300
train_data_path = "./train data"
ground_data_path = "./ground data"
total_data_num = 256#400
batch_size = 8 
epoch = 40
train_times = epoch*(total_data_num//batch_size)



starting_learning_rate_ = 0.05
learning_rate_decay_rate = 0.2
DECAY_STEP = 200


extra_dataset1a_train = "./dataset/ShanghaiTech/part_A/train_data/images"
extra_dataset1a_ground = "./dataset/ShanghaiTech/part_A/train_data/ground-truth"
extra_dataset1b_train = "./dataset/ShanghaiTech/part_B/train_data/images"
extra_dataset1b_ground = "./dataset/ShanghaiTech/part_B/train_data/ground-truth"
extra_dataset2 = "./dataset/mall_dataset/"
