import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
#from torch.optim import lr_scheduler as torchLR
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import datetime
import scipy.io as scio

#from data import Source_Data
from transfer_data import get_data
import config as cfg

torch.manual_seed(1)    # reproducible

epoch = 1#write in cfg file
BATCH_SIZE = 1
learning_rate = 0.1 ##### find funtion to change it every training time 

TEST_SIZE = 28

train_times = 50
change_data = 2 

test_time = 2

data_class = get_data(train_flag=True)
train_source,ground_source = data_class.data()
torch_dataset = Data.TensorDataset(data_tensor = torch.from_numpy(train_source), target_tensor = torch.from_numpy(ground_source))
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 1,
)
print("train data ready-----")

'''
data_class = get_data(test_flag=True)
test_source,test_ground = data_class.data()
torch_dataset = Data.TensorDataset(
data_tensor = torch.from_numpy(test_source), target_tensor = torch.from_numpy(test_ground))
loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = TEST_SIZE,
    shuffle = False,
    num_workers = 2,
)
print("test data ready-----")
'''
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
    #return s_img
                    



class MCNN(nn.Module):
    def __init__(self):
        super(MCNN,self).__init__()
        #############################cnn1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 12,
                      kernel_size = 5,
                      stride = 1,
                      ##padding = (kernel_size-1/2)
                      padding = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 12,
                      out_channels = 24,
                      kernel_size = 7,
                      stride = 1,
                      padding = 3
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 24,
                      out_channels = 12,
                      kernel_size = 13,
                      stride = 1,
                      padding = 6
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 12,
                      out_channels = 1,
                      kernel_size = 11,
                      stride = 1,
                      padding = 5
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
        )
        
        
        ##########################cnn2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 9,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 18,
                      kernel_size = 5,
                      stride = 1,
                      padding = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 18,
                      out_channels = 9,
                      kernel_size = 15,
                      stride = 1,
                      padding = 7
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 1,
                      kernel_size = 11,
                      stride = 1,
                      padding = 5
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
        )
        
        
        
        ##################################cnn3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 9,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 18,
                      kernel_size = 5,
                      stride = 1,
                      padding = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 18,
                      out_channels = 6,
                      kernel_size = 11,
                      stride = 1,
                      padding = 5
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 6,
                      out_channels = 1,
                      kernel_size = 9,
                      stride = 1,
                      padding = 4
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
        )
        
        #################################conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 9,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 18,
                      kernel_size = 9,
                      stride = 1,
                      padding = 4
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 18,
                      out_channels = 9,
                      kernel_size = 9,
                      stride = 1,
                      padding = 4,
                      dilation = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 1,
                      kernel_size = 7,
                      stride = 1,
                      padding = 3,
                      dilation = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
        )
        
        
        
        ###################################conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 9,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 18,
                      kernel_size = 5,
                      stride = 1,
                      #padding = (kernel_size-1/2)
                      padding = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 18,
                      out_channels = 9,
                      kernel_size = 5,
                      stride = 1,
                      padding = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 9,
                      out_channels = 1,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      dilation = 2
                      ),
                      
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size = 2),
        )
        
        self.out = nn.Sequential(
            nn.Conv2d(in_channels = 5,
                      out_channels = 1,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0,
                      dilation = 2
                      )
        )
        
    def forward(self,x):
        #print x
        
        x1 = self.conv1(x)
        print(x1)
        x2 = self.conv2(x)
        print(x2)
        x3 = self.conv3(x)
        print(x3)
        x4 = self.conv4(x)
        print(x4)
        x5 = self.conv5(x)
        print(x5)
        merge = torch.cat((x1,x2,x3,x4,x5),1)
        print(merge)
        #x_out = self.out(torch.utils.data.ConcatDataset([x1,x2]))
        x_out = self.out(merge)
        return x_out 
        
model = MCNN()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
#scheduler = torchLR.MultiStepLR(optimizer, milestones=[1,3], gamma=0.5)
loss_func = nn.MSELoss()

print("start training -----")

for epoch_times in range(epoch):
    for step, (batch_x,batch_y) in enumerate(loader):
        #b_x = Variable(batch_x).double()
        #b_y = Variable(batch_y).double()
        b_x = Variable(batch_x).type(torch.FloatTensor)
        b_y = Variable(batch_y).type(torch.FloatTensor)
        #print b_x
        output = model(b_x)
        loss = loss_func(output, b_y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        if (i-1)%test_time==0:
            test_x = Variable()
            test_output = model()
            stitching(test_output)
        '''








