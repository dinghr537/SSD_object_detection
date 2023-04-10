import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    pred_confidence = pred_confidence.view(pred_confidence.shape[0] * 540, -1)
    ann_confidence = ann_confidence.view(ann_confidence.shape[0] * 540, -1)
    pred_box = pred_box.view(pred_box.shape[0] * 540, -1)
    ann_box = ann_box.view(ann_box.shape[0] * 540, -1)
    object_boxes = ann_confidence[:, 3] != 1


    l_cls = F.binary_cross_entropy(pred_confidence[object_boxes], ann_confidence[object_boxes], reduction='mean') \
        + 3 * F.binary_cross_entropy(pred_confidence[~object_boxes], ann_confidence[~object_boxes], reduction='mean')
    
    l_box = F.smooth_l1_loss(pred_box[object_boxes], ann_box[object_boxes])
    
    return l_box + l_cls
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.

# def conv_basic(in_planes, out_planes, kernelsize, stride):
#     padding = (kernelsize-1) // 2
#     x = nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding),
#             nn.BatchNorm2d(out_planes),
#             nn.ReLU(True),
#         )
#     return x

class basic_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(basic_conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class simple_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(simple_block, self).__init__()
        # padding = (kernel_size-1) // 2
        # print(f'kernel {kernel_size}, padding {padding}')
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        self.base_net = nn.Sequential(
            basic_conv_block(3, 64),
            basic_conv_block(64, 128),
            basic_conv_block(128, 256),
            basic_conv_block(256, 512),
            nn.Conv2d(512, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # self.conv1 = conv_basic(3, 64, 3, 2)
        # self.conv2 = conv_basic(64, 64, 3, 1)
        # self.conv3 = conv_basic(64, 64, 3, 1)
        # self.conv4 = conv_basic(64, 128, 3, 2)
        # self.conv5 = conv_basic(128, 128, 3, 1)
        # self.conv6 = conv_basic(128, 128, 3, 1)
        # self.conv7 = conv_basic(128, 256, 3, 2)
        # self.conv8 = conv_basic(256, 256, 3, 1)
        # self.conv9 = conv_basic(256, 256, 3, 1)
        # self.conv10 = conv_basic(256, 512, 3, 2)
        # self.conv11 = conv_basic(512, 512, 3, 1)
        # self.conv12 = conv_basic(512, 512, 3, 1)
        # self.conv13 = conv_basic(512, 256, 3, 2)

        self.conv1 = simple_block(256, 256, 1, 1, 0)
        self.conv2 = simple_block(256, 256, 3, 2 ,1)

        self.conv3 = simple_block(256, 256, 1, 1, 0)
        self.conv4 = simple_block(256, 256, 3, 1, 0)

        self.conv5 = simple_block(256, 256, 1, 1, 0)
        self.conv6 = simple_block(256, 256, 3, 1, 0)
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3,stride=3,padding=1),
        #     nn.ReLU(True)
        # )
        self.conv1_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0)
        #TODO: define layers
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        out = self.base_net(x)
        # print(out.shape)
        # 10 * 10
        right1 = self.conv1_1(out)
        right1 = right1.reshape((right1.shape[0], right1.shape[1], -1))
        left1 = self.conv1_2(out)
        left1 = left1.reshape((left1.shape[0], left1.shape[1], -1))
        out = self.conv1(out)
        out = self.conv2(out)
        # print(left1.shape)
        # print(out.shape)
        # 5 * 5
        right2 = self.conv2_1(out)
        right2 = right2.reshape((right2.shape[0], right2.shape[1], -1))
        left2 = self.conv2_2(out)
        left2 = left2.reshape((left2.shape[0], left2.shape[1], -1))
        # print(left2.shape)
        out = self.conv3(out)
        out = self.conv4(out)
        # print(out.shape)
        # 3 * 3
        right3 = self.conv3_1(out)
        right3 = right3.reshape((right3.shape[0], right3.shape[1], -1))
        left3 = self.conv3_2(out)
        left3 = left3.reshape((left3.shape[0], left3.shape[1], -1))
        out = self.conv5(out)
        out = self.conv6(out)
        # print(left3.shape)
        # print(out.shape)
        # 1 * 1
        right4 = self.conv4_1(out)
        right4 = right4.reshape((right4.shape[0], right4.shape[1], -1))
        left4 = self.conv4_2(out)
        left4 = right4.reshape((left4.shape[0], left4.shape[1], -1))

        # print(left4.shape)
        
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]

        # the concatenating order need to be the same as default bounding box !!!!!!!!!!!!!!!!!!!!!!!!
        confidence = torch.cat((left1, left2, left3, left4), 2)
        bboxes = torch.cat((right1, right2, right3, right4), 2)
        # print(confidence.shape)
        # print(bboxes.shape)

        confidence =  torch.permute(confidence, (0, 2, 1))
        bboxes =  torch.permute(bboxes, (0, 2, 1))

        # print(confidence.shape)
        # print(bboxes.shape)
        # print(self.class_num)
        confidence = torch.reshape(confidence, (confidence.shape[0], 540, -1))
        bboxes = torch.reshape(bboxes, (bboxes.shape[0], 540, -1))
        confidence = F.softmax(confidence, dim=2)
        return confidence, bboxes


if __name__ == "__main__":
    network = SSD(10)

    # network.cuda()
    img = torch.randn((8, 3, 320, 320))
    # img = img.cuda()
    confidence, bboxes = network(img)
    print(confidence)
    print(bboxes.shape)








