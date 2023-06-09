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
import numpy as np
import os
import cv2

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    def clip(coordinate):
        '''
        Clamp points out of the image size
        '''
        if coordinate < 0:
            return 0
        elif coordinate > 1:
            return 1
        else:
            return coordinate

    # number of grids for each layer: 100, 25, 9, 1
    grid_num = np.array(layers)**2
    # boxes size: [10*10+5*5+3*3+1*1,4,8]
    boxes = np.zeros([np.sum(grid_num),len(large_scale),8],dtype=np.float32)
    index_start = np.append([0],grid_num)
    # box: 0,1,2,3
    for box in range(len(layers)):
        length = 1.0/layers[box]
        start = np.sum(index_start[:box+1])
        # x: 1-10, 1-5, 1-3, 1, x is gonna be the rightmost value of the grid
        for x in range(1,layers[box]+1):
            for y in range(1,layers[box]+1):
                # for each four boxes in one grid
                for i in range(4):
                    x_center = (x - 0.5) * length
                    y_center = (y - 0.5) * length
                    if i == 0:
                        # [ssize,ssize]
                        box_width = small_scale[box]
                        box_height = small_scale[box]
                    elif i == 1:
                        # [lsize,lsize]
                        box_width = large_scale[box]
                        box_height = large_scale[box]
                    elif i == 2:
                        # [lsize*sqrt(2),lsize/sqrt(2)]
                        box_width = large_scale[box] * np.sqrt(2)
                        box_height = large_scale[box] / np.sqrt(2)
                    else:
                        # [lsize/sqrt(2),lsize*sqrt(2)]
                        box_width = large_scale[box] / np.sqrt(2)
                        box_height = large_scale[box] * np.sqrt(2)
                    x_min = clip(x_center - box_width/2.0)
                    x_max = clip(x_center + box_width/2.0)
                    y_min = clip(y_center - box_height/2.0)
                    y_max = clip(y_center + box_height/2.0)
                    # boxes[index of grid, ith box, :]
                    boxes[(x-1)*layers[box] + y-1 + start,i,:] = [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]    
    return boxes.reshape(np.sum(grid_num)*len(large_scale),8)


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    x_center, y_center, box_width, box_height = (x_min+x_max)/2, (y_min+y_max)/2, (x_max-x_min), (y_max-y_min)
    true_idx = [index for (index,value) in enumerate(ious_true) if value == True]
    # print(true_idx)
    for i in true_idx:
        ann_confidence[i][3] = 0
        ann_confidence[i][cat_id] = 1
        ann_box[i][0] = (x_center-boxs_default[i][0])/boxs_default[i][2]
        ann_box[i][1] = (y_center-boxs_default[i][1])/boxs_default[i][3]
        ann_box[i][2] = np.log(box_width/boxs_default[i][2])
        ann_box[i][3] = np.log(box_height/boxs_default[i][3])

    # Assign to the box with largest iou
    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    ann_confidence[ious_true][3] = 0
    ann_confidence[ious_true][cat_id] = 1
    ann_box[ious_true][0] = (x_center-boxs_default[ious_true][0])/boxs_default[ious_true][2]
    ann_box[ious_true][1] = (y_center-boxs_default[ious_true][1])/boxs_default[ious_true][3]
    ann_box[ious_true][2] = np.log(box_width/boxs_default[ious_true][2])
    ann_box[ious_true][3] = np.log(box_height/boxs_default[ious_true][3])

    return ann_box, ann_confidence

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320, with_ann=True):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        self.with_ann = with_ann
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if self.train:
            self.img_names = self.img_names[:int(0.9 * len(self.img_names))]
        else:
            self.img_names = self.img_names[int(0.9 * len(self.img_names)):]


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)
        
        
        
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        class_id, x_min, y_min, x_max, y_max = [], [], [], [], []
        if self.with_ann: 
            
            with open(ann_name, 'r') as f:
                for line in f:
                    c_id, x_s, y_s, w_box, h_box = line.split()
                    class_id.append(int(c_id)) 
                    x_min.append(float(x_s))  
                    y_min.append(float(y_s)) 
                    x_max.append(float(x_s) + float(w_box)) 
                    y_max.append(float(y_s) + float(h_box)) 
            class_id, x_min, y_min, x_max, y_max = np.asarray(class_id), np.asarray(x_min), np.asarray(y_min), np.asarray(x_max), np.asarray(y_max)

            

        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        if self.train: 
            random_x_min = np.random.randint(0, np.min(x_min), size=1)[0]
            random_y_min = np.random.randint(0, np.min(y_min), size=1)[0]
            random_x_max = np.random.randint(np.max(x_max)+1, width, size=1)[0]
            random_y_max = np.random.randint(np.max(y_max)+1, height, size=1)[0]

            x_min -= random_x_min
            y_min -= random_y_min
            x_max -= random_x_min
            y_max -= random_y_min
            
            width = random_x_max - random_x_min
            height = random_y_max - random_y_min
            image = image[random_y_min:random_y_max,random_x_min:random_x_max,:]

        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        for i in range(len(class_id)):
            ann_box, ann_confidence = match(ann_box, ann_confidence, self.boxs_default, self.threshold,\
                                            class_id[i], x_min[i]/width, y_min[i]/height, x_max[i]/width, y_max[i]/height)
        

        # if self.anndir: #self.train:
        #     for i in range(len(class_id)):
        #         ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,\
        #                                         class_id[i],x_min[i]/width,y_min[i]/height,x_max[i]/width,y_max[i]/height)
        
        image_preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize([self.image_size, self.image_size])])
        image  = image_preprocess(image)
        ann_box = torch.from_numpy(ann_box)
        ann_confidence = torch.from_numpy(ann_confidence)        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        if self.with_ann:
            return image, ann_box, ann_confidence
        else:
            return image
