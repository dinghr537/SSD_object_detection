import numpy as np
import cv2, torch
from dataset import iou
import matplotlib.pyplot as plt


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4 
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    image_ = image_ * 255
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                gt_x_center = (ann_box[i][0] * boxs_default[i][2]) + boxs_default[i][0]
                gt_y_center = (ann_box[i][1] * boxs_default[i][3]) + boxs_default[i][1]

                gt_box_width = boxs_default[i][2] * np.exp(ann_box[i][2])
                gt_box_height = boxs_default[i][3] * np.exp(ann_box[i][3])

                gt_x_min = gt_x_center - gt_box_width/2
                gt_y_min = gt_y_center - gt_box_height/2

                gt_x_max = gt_x_center + gt_box_width/2
                gt_y_max = gt_y_center + gt_box_height/2

                gt_x_min = gt_x_min * image.shape[0]
                gt_y_min = gt_y_min * image.shape[1]

                gt_x_max = gt_x_max * image.shape[0]
                gt_y_max = gt_y_max * image.shape[1]

                start_point = (int(gt_x_min), int(gt_y_min))
                end_point = (int(gt_x_max), int(gt_y_max)) 

                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                df_x_min =int(boxs_default[i][4] *  image.shape[0])
                df_y_min =int(boxs_default[i][5] *  image.shape[1])

                df_x_max =int(boxs_default[i][6] *  image.shape[0])
                df_y_max =int(boxs_default[i][7] *  image.shape[1])

                start_point_df = (df_x_min, df_y_min)
                end_point_df = (df_x_max, df_y_max)
              
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                color1 = (255, 0, 0)
                thickness = 1
                image1 = cv2.rectangle(image1, start_point, end_point, colors[j], thickness)
                image2 = cv2.rectangle(image2, start_point_df, end_point_df, colors[j], thickness)
    
    #pred
    res = []
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3                
                gt_x_center = (pred_box[i][0] * boxs_default[i][2]) + boxs_default[i][0]
                gt_y_center = (pred_box[i][1] * boxs_default[i][3]) + boxs_default[i][1]

                gt_box_width = boxs_default[i][2] * np.exp(pred_box[i][2])
                gt_box_height = boxs_default[i][3] * np.exp(pred_box[i][3])

                gt_x_min = gt_x_center - gt_box_width/2
                gt_y_min = gt_y_center - gt_box_height/2

                gt_x_max = gt_x_center + gt_box_width/2
                gt_y_max = gt_y_center + gt_box_height/2

                gt_x_min = gt_x_min * image.shape[0]
                gt_y_min = gt_y_min * image.shape[1]

                gt_x_max = gt_x_max * image.shape[0]
                gt_y_max = gt_y_max * image.shape[1]

                start_point = (int(gt_x_min), int(gt_y_min))
                end_point = (int(gt_x_max), int(gt_y_max)) 

                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                df_xmin =int(boxs_default[i][4] *  image.shape[0])
                df_ymin =int(boxs_default[i][5] *  image.shape[1])
                df_xmax =int(boxs_default[i][6] *  image.shape[0])
                df_ymax =int(boxs_default[i][7] *  image.shape[1])
                start_point_df = (df_xmin, df_ymin)
                end_point_df = (df_xmax, df_ymax)
                
                if gt_x_min < 0 or gt_y_min < 0 or gt_x_max > 320 or gt_y_max > 320:
                    continue

                thickness = 1
                image3 = cv2.rectangle(image3, start_point, end_point, colors[j], thickness)
                image4 = cv2.rectangle(image4, start_point_df, end_point_df, colors[j], thickness)
                res.append([j, gt_x_min, gt_y_min, gt_box_width, gt_box_height])

    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.imwrite('./test_result/' + windowname+".jpg",image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.


# def calculate_overlap(abs_x_min, abs_y_min, abs_x_max, abs_y_max, boxB, boxs_default):
#     # boxB here is the other boxes ready to test with the selected box
#     B_x_center = (boxB[0] * boxs_default[2]) + boxs_default[0]
#     B_y_center = (boxB[1] * boxs_default[3]) + boxs_default[1]
#     B_box_width = boxs_default[2] * np.exp(boxB[2])
#     B_box_height = boxs_default[3] * np.exp(boxB[3])
#     B_x_min = B_x_center - B_box_width/2
#     B_y_min = B_y_center - B_box_height/2
#     B_x_max = B_x_center + B_box_width/2
#     B_y_max = B_y_center + B_box_height/2

#     inter = np.maximum(np.minimum(B_x_max,abs_x_max) - np.maximum(B_x_min,abs_x_min),0)*np.maximum(np.minimum(B_y_max,abs_y_max) - np.maximum(abs_y_min,B_y_min),0)
#     area_a = (abs_x_max-abs_x_min)*(abs_y_max-abs_y_min)
#     area_b = (B_x_max-B_x_min)*(B_y_max-B_y_min)
#     union = area_a + area_b - inter
#     res = inter/np.maximum(union,1e-8)
#     # print(res)
#     return res

def get_iou_without_rescale(boxA, boxB, boxes_default):
    # [540, 4]
    x1, y1, w1, h1 = boxA
    x2, y2, w2, h2 = boxB
    xA = max(x1 - w1/2, x2 - w2/2)
    yA = max(y1 - h1/2, y2 - h2/2)
    xB = min(x1 + w1/2, x2 + w2/2)
    yB = min(y1 + h1/2, y2 + h2/2)

    interArea = max(0, xB - xA) * max(0, yB - yA)
 
    boxAArea = w1 * h1
    boxBArea = w2 * h2
 
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def combine_box_confidence(confidence, boxes):
    result = []
    # confidence = confidence.clone().detach()
    # boxes = boxes.clone().detach()
    confidence = torch.tensor(confidence)
    boxes = torch.tensor(boxes)
    for conf, box in zip(confidence, boxes):
        # for each image in the list
        prob = []
        
        # print(conf.shape) # 540 x 4 ?
        idx = torch.argmax(conf, dim=1)
        # idx will be a list like [0, 1, 1, 2, 3, 0, 0 ...] with len=540
        # for each box with its correspondence class 
        for c, i in zip(conf, idx):
            # add in the probability of that class, like 0.2 / 0.3
            prob.append(c[i].item())
        prob = torch.tensor(np.asarray(prob))
        box = torch.cat((box, prob.unsqueeze(1)), dim=1)
        box = torch.cat((box, idx.unsqueeze(1)), dim=1)

        result.append(box.numpy())
        

    return np.asarray(result)


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.

    # get box from relative to abs
    dx,dy,dw,dh = box_[:,0], box_[:,1], box_[:,2], box_[:,3]
    px,py,pw,ph = boxs_default[:,0], boxs_default[:,1], boxs_default[:,2], boxs_default[:,3]

    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)

    bbox = np.zeros_like(boxs_default, dtype=np.float32) # 8 elements
    bbox[:,0] = gx # xcenter
    bbox[:,1] = gy # ycenter
    bbox[:,2] = gw # w
    bbox[:,3] = gh # h
    bbox[:,4] = gx - gw/2.0 # xmin
    bbox[:,5] = gy - gh/2.0 # ymin
    bbox[:,6] = gx + gw/2.0 # xmax
    bbox[:,7] = gy + gh/2.0 # ymax
    
    # the best box that can be find
    out_bbox = np.zeros_like(box_, dtype=np.float32)
    out_confidence = np.zeros_like(confidence_, dtype=np.float32)
    # set background class to 1
    # print(out_confidence.shape)
    out_confidence[:,-1] = 1
    while True:
        # for all boxes, discard background
        highest = np.argmax(confidence_[:,0:-1])
        # get the 2-d index of the highest value(confident) in it, r is the box index among 540 and c is the class
        r, c = np.unravel_index(np.argmax(confidence_[:,0:-1]), confidence_[:,0:-1].shape)
        # as long as this confident is higher than threshold
        if confidence_[r,c] >= threshold: # take predictions that have a prob over 0.5
            # copy this box to output
            out_bbox[r,:] = box_[r,:]
            out_confidence[r,:] = confidence_[r,:]

            # if predictions overlap over threshold
            ious = iou(bbox,bbox[r,4],bbox[r,5],bbox[r,6],bbox[r,7])
            ious_bigger_threshold = np.where(ious > overlap)[0]
            # remove overlapped bboxes, includes it self, so that next loop will not find itself again
            box_[ious_bigger_threshold,:] = [0., 0., 0., 0.]
            confidence_[ious_bigger_threshold,:] = [0, 0, 0, 1]
            bbox[ious_bigger_threshold,:] = [0., 0., 0., 0., 0., 0., 0., 0.]
        else:
            return out_confidence, out_bbox


def generate_mAP(pred_confidence, pred_bboxes, gt_confidence, gt_bboxes, boxes_default, iou_threshold = 0.5):
    #TODO: Generate mAP
    # Initialize variables to store precision and recall values
    # pred_confidence: shape is (batch, 540, 4) [0,0,1,0] one-hot class confidence
    # pred_bboxes: elements [xcenter, ycenter, width, height], has the same shape as confidence
    # gt_confidence: the same as prediction confidence
    # gt_bboxes: the same as prediction boxes
    # boxes_default: default boxes of an image
    # print('In [0]')
    new_predictions = []
    new_ground_truths = []

    # recover bboxes
    for pre in pred_bboxes:
        # new_predictions.append(recover(pre, boxes_default))
        dx = pre[:,0]
        dy = pre[:,1]
        dw = pre[:,2]
        dh = pre[:,3]
        px = boxes_default[:, 0]
        py = boxes_default[:, 1]
        pw = boxes_default[:, 2]
        ph = boxes_default[:, 3]
        gx = pw*dx+px
        gy = ph*dy+py
        gw = pw*np.exp(dw)
        gh = ph*np.exp(dh)
        new_predictions.append([gx, gy, gw, gh])

    # print('In [1]')
    for gt in gt_bboxes:
        # new_ground_truths.append(recover(gt, boxes_default))
        dx = gt[:,0]
        dy = gt[:,1]
        dw = gt[:,2]
        dh = gt[:,3]
        px = boxes_default[:, 0]
        py = boxes_default[:, 1]
        pw = boxes_default[:, 2]
        ph = boxes_default[:, 3]
        gx = pw*dx+px
        gy = ph*dy+py
        gw = pw*np.exp(dw)
        gh = ph*np.exp(dh)
        new_ground_truths.append([gx, gy, gw, gh])
    # print('In [2]')
    # change the shape (b, 540, 4)
    # print(np.asarray(new_predictions).shape)
    # print(np.asarray(new_ground_truths).shape)
    new_predictions = np.transpose(np.asarray(new_predictions),(0,2,1))
    new_ground_truths = np.transpose(np.asarray(new_ground_truths),(0,2,1))
    # print('In [3]')
    # combine with conf (b, 540, 6)
    predictions = combine_box_confidence(pred_confidence, new_predictions)
    ground_truths = combine_box_confidence(gt_confidence, new_ground_truths)
    

    epsilon = 1e-6
    AP = []
    # print('In [4]')
    colormap = ['red', 'blue', 'green']
    plt.title("Precision - Recall Curve")
    for c in range(3): # traverse over all classes
        # print(f'class {c}')
        preds = []
        gts = []
        recalls = []
        precisions = []
        # get predictions for the current class
        # print(f'len(predictions) {len(predictions)}')
        for pr in predictions:
            
            for p in pr:
                if p[-1] == c:
                    preds.append(p)
        for gt in ground_truths:
            for g in gt:
                if g[-1] == c:
                    gts.append(g)
        # print('In [5]')            
        # logger
        # TP = np.zeros(len(preds))
        # FP = np.zeros(len(preds))
        TP = 0
        FP = 0
        # sort preds by conf
        preds.sort(key=lambda x: x[-2], reverse=True) # sort it by confidence
        total_true_bboxes = len(gts)
        if total_true_bboxes == 0:
            continue
        # print(total_true_bboxes)
        # print(len(preds))
        # print('In [6]')
        # print(len(predictions))
        # calculate iou to count TPs and FPs
        for idx, p in enumerate(preds):
            # print(idx)
            best_iou = 0
            for gt in gts:
                #TODO:calculate iou
                iou = get_iou_without_rescale(p[:4], gt[:4], boxes_default)
                if iou > best_iou:
                    best_iou = iou
            if best_iou > iou_threshold:
                # TP[idx] = 1
                TP += 1
                precisions.append(TP/(idx+1+epsilon))
                recalls.append(TP/(total_true_bboxes+epsilon))
            else:
                # FP[idx] = 1
                FP += 1
                precisions.append(TP/(idx+1+epsilon))
                recalls.append(TP/(total_true_bboxes+epsilon))

        # print('In [7]')
        # accumulate TP and FP
        # TP = np.cumsum(TP)
        # FP = np.cumsum(FP)
        # print('In [8]')
        # calculate recalls and precisions for the current class
        # print(f'recall: TP = {TP}, total_true_bboxes = {total_true_bboxes}')
        # print(f'precision: TP = {TP}, FP = {FP}')
        # recalls = TP / (total_true_bboxes + epsilon)
        # precisions = TP / (TP + FP + epsilon)
        # precisions.append()
        # recalls.append()

        # calculate AP
        # print(recalls)
        AP.append(np.trapz(precisions, recalls))
        plt.plot(recalls, precisions, color=colormap[c], label="Class:{}".format(c))

        # mAP over all classes
    
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig('PR.jpg')
    return sum(AP) / (len(AP))









