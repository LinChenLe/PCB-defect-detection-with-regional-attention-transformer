import re
import os
import numpy as np
import cv2
import pdb
dir_path = './augmentation_Surface-Defect-Detection-master/DeepPCB/PCBData/'
file_path = dir_path+'test.txt'
with open(file_path) as file:
    for i in file:
        img_name = re.findall('(.*).jpg',i)[0]
        label_path = dir_path+re.findall('.jpg (.*.txt)',i)[0]
        with open(label_path) as label_file:
            bbox = []
            category = []
            for each_label_txt in label_file:
                each_label = re.findall('(.*) (.*) (.*) (.*) (\d{1})',each_label_txt)[0]
                each_label = [int(x) for x in each_label]
                bbox +=[each_label[:-1]]
                category += [each_label[-1]-1]
            img_info = {'temp_img_path':dir_path+img_name+'_temp.jpg',
                          'test_img_path':dir_path+img_name+'_test.jpg',
                          'bbox':bbox,
                          'category':category}
            test_img = cv2.imread(img_info['test_img_path'],0)
            H,W = test_img.shape
            mask = np.zeros([H,W,len(bbox)],dtype='uint8')
            category = np.array(category)
            for label_idx in range(len(bbox)):
                    box_SX,box_SY,box_EX,box_EY = bbox[label_idx]
                    mask[box_SY:box_EY,
                         box_SX:box_EX,
                         label_idx] = 255
            np.save(dir_path+img_name+'_mask',mask)
            np.save(dir_path+img_name+'_category',category)