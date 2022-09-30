import re
import os
import numpy as np
import cv2
import pdb
dir_path = './DeepPCB-master/PCBData/'
file_path = dir_path+'trainval.txt'# trainval.txt or test.txt
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
                category += [each_label[-1]]
            img_info = {'temp_img_path':dir_path+img_name+'_temp.jpg',
                          'test_img_path':dir_path+img_name+'_test.jpg',
                          'bbox':bbox,
                          'category':category}
            test_img = cv2.imread(img_info['test_img_path'],0)/255
            temp_img = cv2.imread(img_info['temp_img_path'],0)/255
            # *255).clip(0,255).astype("uint8")
            
            H,W = test_img.shape
            mask = np.zeros([H,W,len(bbox)+1],dtype='float64')
            mask[:,:,-1] = np.full([H,W],1,dtype='float64')
            category = np.array(category)
            for label_idx in range(len(bbox)):
                    # img = ((((temp_img.copy()-test_img.copy())**2)**0.5))
                    
                    box_SX,box_SY,box_EX,box_EY = bbox[label_idx]
                    mask[box_SY:box_EY,
                         box_SX:box_EX,
                         label_idx] = 1
                    mask[box_SY:box_EY,
                         box_SX:box_EX,
                         -1] = 0
                    # mask[:,:,label_idx] = img*mask[:,:,label_idx]
                    # mask[:,:,label_idx] = (mask[:,:,label_idx])
            mask = (mask*255).clip(0,255).astype("uint8")
            # for label_idx in range(len(bbox)):
            #     cv2.imshow("a",mask[:,:,label_idx])
            #     cv2.waitKey(0)
            # cv2.imshow("a",mask[:,:,-1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print('processing: ',img_name)
            
            np.save(dir_path+img_name+'_mask',mask)
            np.save(dir_path+img_name+'_category',category)