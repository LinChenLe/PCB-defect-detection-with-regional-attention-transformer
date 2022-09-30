import sys 
sys.path.append("../..") 
import os
import pdb
import re
import torch
from utils import box_iou
import matplotlib.pyplot as plt
import random
import numpy as np
# =============================================================================
# config
gt_path = "./gt/"
submit_path = "./submit/"
iou_th = 0.33
# =============================================================================
gt_list = os.listdir(gt_path)
submit_list = os.listdir(submit_path)

assert len(gt_list) == len(submit_list),f"the length of submit file is {len(submit_list)}"+\
    " not match with length of ground true {len(gt_list)}"
total_num = {1:0,
             2:0,
             3:0,
             4:0,
             5:0,
             6:0}
AP_info = {1:[],
           2:[],
           3:[],
           4:[],
           5:[],
           6:[]}
for idx in range(len(gt_list)):
    with open(gt_path+gt_list[idx],"r") as gt_str:
        gt_str = re.findall("(.*)\n",gt_str.read())
        gt_file_name = re.findall("(.*).txt",gt_list[idx])[0]
        gt_bbox = []
        gt_category = []
        for seq in range(len(gt_str)):
            gt_data = [float(x) for x in re.findall("(.*),(.*),(.*),(.*),(.*)",gt_str[seq])[0]]
            gt_bbox += [gt_data[:4]]
            gt_category += [gt_data[-1]]

            total_num[gt_data[-1]] +=1
        gt_bbox = torch.tensor(gt_bbox,dtype=torch.float32)
        gt_category = torch.tensor(gt_category,dtype=torch.long)
        
        
    with open(submit_path+submit_list[idx],"r") as submit_str:
        submit_str = re.findall("(.*)\n",submit_str.read())
        submit_file_name = re.findall("(.*).txt",submit_list[idx])[0]
        submit_bbox = []
        submit_conf = []
        submit_category = []
        for seq in range(len(submit_str)):
            submit_data = [float(x) for x in re.findall("(.*),(.*),(.*),(.*),(.*),(.*)",submit_str[seq])[0]]
            submit_bbox += [submit_data[:4]]
            submit_conf += [submit_data[4]]
            submit_category += [submit_data[-1]]
            
        submit_bbox = torch.tensor(submit_bbox,dtype=torch.float32)
        submit_conf = torch.tensor(submit_conf,dtype=torch.float32)
        submit_category = torch.tensor(submit_category,dtype=torch.long)
        
    match_iou = box_iou(submit_bbox,gt_bbox)[0]
    submit_conf,conf_idx = submit_conf.sort(descending=True)
    submit_bbox = submit_bbox[conf_idx]
    submit_category = submit_category[conf_idx]
    match_iou = match_iou[conf_idx]
    match_mask = match_iou>iou_th

    for idx in range(len(match_mask)):
        match_gt_idx = torch.argmax(match_iou[idx])
        # pdb.set_trace()
        category_mask = gt_category[match_gt_idx] == submit_category[idx]
        detected = match_mask[idx,match_gt_idx].clone()
        if detected != False:
            match_mask[:,match_gt_idx] = False
            match_mask[idx,match_gt_idx] = detected
        
        ToF = category_mask*match_mask[idx,match_gt_idx]
        a,b = match_mask.shape

        if ToF == True:
            AP_info[gt_category[match_gt_idx].item()] += [1]
        elif ToF == False:
            AP_info[gt_category[match_gt_idx].item()] += [0]

Precision ={1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[]}
Recall = {1:[],
          2:[],
          3:[],
          4:[],
          5:[],
          6:[]}
AP ={1:0,
     2:0,
     3:0,
     4:0,
     5:0,
     6:0}

for j in range(1,1+len(AP_info)):
    # pdb.set_trace()
    TP = np.cumsum(np.array(AP_info[j]))
    FP = np.cumsum(1-np.array(AP_info[j]))
    
    Recall[j] = TP/total_num[j]
    Precision[j] = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
    
 

for j in range(1,1+len(AP_info)):
    pre_recall = 0
        
    for i in range(len(Precision[j])):
        
        Precision[j][i] = max(Precision[j][i:])
        # pdb.set_trace()
        AP[j] += Precision[j][i]*(Recall[j][i]-pre_recall)
        pre_recall = Recall[j][i]

    plt.plot(Recall[j],Precision[j])


plt.title("Precision-Recall Curve",fontsize=20)
plt.ylabel("Precision",fontsize=15)
plt.xlabel("recall",fontsize=15)
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.grid(color = [0,0,0], linestyle = '--', linewidth = 0.5)  # 設定格線顏色、種類、寬度
plt.text(0.65,0.05,f"mAP:{(sum(AP.values())/6):0.4f}",color='black',fontsize=20)
# plt.text(0.55, 0.55, 'Hello World!', fontsize=20, color='green')
plt.legend([f"open (AP={AP[1]:.04f})",
            f"short (AP={AP[2]:.04f})",
            f"mousebite (AP={AP[3]:.04f})",
            f"spur (AP={AP[4]:.04f})",
            f"copper (AP={AP[5]:.04f})",
            f"pin-hole (AP={AP[6]:.04f})"],loc='lower left')
plt.savefig("../../PR.pdf")