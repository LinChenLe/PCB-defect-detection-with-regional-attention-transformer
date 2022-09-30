from dataloader import build_loader
from config import get_args_parser
import torch
from torch.utils import data
import random
import pdb
from setcriterion import compute_segmentation_loss,Hungarian_match
from creat_model import build_segmentation_model
from Metric_Logger import Logger
from train_val import train_one_epoch,validation
import os
import matplotlib.pyplot as plt
import seaborn
from utils import load_model_parameter,smooth_value
from torch import nn

import cv2
from utils import bbox_xcychw2xyxy,bbox_xyxy2Oxyxy
def plotfig(x,y,title,xlabel,ylabel,fontsize,save_name=''):
    plt.plot(x,y,label=title)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.legend(loc='upper right')
    if save_name:
        plt.savefig(save_name)
        
def plot_confusion_matrix(x,title,xlabel,ylabel,fontsize,save_name=''):
    seaborn.heatmap(x)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if save_name:
        plt.savefig(save_name)
        
def get_lr(max_lr,warmup_start_lr,warmup_steps,end_epoch,epoch,p=3,clamp_lr=1e-7):
    warmup_factor = (max_lr/warmup_start_lr)**(1/warmup_steps)
    if epoch <= warmup_steps:
        lr = max(warmup_start_lr*(warmup_factor**epoch),clamp_lr)
    else:
        factor = (1-(epoch-warmup_steps)/(end_epoch-warmup_steps))**p
        lr = max((max_lr*factor),clamp_lr)

    return lr 
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # =============================================================================
    # training config


    # seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # =============================================================================
    train_data = build_loader(args,'train',True)
    val_data = build_loader(args,'val',False)
    
    print('building model...')
    dict_train_record = {'start_epoch':0,'end_epoch':500,
                         'train_losses':[],'val_losses':[],
                         'lr':[],
                         'train_acc':[],'val_acc':[],
                         "train_conf_acc":[],"val_conf_acc":[]}
    
    segmentation_loss_function = compute_segmentation_loss(device=args.device)
    backbone_model, segmentation_model = build_segmentation_model(args)
    
    segmentation_param_dicts = [{"params":[]},{"params":[]}]
    for n, p in segmentation_model.named_parameters():
        if "backbone" in n:
            segmentation_param_dicts[0]["params"] += [p]
            print("bcackbone param:",n)
        else:
            # print("segmentation param:",n)
            segmentation_param_dicts[1]["params"] += [p]
            print("segmentation param:",n)
        # else:
        #     raise ValueError("the model parameter has some worng, please check it")
    Hungarian = Hungarian_match(device=args.device)
    segmentation_optimizer = torch.optim.AdamW(segmentation_param_dicts, lr=args.segmentation_min_lr,weight_decay=1e-4)



    if args.load_ckpt == '' or args.load_ckpt == 'ckpt/':
        backbone_model = load_model_parameter(args.load_backbone_ckpt,
                                              backbone_model,False,False,args.device,mode_name="backbone")[0]
    if args.load_ckpt != '' and args.load_ckpt != 'ckpt/':
        model,segmentation_optimizer,record = load_model_parameter(args.load_ckpt,
                                                           [backbone_model,segmentation_model],segmentation_optimizer,True,args.device,
                                                           mode_name="segmentation")
        backbone_model = model[0]
        segmentation_model = model[1]
        # tensor([-0.0132, -0.0516, -0.1114, -0.0347, -0.0823,  0.1029], device='cuda:0'))])
        if args.training_mode == 'segmentation':
            for key in record.keys():
                dict_train_record[key] = record[key]

        
# =============================================================================
    dict_train_record['end_epoch'] = 500
# =============================================================================
    
    
    epochs = range(dict_train_record['start_epoch'],dict_train_record['end_epoch'])
    
    
    train_logger = Logger(args.logger_pfreq,train_data,args.batch_size,
                          args.not_show_list,args.not_smooth_list,dict_train_record['end_epoch'])
    
    val_logger = Logger(1,val_data,args.batch_size,args.not_show_list,
                        args.not_smooth_list,dict_train_record['end_epoch'])
    for epoch in epochs:
        lest_epoch = dict_train_record['end_epoch']-args.max_lr_epoch
        segmentation_optimizer.param_groups[0]['lr'] = get_lr(args.backbone_lr, args.backbone_min_lr,
                                                             args.max_lr_epoch, dict_train_record['end_epoch'], epoch,
                                                             clamp_lr=args.clamp_lr)
        segmentation_optimizer.param_groups[1]['lr'] = get_lr(args.segmentation_lr, args.segmentation_min_lr,
                                                             args.max_lr_epoch, dict_train_record['end_epoch'], epoch,
                                                             clamp_lr=args.clamp_lr)
        train_history = train_one_epoch(segmentation_model,segmentation_optimizer,
                                        Hungarian,train_data,segmentation_loss_function,
                                        train_logger,epoch+1,args.num_object,
                                        args.training_mode,args.device)

        dict_train_record['lr'] += [segmentation_optimizer.param_groups[0]['lr']]
        
        dict_train_record['train_losses'] += list(map(float,[smooth_value(train_history['loss'])]))
        # dict_train_record['train_acc'] += list(map(float,[smooth_value(train_history['acc'])]))
        # dict_train_record['train_confusion_matrix'] = train_history['confusion_matrix']
        # dict_train_record['train_conf_acc'] += list(map(float,[smooth_value(train_history['conf_acc'])]))
        # dict_train_record['train_conf_confusion_matrix'] = train_history['conf_confusion_matrix']

        if  args.val_rate != 0:
            if ((epoch+1) % args.val_rate) == 0:
                val_history = validation(segmentation_model,
                                         Hungarian,val_data,
                                         segmentation_loss_function,
                                         val_logger,epoch+1,args.num_object,
                                         args.training_mode,args.device)
                
                dict_train_record['val_losses'] += list(map(float,[smooth_value(val_history['loss'])]))
                # dict_train_record['val_acc'] += list(map(float,[smooth_value(val_history['acc'])]))
                # dict_train_record['val_confusion_matrix'] = val_history['confusion_matrix']
                # dict_train_record['val_conf_acc'] += list(map(float,[smooth_value(val_history['conf_acc'])]))
                # dict_train_record['val_conf_confusion_matrix'] = val_history['conf_confusion_matrix']

                if args.save_model_path:
                    if os.path.exists(args.save_model_path) == False:
                        os.makedirs(args.save_model_path)
                        pass
                    
                    if args.training_mode == 'segmentation':
                        torch.save({"backbonemodel":backbone_model.state_dict(),
                                    'segmentationmodel': segmentation_model.state_dict(),
                                    'optimizer': segmentation_optimizer.state_dict(),
                                    'start_epoch': epoch+1,
                                    'lr': dict_train_record['lr'],
                                    'end_epoch': dict_train_record['end_epoch'],
                                    'train_losses': dict_train_record['train_losses'],
                                    'val_losses': dict_train_record['val_losses'],
                                    'train_acc': dict_train_record['train_acc'],
                                    'val_acc': dict_train_record['val_acc'],
                                    "train_conf_acc": dict_train_record['train_conf_acc'],
                                    "val_conf_acc": dict_train_record['val_conf_acc']},
                                    args.save_model_path+
            f"/epoch_{epoch+1}_{args.training_mode}_trainloss_{round(dict_train_record['train_losses'][-1],2)}"+
            f"_valloss_{round(dict_train_record['val_losses'][-1],2)}")

    plt.figure()
    train_epochs = range(0,dict_train_record['end_epoch'])
    val_epochs = range(4,dict_train_record['end_epoch'],args.val_rate)
    plotfig(train_epochs, dict_train_record['train_losses'], 'loss', 'epochs', 'loss', 20, 'loss.pdf')
    plotfig(val_epochs, dict_train_record['val_losses'], 'loss', 'epochs', 'loss', 20, 'loss.pdf')
    # plt.figure()
    # plotfig(train_epochs, dict_train_record['train_acc'], 'acc', 'epochs', 'acc', 20, 'acc.pdf')
    # plotfig(val_epochs, dict_train_record['val_acc'], 'acc', 'epochs', 'acc', 20, 'acc.pdf')
    # plt.figure()
    # plt.plot(train_epochs,dict_train_record['lr'])
    # plt.figure()
    # plot_confusion_matrix(train_history['confusion_matrix'],'train_Confusion_Matrix',
    #                       'Predict','GroundTrue',20,'val_Confusion_matrix.pdf')
    # # plt.figure()
    # plot_confusion_matrix(val_history['confusion_matrix'],'val_Confusion_Matrix',
    #                       'Predict','GroundTrue',20,'val_Confusion_matrix.pdf')
