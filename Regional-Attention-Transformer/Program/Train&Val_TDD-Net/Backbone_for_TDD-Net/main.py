from dataloader import build_loader
from config import get_args_parser
import torch
from torch.utils import data
import random
import pdb
from setcriterion import compute_backbone_loss,Hungarian_match
from creat_model import build_backbone_model
from Metric_Logger import Logger
from train_val import train_one_epoch,validation
import os
import matplotlib.pyplot as plt
import seaborn
from utils import smooth_value,load_model_parameter

import cv2
from utils import bbox_xcychw2xyxy,bbox_xyxy2Oxyxy
def plotfig(x,y,legend,title,xlabel,ylabel,fontsize,save_name='',loc="upper right"):
    plt.plot(x,y,label=legend)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.legend()
    plt.title(title,fontsize=fontsize)
    if save_name:
        plt.savefig(save_name,dpi=900)

        
def plot_confusion_matrix(x,title,xlabel,ylabel,fontsize,save_name=''):
    seaborn.heatmap(x)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    if save_name:
        plt.savefig(save_name)
        
def get_lr(max_lr,warmup_start_lr,warmup_steps,end_epoch,epoch,p=3):
    warmup_factor = (max_lr/warmup_start_lr)**(1/warmup_steps)
    if epoch <= warmup_steps:
        lr = warmup_start_lr*(warmup_factor**epoch)
    else:
        factor = (1-(epoch-warmup_steps)/(end_epoch-warmup_steps))**p
        lr = max_lr*factor

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
                         'train_acc':[],'val_acc':[]}
    
    backbone_loss_function = compute_backbone_loss()
    backbone_model = build_backbone_model(args)
    
    backbone_param_dicts = [{"params": [p for n, p in backbone_model.named_parameters()]}]
    backbone_optimizer = torch.optim.AdamW(backbone_param_dicts, lr=args.backbone_min_lr,weight_decay=1e-4)
    Hungarian = Hungarian_match(args.device)

    if args.load_backbone_ckpt != '' and args.load_backbone_ckpt != 'ckpt/':
        backbone_model,backbone_optimizer,record = load_model_parameter(args.load_backbone_ckpt,
                                                                    backbone_model,backbone_optimizer,args.device)
        if args.training_mode == 'backbone':
            dict_train_record = record

    
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
        backbone_optimizer.param_groups[0]['lr'] = get_lr(args.backbone_lr, args.backbone_min_lr,
                                                              args.max_lr_epoch, dict_train_record['end_epoch'], epoch)
        
        train_history = train_one_epoch(backbone_model,backbone_optimizer,
                                        Hungarian,train_data,backbone_loss_function,
                                        train_logger,epoch+1,args.num_object,
                                        args.training_mode,args.device)


        dict_train_record['lr'] += [backbone_optimizer.param_groups[0]['lr']]
        
        dict_train_record['train_losses'] += list(map(float,[smooth_value(train_history['loss'])]))
        dict_train_record['train_acc'] += list(map(float,[smooth_value(train_history['acc'])]))
        dict_train_record['train_confusion_matrix'] = train_history['confusion_matrix']
        if  args.val_rate != 0:
            if ((epoch+1) % args.val_rate) == 0:
                val_history = validation(backbone_model,
                                         Hungarian,val_data,
                                         backbone_loss_function,
                                         val_logger,epoch+1,args.num_object,
                                         args.training_mode,args.device)
                
                dict_train_record['val_losses'] += list(map(float,[smooth_value(val_history['loss'])]))
                dict_train_record['val_acc'] += list(map(float,[smooth_value(val_history['acc'])]))
                dict_train_record['val_confusion_matrix'] = val_history['confusion_matrix']
                if args.save_model_path:
                    if os.path.exists(args.save_model_path) == False:
                        os.makedirs(args.save_model_path)
                        pass
                    
                    if args.training_mode == 'backbone':
                        torch.save({'backbonemodel': backbone_model.state_dict(),
                                    'optimizer': backbone_optimizer.state_dict(),
                                    'start_epoch': epoch+1,
                                    'lr': dict_train_record['lr'],
                                    'end_epoch': dict_train_record['end_epoch'],
                                    'train_losses': dict_train_record['train_losses'],
                                    'val_losses': dict_train_record['val_losses'],
                                    'train_acc': dict_train_record['train_acc'],
                                    'val_acc': dict_train_record['val_acc']},
                                    args.save_model_path+
            f"/epoch_{epoch+1}_{args.training_mode}_trainloss_{round(dict_train_record['train_losses'][-1],2)}"+
            f"_valloss_{round(dict_train_record['val_losses'][-1],2)}")

    plt.figure()
    plt.annotate("train loss: "+str(round(dict_train_record['train_losses'][-1],3)),
                 xy=(500,dict_train_record['train_losses'][-1]),
                 xytext=(250,dict_train_record['train_losses'][-1]+2),
                 arrowprops = dict(
                     arrowstyle = "->",
                     connectionstyle = "angle,angleA=0,angleB=135,rad=0"))
    
    plt.annotate("val loss: "+str(round(dict_train_record['val_losses'][-1],3)),
                 xy=(500,dict_train_record['val_losses'][-1]),
                 xytext=(250,dict_train_record['val_losses'][-1]+2.5),
                 arrowprops = dict(
                     arrowstyle = "->",
                     connectionstyle = "angle,angleA=0,angleB=120,rad=0"))
    train_epochs = range(0,dict_train_record['end_epoch'])
    val_epochs = range(4,dict_train_record['end_epoch'],args.val_rate)
    plotfig(train_epochs,
            dict_train_record['train_losses'],
            'train loss',
            "train&val loss",
            'epoch',
            'loss',
            20,
            'loss.png')
    plotfig(val_epochs,
            dict_train_record['val_losses'],
            'val loss',
            "train&val loss",
            'epoch',
            'loss',
            20,
            'loss.png')
# =============================================================================
# 

# =============================================================================
    plt.figure()
    plt.annotate("train acc: "+str(round(dict_train_record['train_acc'][-1],3)),
                 xy=(500,dict_train_record['train_acc'][-1]),
                 xytext=(250,dict_train_record['train_acc'][-1]-0.1),
                 arrowprops = dict(
                     arrowstyle = "->",
                     connectionstyle = "angle,angleA=0,angleB=30,rad=0"))
    
    plt.annotate("val acc: "+str(round(dict_train_record['val_acc'][-1],3)),
                 xy=(500,dict_train_record['val_acc'][-1]),
                 xytext=(250,dict_train_record['val_acc'][-1]-0.15),
                 arrowprops = dict(
                     arrowstyle = "->",
                     connectionstyle = "angle,angleA=0,angleB=50,rad=0"))
    plotfig(train_epochs,
            dict_train_record['train_acc'],
            'train acc',
            "train&val acc",
            'epoch',
            'accuracy',
            20,
            'acc.png')
    plotfig(val_epochs,
            dict_train_record['val_acc'],
            'val acc',
            "train&val acc",
            'epoch',
            'accuracy',
            20,
            'acc.png')
# def plotfig(x,y,legend,title,xlabel,ylabel,fontsize,save_name='',loc="upper right"):
# =============================================================================
# 

# # =============================================================================
#     plt.figure()
#     plt.plot(train_epochs,dict_train_record['lr'])
#     # plt.figure()
#     plot_confusion_matrix(train_history['confusion_matrix'],'train_Confusion_Matrix',
#                           'Predict','GroundTrue',20,'val_Confusion_matrix.pdf')
#     # plt.figure()
#     plot_confusion_matrix(val_history['confusion_matrix'],'val_Confusion_Matrix',
#                           'Predict','GroundTrue',20,'val_Confusion_matrix.pdf')
