import torch
from torch import nn
import pdb
from typing import Iterable
import numpy as np
from utils import smooth_value
from setcriterion import confusion_matrix_F
@torch.no_grad()
def train_one_epoch(backbone_model:nn.Module,backbone_optimizer:torch.optim,
                    Hungarian,dataloader:Iterable, 
                    backbone_loss_function:nn.Module,
                    Metric_Logger:callable, epoch:int,num_object:int,
                    training_mode:str,device:torch.device):
# =============================================================================
    record = {}
    record['loss'] = []
    record['category_loss'] = []
    record['acc'] = []
    record['lr'] = []
    record['confusion_matrix'] = np.zeros([num_object+1,num_object+1])

    backbone_model.train()
    backbone_loss_function.train()

    Metric_Logger.reset_log()
    for idx,(tested,categories) in enumerate(dataloader):

        tested = tested.to(device)
        categories = categories.to(device)

        backbone_result = backbone_model(tested)
             
        
        loss = backbone_loss_function(backbone_result,categories)
        
        category_loss = loss['category_loss'].item()
        # backbone_optimizer.zero_grad()
        # (loss['category_loss']).backward()
        # backbone_optimizer.step()
        
        record['loss'] +=[category_loss]
        record['category_loss'] += [category_loss]
        record['confusion_matrix'] += confusion_matrix_F(backbone_result,categories,num_object)
        if np.sum(record['confusion_matrix'])==0:
            record["acc"] += [0]
        else:
            record['acc'] += [np.sum(record['confusion_matrix'].diagonal())/np.sum(record['confusion_matrix'])]
        record['lr'] +=[backbone_optimizer.param_groups[0]['lr']]
# =============================================================================
        Metric_Logger.update(record,epoch,'train')
# =============================================================================
   
    return record
    pass
@torch.no_grad()
def validation(backbone_model:nn.Module,
                    Hungarian,dataloader:Iterable, 
                    backbone_loss_function:nn.Module,
                    Metric_Logger:callable, epoch:int,num_object:int,
                    training_mode:str,device:torch.device):
# =============================================================================
    record = {}
    record['loss'] = []
    record['category_loss'] = []
    record['acc'] = []
    record['confusion_matrix'] = np.zeros([num_object+1,num_object+1])

    backbone_model.eval()
    backbone_loss_function.eval()

    Metric_Logger.reset_log()
    for idx,(tested,categories) in enumerate(dataloader):

        tested = tested.to(device)
        categories = categories.to(device)
        backbone_result = backbone_model(tested)
             
        
        loss = backbone_loss_function(backbone_result,categories)
        
        category_loss = loss['category_loss'].item()


        record['loss'] +=[category_loss]
        record['category_loss'] += [category_loss]
        record['confusion_matrix'] += confusion_matrix_F(backbone_result,categories,num_object)
        if np.sum(record['confusion_matrix'])==0:
            record["acc"] += [0]
        else:
            record['acc'] += [np.sum(record['confusion_matrix'].diagonal())/np.sum(record['confusion_matrix'])]

# =============================================================================
        Metric_Logger.update(record,epoch,'val')
# =============================================================================
       
    return record
    pass