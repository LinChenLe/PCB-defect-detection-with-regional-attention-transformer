import torch
from torch import nn
import pdb
from typing import Iterable
import numpy as np
from utils import smooth_value
from setcriterion import confusion_matrix_F,Conf_confusion_matrix_F
import time

def train_one_epoch(segmentation_model:nn.Module,segmentation_optimizer:torch.optim,
                    Hungarian,dataloader:Iterable, 
                    segmentation_loss_function:nn.Module,
                    Metric_Logger:callable, epoch:int,num_object:int,
                    training_mode:str,device:torch.device):
# =============================================================================
    # box_loss_function = compute_box_loss(False,device=device)
    record = {}
    record['loss'] = []
    # record["GIOULoss"] = []
    record["CIOULoss"] = []
    record["smoothL1loss"] = []
    # record['MSEloss'] = []
    # record["centerloss"] = []
    # record["SAloss"] = []
    # record["confL1loss"] = []
    # record["biasloss"] = []
    record["matchness"] = []
    # record["boxMSEloss"] = []
    # record['segmenMSEloss'] = []
    # record['PPMCCloss'] = []
    # record['focalloss'] = []
    # record['diceloss'] = []
    # record['BCEloss'] = []
    # record['varloss'] = []
    record['categoryloss'] = []
    record['acc'] = []
    record['confusion_matrix'] = np.zeros([num_object+1,num_object+1])
    
    # record['S_acc'] = []
    # record['S_confusion_matrix'] = np.zeros([2,2])
    
    # record['C_acc'] = []
    # record['C_confusion_matrix'] = np.zeros([2,2])
    
    record['blr'] = []
    record['slr'] = []

    # record["conf_acc"] = []
    # record["conf_confusion_matrix"] = np.zeros([2,2])
    segmentation_model.train()
    segmentation_loss_function.train()


        
    Hungarian.eval()

    Metric_Logger.reset_log()
    for idx,(img,tgt) in enumerate(dataloader):
        N,C,H,W = img.shape
        img = img.to(device)


        for batch in range(N):
            tgt[batch]['bbox'] = tgt[batch]['bbox'].to(device)
            tgt[batch]['category'] = tgt[batch]['category'].to(device)
        #20sec
        segmen_result = segmentation_model(img)
        matchness,match_loss, match_idx = Hungarian(segmen_result,tgt)
        # pdb.set_trace()

        loss = segmentation_loss_function(segmen_result,tgt,match_idx,match_loss)
        # boxloss = box_loss_function(segmen_result,tgt,match_idx,True)
        # pdb.set_trace()
        # SAloss = loss["SAloss"].item()
        # MSEloss = loss['MSEloss'].item()
        # centerloss = loss["centerloss"].item()
        # biasloss = loss["biasloss"].item()
        # confL1loss = loss["confL1loss"].item()
        categoryloss = loss['categoryloss'].item()
        # GIOULoss = loss["GIOULoss"].item()
        CIOULoss = loss["CIOULoss"].item()
        smoothL1loss = loss["smoothL1loss"].item()
        # boxMSEloss = boxloss["boxMSEloss"].item()
        # segmenMSEloss = loss["segmenMSEloss"].item()
        # PPMCCloss = loss['PPMCCloss'].item()
        # focalloss = loss['focalloss'].item()
        # diceloss = loss['diceloss'].item()
        # BCEloss = loss['BCEloss'].item()
        # varloss = loss['varloss'].item()
        record['loss'] +=[CIOULoss+smoothL1loss+categoryloss]
        record["CIOULoss"] += [CIOULoss]
        # record["GIOULoss"] += [GIOULoss]
        record["smoothL1loss"] += [smoothL1loss]
        # record['MSEloss'] +=[MSEloss]
        # record["centerloss"]+=[centerloss]
        # record["biasloss"] += [biasloss]
        # record["SAloss"] += [SAloss]
        # record["confL1loss"] += [confL1loss]
        matchness = torch.mean(torch.stack(matchness)).item()
        record["matchness"] += [matchness]
        # record['boxMSEloss'] +=[boxMSEloss]
        # record['segmenMSEloss'] +=[segmenMSEloss*1000]
        record['categoryloss'] +=[categoryloss]
        # record['PPMCCloss'] += [PPMCCloss*10]
        # record['focalloss'] +=[focalloss]
        # record['diceloss'] +=[diceloss]
        # record['BCEloss'] +=[BCEloss*1000]
        # record['varloss'] +=[varloss*1000]
        
        record['confusion_matrix'] = confusion_matrix_F(segmen_result,tgt,num_object,match_idx)
        record['acc'] += [np.sum(np.diag(record['confusion_matrix']))/np.sum(record['confusion_matrix'])]
        
        
        # S_confusion_matrix,C_confusion_matrix = Conf_confusion_matrix_F(segmen_result,tgt,match_idx,match_loss)
        # record['S_confusion_matrix'] = S_confusion_matrix
        # record['S_acc'] += [np.sum(np.diag(record['S_confusion_matrix']))/max(np.sum(record['S_confusion_matrix']),1)]
        
        # record['C_confusion_matrix'] = C_confusion_matrix
        # record['C_acc'] += [np.sum(np.diag(record['C_confusion_matrix']))/max(np.sum(record['C_confusion_matrix']),1)]
        
        segmentation_optimizer.zero_grad()
        (loss['CIOULoss']+loss["smoothL1loss"]+loss['categoryloss']).backward()
        segmentation_optimizer.step()

        record['blr'] +=[segmentation_optimizer.param_groups[0]['lr']]
        record['slr'] +=[segmentation_optimizer.param_groups[1]['lr']]
        
# =============================================================================
        Metric_Logger.update(record,epoch,'train')
# =============================================================================
                    
    return record
    pass
@torch.no_grad()
def validation(segmentation_model:nn.Module,
                    Hungarian,dataloader:Iterable, 
                    segmentation_loss_function:nn.Module,
                    Metric_Logger:callable, epoch:int,num_object:int,
                    training_mode:str,device:torch.device):
# =============================================================================
    # box_loss_function = compute_box_loss(False,device=device)
    record = {}
    record['loss'] = []
    # record["GIOULoss"] = []
    record["CIOULoss"] = []
    record["smoothL1loss"] = []
    # record['MSEloss'] = []
    # record["centerloss"] = []
    # record["SAloss"] = []
    # record["confL1loss"] = []
    # record["biasloss"] = []
    record["matchness"] = []
    # record["boxMSEloss"] = []
    # record['segmenMSEloss'] = []
    # record['PPMCCloss'] = []
    # record['focalloss'] = []
    # record['diceloss'] = []
    # record['BCEloss'] = []
    # record['varloss'] = []
    record['categoryloss'] = []
    record['acc'] = []
    record['confusion_matrix'] = np.zeros([num_object+1,num_object+1])
    
    # record['S_acc'] = []
    # record['S_confusion_matrix'] = np.zeros([2,2])
    
    # record['C_acc'] = []
    # record['C_confusion_matrix'] = np.zeros([2,2])
    


    # record["conf_acc"] = []
    # record["conf_confusion_matrix"] = np.zeros([2,2])
    segmentation_model.eval()
    segmentation_loss_function.eval()


        
    Hungarian.eval()

    Metric_Logger.reset_log()
    for idx,(img,tgt) in enumerate(dataloader):
        N,C,H,W = img.shape
        img = img.to(device)


        for batch in range(N):
            tgt[batch]['bbox'] = tgt[batch]['bbox'].to(device)
            tgt[batch]['category'] = tgt[batch]['category'].to(device)
        #20sec
        segmen_result = segmentation_model(img)
        matchness,match_loss, match_idx = Hungarian(segmen_result,tgt)
        # pdb.set_trace()

        loss = segmentation_loss_function(segmen_result,tgt,match_idx,match_loss)
        # boxloss = box_loss_function(segmen_result,tgt,match_idx,True)
        # pdb.set_trace()
        # SAloss = loss["SAloss"].item()
        # MSEloss = loss['MSEloss'].item()
        # centerloss = loss["centerloss"].item()
        # biasloss = loss["biasloss"].item()
        # confL1loss = loss["confL1loss"].item()
        categoryloss = loss['categoryloss'].item()
        # GIOULoss = loss["GIOULoss"].item()
        CIOULoss = loss["CIOULoss"].item()
        smoothL1loss = loss["smoothL1loss"].item()
        # boxMSEloss = boxloss["boxMSEloss"].item()
        # segmenMSEloss = loss["segmenMSEloss"].item()
        # PPMCCloss = loss['PPMCCloss'].item()
        # focalloss = loss['focalloss'].item()
        # diceloss = loss['diceloss'].item()
        # BCEloss = loss['BCEloss'].item()
        # varloss = loss['varloss'].item()
        record['loss'] +=[CIOULoss+smoothL1loss+categoryloss]
        record["CIOULoss"] += [CIOULoss]
        # record["GIOULoss"] += [GIOULoss]
        record["smoothL1loss"] += [smoothL1loss]
        # record['MSEloss'] +=[MSEloss]
        # record["centerloss"]+=[centerloss]
        # record["biasloss"] += [biasloss]
        # record["SAloss"] += [SAloss]
        # record["confL1loss"] += [confL1loss]
        matchness = torch.mean(torch.stack(matchness)).item()
        record["matchness"] += [matchness]
        # record['boxMSEloss'] +=[boxMSEloss]
        # record['segmenMSEloss'] +=[segmenMSEloss*1000]
        record['categoryloss'] +=[categoryloss]
        # record['PPMCCloss'] += [PPMCCloss*10]
        # record['focalloss'] +=[focalloss]
        # record['diceloss'] +=[diceloss]
        # record['BCEloss'] +=[BCEloss*1000]
        # record['varloss'] +=[varloss*1000]
        
        record['confusion_matrix'] = confusion_matrix_F(segmen_result,tgt,num_object,match_idx)
        record['acc'] += [np.sum(np.diag(record['confusion_matrix']))/np.sum(record['confusion_matrix'])]
        
        
        # S_confusion_matrix,C_confusion_matrix = Conf_confusion_matrix_F(segmen_result,tgt,match_idx,match_loss)
        # record['S_confusion_matrix'] = S_confusion_matrix
        # record['S_acc'] += [np.sum(np.diag(record['S_confusion_matrix']))/max(np.sum(record['S_confusion_matrix']),1)]
        
        # record['C_confusion_matrix'] = C_confusion_matrix
        # record['C_acc'] += [np.sum(np.diag(record['C_confusion_matrix']))/max(np.sum(record['C_confusion_matrix']),1)]
        



        
# =============================================================================
        Metric_Logger.update(record,epoch,'val')
# =============================================================================
                    
    return record
    pass