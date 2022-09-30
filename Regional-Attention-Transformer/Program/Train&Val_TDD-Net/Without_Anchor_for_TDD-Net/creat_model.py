from torch import nn
import torch
import pdb
from math import sqrt
from torch.nn.parameter import Parameter
import copy
import math
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from convolutional_Attention import CAttention

from utils import masks_to_boxes,bbox_Oxyxy2normxyxy,bbox_xyxy2xcychw
import cv2
import numpy as np
import torchvision
# =============================================================================
# segmentation
dropout_rate = 0.1
num_channel = 256

class creat_segmentation_main_model(nn.Module):
    def __init__(self,backbone_model,segmentation_model):
        super().__init__()
        ly1_model    = list([backbone_model.backbone_model.conv1,
                            backbone_model.backbone_model.max_pool2d,
                            backbone_model.backbone_model.layer1])
       
        
        ly2_model    = list([backbone_model.backbone_model.layer2])
        ly3_model    = list([backbone_model.backbone_model.layer3])
        
        ly4_model  = list([backbone_model.backbone_model.layer4])
        

        
        
        
        
        self.ly1_model    = nn.Sequential(*ly1_model)
        self.ly2_model    = nn.Sequential(*ly2_model)
        self.ly3_model    = nn.Sequential(*ly3_model)
        self.ly4_model    = nn.Sequential(*ly4_model)
        
        self.segmentation_model = segmentation_model

        self._reset_parameters()
    def _reset_parameters(self):
        for n,p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,img):
        N,C,H,W = img.shape

        ly1feature = self.ly1_model(img)
        ly2feature = self.ly2_model(ly1feature)
        ly3feature = self.ly3_model(ly2feature)
        ly4feature = self.ly4_model(ly3feature)
      
        
        proposal_center,bbox,category = self.segmentation_model(ly4feature)
        N = bbox.shape[0]
        result = []
        
        for batch in range(N):
            result += [{"proposal_center":proposal_center[batch],
                        "bbox":bbox[batch],
                        "category":category[batch]}]

        return result
        

class creat_segmentation_model(nn.Module):
    def __init__(self,num_object,device):
        super().__init__()


        self.num_cluster = 100
        
        ly4_num_channel = 512
        self.xy_position_embdding = xy_position_embdding(256)
        
        encoder_layer = TransformerEncoderLayer(ly4_num_channel,1,3,1024,dropout_rate)
        self.Encoder = TransformerEncoder(encoder_layer,1)
            
        decoder_layer = TransformerDecoderLayer(ly4_num_channel,1,3,1024,dropout_rate)
        self.Dncoder = TransformerDecoder(decoder_layer,1)
        
        self.xy_L1 = nn.Linear(ly4_num_channel,2)
        self.wh_L1 = nn.Linear(ly4_num_channel,2)

        
        self.clssify = nn.Linear(ly4_num_channel,7)

        

    def forward(self,g4_inp):

        
        N,C,H,W = g4_inp.shape

        encoder_inp = g4_inp
        encoder_out = self.Encoder(encoder_inp)
    
        querires = g4_inp
        decoder_out = self.Dncoder(querires,
                                   encoder_out).flatten(-2,-1).permute(0,2,1)

        x,y = self.xy_L1(decoder_out).unbind(dim=-1)
        w,h = self.wh_L1(decoder_out).unbind(dim=-1)
        
        x = x.sigmoid()
        y = y.sigmoid()
        w = w.sigmoid()
        h = h.sigmoid()
        bbox = torch.stack([x,y,w,h],dim=-1)
        if torch.any(torch.isnan(bbox)):
            pdb.set_trace()
        
        category = self.clssify(decoder_out)


        position_flatten = torch.arange(H*W,dtype=torch.float32,device=g4_inp.device)[None].repeat(N,1)
        proposal_center = self.get_position(position_flatten, (W,H))
        # =============================================================================

        return proposal_center,bbox,category
    def get_position(self,idx,WH):
        N,num = idx.shape
        max_x = idx%WH[0]
        max_y = idx//WH[0]

        max_Y = max_y.contiguous().view(N,num,1)/WH[1]+1/(WH[1]*2)
        max_X = max_x.contiguous().view(N,num,1)/WH[0]+1/(WH[0]*2)
        XY = torch.cat([max_X,max_Y],dim=-1)
        return XY
    def center_proposal(self,inputs):

        N,C,H,W = inputs.shape
        max_inputs = torch.max(inputs,dim=1)[0].flatten(-2,-1)
        max_score, max_idx = max_inputs.sort(descending=True)
        max_score = max_score[:,:self.num_cluster]
        max_idx = max_idx[:,:self.num_cluster]
        normXY = self.get_position(max_idx,(W,H))
        X,Y = normXY.unbind(-1)
        X = (X * W).to(torch.long).flatten()
        Y = (Y * H).to(torch.long).flatten()
        
        batch_sele = torch.arange(N)[:,None].repeat(1,self.num_cluster).flatten()

        querires = inputs[batch_sele,:,Y,X].contiguous().view(N,self.num_cluster,C).permute(1,0,2)

        return querires,normXY
class xy_position_embdding(nn.Module):
    def __init__(self,feedforward):
        super().__init__()
        self.relu = nn.ReLU()
        self.L1 = nn.Linear(2,feedforward)
        self.L2 = nn.Linear(feedforward,1)
    def forward(self,normXY):

        L1 = self.relu(self.L1(normXY))
        embdding = self.L2(L1)
        return embdding
    

        
class creat_backbone_main_model(nn.Module):
    def __init__(self,backbone_model):
        super().__init__()
        self.backbone_model = backbone_model
        self._reset_parameters()
    def _reset_parameters(self):
        for n,p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self,img):
        feature = self.backbone_model(img)
        return feature
    
'''ResNet-18 Image classfication for cifar-10 with PyTorch 

Author 'Sun-qian'.

'''

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            norm2D(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            norm2D(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride),
                norm2D(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=[7,7], stride=2, padding=3),
            norm2D(64),
            nn.ReLU(),
        )
        self.max_pool2d = nn.MaxPool2d([3,3],stride=2,padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool2d(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)

        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        return out


def ResNet18():

    return ResNet(ResidualBlock,6)


    return model

    
class norm2D(nn.Module):
    def __init__(self ,num_channel,gamma=1, beta=0):
        super().__init__()
        self.gamma = Parameter(torch.full([1,num_channel,1,1],gamma,dtype=torch.float32),requires_grad=True)
        self.beta = Parameter(torch.full([1,num_channel,1,1],beta,dtype=torch.float32),requires_grad=True)
    def forward(self,inputs):
        
        mean = torch.mean(inputs,dim=(-1,-2),keepdim=True)
        std = torch.std(inputs,dim=(-1,-2),keepdim=True)+1e-05
        inputs = (((inputs-mean)/std)*self.gamma)+self.beta
               
        return inputs
    
class norm1D(nn.Module):
    def __init__(self, num_sequence, gamma=1, beta=0):
        super().__init__()
        self.gamma = Parameter(torch.full([1,1,num_sequence],gamma,dtype=torch.float32),requires_grad=True)
        self.beta = Parameter(torch.full([1,1,num_sequence],beta,dtype=torch.float32),requires_grad=True)
    def forward(self,inputs):
        mean = torch.mean(inputs,dim=(-1),keepdim=True)
        std = torch.std(inputs,dim=(-1),keepdim=True)+1e-05
        inputs = (((inputs-mean)/std)*self.gamma)+self.beta

        return inputs
    
class mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs):
        inputs = inputs*(torch.tanh(F.softplus(inputs)))
        return inputs
        pass
class Channel_norm(nn.Module):
    def __init__(self, num_channel, gamma=1, beta=0):
        super().__init__()
        self.gamma = Parameter(torch.full([1,num_channel,1,1],gamma,dtype=torch.float32),requires_grad=True)
        self.beta = Parameter(torch.full([1,num_channel,1,1],beta,dtype=torch.float32),requires_grad=True)
    def forward(self,inputs):
        mean = torch.mean(inputs,dim=(1),keepdim=True)
        std = torch.std(inputs,dim=(1),keepdim=True)+1e-05
        inputs = (((inputs-mean)/std)*self.gamma)+self.beta

        return inputs

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, kernel, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.CBAM = CBAM(d_model,dim_feedforward,dim_feedforward)
        self.self_attn = CAttention(d_model,nhead,kernel=kernel)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # pdb.set_trace()
        src = self.CBAM(src)
        src2 = self.self_attn(src,src)
        src = src + self.dropout1(src2)# residual
        src = self.norm1(src.permute(0,2,3,1))# layer norm
        #經過linear1後再經過dropout再經過linear2輸出
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)# 再次residual
        src = self.norm2(src).permute(0,3,1,2)# 再次layer norm
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        #post跟pre差在最後的layer norm 位置不一樣(post在最後面,pre在最前面)
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []   
        #經過6次decoder並以intermediate記錄下來
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output#.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, kernel, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = CAttention(d_model,nhead,kernel=kernel)
        self.multihead_attn = CAttention(d_model,nhead,kernel=kernel)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        tgt2 = self.self_attn(tgt,tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt.permute(0,2,3,1)).permute(0,3,1,2)
        tgt2 = self.multihead_attn(queries=tgt,
                                   key=memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt.permute(0,2,3,1))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt.permute(0,3,1,2)

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
    
        
class CBAM(nn.Module):
    def __init__(self,num_Channel,convfeedforward=256,feedforward=512):
        super().__init__()

        self.linear1 = nn.Linear(num_Channel,feedforward)
        self.linear2 = nn.Linear(feedforward,feedforward)
        self.linear3 = nn.Linear(feedforward,num_Channel)

        self.spatial_conv1 = nn.Conv2d(2, convfeedforward,[3,3],stride=1,padding=1)
        self.spatial_conv2 = nn.Conv2d(convfeedforward, convfeedforward,[3,3],stride=1,padding=1)
        self.spatial_conv3 = nn.Conv2d(convfeedforward, 1,[3,3],stride=1,padding=1)
    def forward(self,inputs):
        N,C,H,W = inputs.shape
        MP2D_result = torch.amax(inputs,dim=(-2,-1))
        AP2D_result = torch.mean(inputs,dim=(-2,-1))
        CA = torch.stack([MP2D_result,AP2D_result],dim=1)
        CA = self.linear1(CA)
        CA = self.linear2(CA)
        CA = self.linear3(CA)

        CA = torch.sum(CA,dim=1).sigmoid()[:,:,None,None]
        SA_inputs = (CA*inputs)
        
        MP1D_result = torch.max(SA_inputs,dim=1,keepdim=True)[0]
        AP1D_result = torch.mean(SA_inputs,dim=1,keepdim=True)
        SA = torch.cat([MP1D_result,AP1D_result],dim=1)
        SA = self.spatial_conv1(SA)
        SA = self.spatial_conv2(SA)
        SA = self.spatial_conv3(SA).sigmoid()
        
        SA_result = SA*SA_inputs
        
        return SA_result

class SSAM(nn.Module):
    def __init__(self,num_Channel):
        super().__init__()
        self.Weight_q = nn.parameter.Parameter(torch.FloatTensor(1,1,num_Channel,num_Channel),requires_grad=True)
        self.Weight_k = nn.parameter.Parameter(torch.FloatTensor(1,1,num_Channel,num_Channel),requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self,inputs):
        S,N,C = inputs.shape
        Attention_inp = inputs
        q = torch.sum(self.Weight_q * Attention_inp.unsqueeze(-1), dim=-1)
        k = torch.sum(self.Weight_k * Attention_inp.unsqueeze(-1), dim=-1)
        alpha =(q * k).sigmoid().contiguous()
        return Attention_inp*alpha
        
        
class gaussian_product(nn.Module):
    def __init__(self,num_Channel,num_cluster,W_var=0.125,H_var=0.125,scale=1):
        super().__init__()
        self.W_var = nn.parameter.Parameter(torch.full([num_cluster],W_var,dtype=torch.float32),requires_grad=True)
        self.H_var = nn.parameter.Parameter(torch.full([num_cluster],H_var,dtype=torch.float32),requires_grad=True)
        self.scale = nn.parameter.Parameter(torch.full([num_cluster],scale,dtype=torch.float32),requires_grad=True)
        self.num_cluster = num_cluster
        # self.W_var = W_var
        # self.H_var = H_var
        # self.scale = scale

        self.relu = nn.ReLU()
    def forward(self,inputs,inv_cat=False):
        max_inputs = torch.max(inputs,dim=1,keepdim=True)[0]
        eps = torch.finfo(max_inputs.dtype).eps
        N,C,H,W = inputs.shape
        W_var = W*self.W_var.clamp(eps)
        H_var = H*self.H_var.clamp(eps)
        scale = self.scale.repeat(N,1)[:,:,None,None].clamp(eps)
        # print("H_var:\n",self.H_var)
        # print("W_var:\n",self.W_var)
        # print("scale:\n",self.scale)
        # Sscore,Sidx = inputs.sort(descending=True)
        

        max_idx = max_inputs.flatten(-2,-1).sort(descending=True)[1][:,:,:self.num_cluster]
        max_x = max_idx%W
        max_y = max_idx//W
        


        max_Y = max_y.contiguous().view(N,self.num_cluster,1,1)
        max_X = max_x.contiguous().view(N,self.num_cluster,1,1)
        # self.show_anchor(inputs,max_X,max_Y)

        
        
        gussian_map = get_gussian_map(max_inputs,
                                      (max_X,max_Y),
                                      [W_var,H_var],
                                      scale,
                                      device=inputs.device)

        result = (inputs[:,:,None]*gussian_map[:,None,:])

        
        return result
    def max_anchor(self,inputs):
        eps = torch.finfo(inputs.dtype).eps

        result = inputs/(torch.amax(inputs,dim=(-2,-1),keepdim=True).clamp(eps))
        
        return result
    def show_anchor(self,inputs,max_x,max_y):

        norm_inputs = self.max_anchor(inputs).unsqueeze(-1).repeat(1,1,1,1,3).cpu().numpy()
        N,C,H,W,_ = norm_inputs.shape
        max_X = (max_x.cpu().numpy()/40*640).astype("int64")
        max_Y = (max_y.cpu().numpy()/40*640).astype("int64")
        for batch in range(N):
            for channel in range(C):
                show_img = cv2.resize(norm_inputs[batch,channel],(640,640))
                for cluster in range(10):
                    X = max_X[batch,cluster,0,0]
                    Y = max_Y[batch,cluster,0,0]
                    cv2.circle(show_img, [X,Y], 10, [0,0,255], -1)
                cv2.imshow("a",show_img)
                cv2.waitKey(0)
                
        cv2.destroyAllWindows()
        
class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          s: norm of input feature
          m: margin
          cos(theta + m)
          """
    def __init__(self, in_features, out_features, s=10.0, m=0.30):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, label=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        logits = F.linear(F.normalize(input), F.normalize(self.weight))
        if label != None:
            logits = logits.clip(-1.0,1.0)
            
            logits = torch.acos(logits)
            
            target_logits = torch.cos(logits+self.m)
    
            one_hot_label = torch.nn.functional.one_hot(label, num_classes= 7)
            logits = logits * (1 - one_hot_label) + target_logits * one_hot_label
            logits = logits * self.s
    
        return logits


class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i+1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, c, -1)
            if self.pool_type == 'avg_pool':
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, c, -1)

            # 展开、拼接
            if (i == 0):
                SPP = tensor.view(num, c, -1)
            else:
                SPP = torch.cat((SPP, tensor.view(num, c, -1)), -1)
        return SPP

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "softmax":
        return nn.Softmax(dim=0)
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_segmentation_model(args):
    backbone_model = ResNet18().to(args.device)
    backbone_model = creat_backbone_main_model(backbone_model).to(args.device)
    
    segmentation_model = creat_segmentation_model(args.num_object,args.device).to(args.device)
    segmentation_model = creat_segmentation_main_model(backbone_model,segmentation_model).to(args.device)
    return backbone_model,segmentation_model


# =============================================================================
if __name__ == '__main__':
    model = build_segmentation_model('gray', 100, 2048, 256, 6)
    
    tensor = torch.rand([2,1,512,512],dtype=torch.float32)
    
    result = model(tensor)
    print(result['bbox'].requires_grad,result['category'].requires_grad)
