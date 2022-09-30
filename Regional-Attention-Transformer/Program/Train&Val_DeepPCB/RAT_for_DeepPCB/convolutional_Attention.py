import torch
from torch import nn
import pdb
class CAttention(nn.Module):
    def __init__(self,num_channel,head,kernel=3):
        super().__init__()

        self.position_encoding = nn.parameter.Parameter(torch.FloatTensor([[kernel**2]]))

        self.attention = nn.MultiheadAttention(num_channel, head)
        self.kernel = kernel
        self.num_channel = num_channel


    def forward(self,queries,key):
        center = (self.kernel-1)//2
        
        N,C,H,W = queries.shape
        
        queries = queries.contiguous().view(N,self.num_channel,-1)
        
        key = torch.nn.functional.pad(key, (center,center,center,center), mode='constant', value=0.0)
        
        patches = key.unfold(dimension=2, size=self.kernel, step=1)
        patches = patches.unfold(dimension=3, size=self.kernel, step=1)
        conv_map = patches.contiguous().view(N,C,-1,self.kernel,self.kernel).flatten(-2,-1)

        conv_map = conv_map+self.position_encoding

        
        result = []
        for batch in range(N):
            batch_querires = queries[batch].unsqueeze(0).permute(0,2,1)
            feature = conv_map[batch].permute(2,1,0)
            result += [self.attention(batch_querires,feature,feature)[0]]

        result = torch.cat(result,dim=0).permute(0,2,1).contiguous().view(N,C,H,W)

        
        return result


    
    def position_encoding_fn(self,parameter,length,idx):

        position_encoding = parameter.repeat(length)
        a = position_encoding[idx:].cumsum(dim=-1)
        b = position_encoding[:idx+1].cumsum(dim=-1)[1:]
        b = b.flip(dims=[-1])
        position_encoding = torch.cat([b,a],dim=-1)
        maxV = torch.max(position_encoding)
        position_encoding = torch.abs(maxV-position_encoding)/(maxV-parameter)
        return position_encoding

    def grid_position_encoding_fn(self,parameter,shape,coord):
        x = self.position_encoding_fn(parameter,shape[0],coord[0])
        y = self.position_encoding_fn(parameter,shape[1],coord[1])
        x = x.unsqueeze(0).repeat((shape[1],1))
        y = y.unsqueeze(-1).repeat(1,shape[0])
        return (x+y)
    
    
# attention = s_attention(256,7).to('cuda')
# attention(torch.rand(4,256,40,40).to('cuda'))