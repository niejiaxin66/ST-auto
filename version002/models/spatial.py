import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from version002.models.transformer import make_model
from version002.tools.spatial_utils import geodistance, caldis, distance_matrix, getA_cosin,getA_corr,getadj,get_adj,scaled_Laplacian
#GCN要先准备好【ST-Tran/stgcn_traffic_prediction/pygcn/models.py】

#GraphConvolution先放进来
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)).float()
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features)).float()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.dtype,self.weight.dtype,self.bias.dtype)
        support = torch.matmul(input, self.weight)
        #print(support.shape)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#GCN本体如下
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        #self.bn = nn.BatchNorm2d(nhid)
 
    def forward(self, x, adj):
        #print(adj.shape)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        #print('after gcn:\n',x)
        return x

class gcnSpatial(nn.Module):
    def __init__(self,dim_in,dim_hid,dim_out,dropout):
        super(gcnSpatial,self).__init__()
        self.spatial = GCN(dim_in,dim_hid,dim_out,dropout)

    def forward(self,path,x_c,A=None,index=None):
        #print('x_c:',x_c)
        N = x_c.shape[-1] #x_c (bs,seq_len,N)
        sx_c = x_c.permute(0,2,1).float() #sx_c(bs,N,seq_len)
        #print('sx',sx_c.shape)
        adj_mx = distance_matrix(path)
        L_tilde = torch.tensor(scaled_Laplacian(adj_mx)).float()
        #adj = getadj(sx_c)
        #print('gcn_adj',adj.shape)
        spatial_c = self.spatial(sx_c[:].cuda(),L_tilde.cuda())
        return  spatial_c



class Spatial(nn.Module):
    def __init__(self,seq_len,k,N,model_d):
        super(Spatial,self).__init__()
        self.spatial = make_model(seq_len,seq_len,N,model_d,spatial=True)
        self.k = k
 
    def forward(self,path,x_c,A=None,index=None):
        '''initial data size
        x_c: bs*closeness*2*N  x_c:bs*seq_len*N
        x:   bs*2*N*closeness  x:bs*N*seq_len
        '''
        '''spatial input
        sx_c: bs*2*N*k*closeness
        tgt: bs*2*N*1*closeness
        '''
        '''spatial output
        sq_c: bs*N*1*closeness
        '''
        bs,seq_len,N = x_c.shape
        x = x_c.permute((0,2,1)).float()
        #print('x',x.shape)
        #calculate the similarity between other nodes
        if A is None:
            A= distance_matrix(path)
            #A.shape=bs,N,N
        #print('A',A.shape)
        #selected top-k node

        sx_c = torch.zeros((bs,N,self.k,seq_len),dtype=torch.float32)
        if index is None:
            index = torch.argsort(A,dim=-1,descending=True)[:,0:self.k]
        # selected_c = []
        for i in range(bs):
          for j in range(N):
            for t in range(self.k):
              sx_c[i,j,t,:] = x[i,index[j,t]]
        #sx_c = torch.cat(selected_c,dim=2).cuda()
        #print('sx_c',sx_c.shape)
        #sx_c:(bs,N,k,closeness)

        tgt = x[:].unsqueeze(dim=-2).cuda()
        #tgt_c = sx_c[:,flow].unsqueeze(-1).cuda()
        #我懂了，sx_c就是相当于储存了k个最相关的地方的时间序列，然后再进行特征提取...
        
       # print('s_tgt',tgt.shape)
        sq_c = self.spatial(sx_c.cuda(), tgt).squeeze(dim=-2)

        return sq_c
