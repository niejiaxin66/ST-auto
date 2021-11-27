import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import pandas as pd
from scipy.spatial.distance import pdist
from scipy import spatial as spt
from math import radians, cos,sin,asin,sqrt



# 定义通过经纬度计算直线距离的函数
def geodistance(lng1,lat1,lng2,lat2):
    if lng1 == None or lng2 == None or lat1 == None or lat2 == None:
        return 999999

    lng1 = float(lng1) ; lng2 = float(lng2) ; lat1 = float(lat1) ; lat2 = float(lat2)
    lng1,lat1,lng2,lat2 = map(radians,[lng1,lat1,lng2,lat2])
    dlon = lng2-lng1
    dlat = lat2-lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance = 2* asin(sqrt(a))*6371*1000
    distance = round(distance/1000,3)
    return distance
def caldis(u,v):
    # 计算输入矩阵中两个向量的距离
    return geodistance(u[0],u[1],v[0],v[1])

def distance_matrix(path):
    # 将cam转化为向量的形式
    X=pd.read_excel(path,sheet_name=3)
    X =X.values
    X_mat = spt.distance.squareform(pdist(X , metric=caldis))
    return X_mat




def get_adj(nums):
    A = np.zeros((int(nums), int(nums)), dtype = np.float32)
    stride = int(np.sqrt(nums))
    for i in range(nums):
        A[i][i]=1.0
        if not i-1<0:
            A[i][i-1]=1.0
        if not i+1>nums-1:
            A[i][i+1] = 1.0
        if not i-stride<0:
            A[i][i-stride] = 1.0
        if not i+stride>nums-1:
            A[i][i+stride] = 1.0
    #print(A)
    return A

def scaled_Laplacian(W):

    
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis = 1))
    
    L = D - W
    
    lambda_max = eigs(L, k = 1, which = 'LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])





def getxy(x,m):
    index = torch.zeros(2,dtype=torch.float32)
    index[0] = x//m
    index[1] = x%m
    return index

def getD(m):
    D = torch.ones((m,m),dtype=torch.float32)*1.5
    for i in range(m):
        for j in range(i):
            D[i,j] = 1/torch.norm((getxy(i,m)-getxy(j,m)),2)
            D[j,i] = D[i,j]
    return D


def getA_cosin(x):
    #(bs,flow,N,c) = x.shape
    (bs,N,c)=x.shape
    #x = x.transpose(1,2).contiguous().view((bs,N,c))

    normed = torch.norm(x,2,dim=-1).unsqueeze(-1)
    print(normed.shape)
    tnormed = normed.transpose(1,2)
    A = x.matmul(x.transpose(1,2))/normed.matmul(tnormed)
    return F.softmax(A,dim=-1)


def getA_corr(x):
    #(bs,flow,N,c) = x.shape
    (bs,N,c)=x.shape
    #x = x.transpose(1,2).contiguous().view((bs,N,c))
    A = torch.zeros((bs,N,N),dtype=torch.float32,requires_grad=False)
    for i in range(bs):
        A[i] = torch.from_numpy(np.absolute(np.corrcoef(x[i].cuda().data.cpu().numpy())))
    for j in range(N):
        A[:,j,j] = -1e9

    return F.softmax(A.reshape(bs,1,-1),dim=-1).reshape(bs,N,N)

def getadj(x):
    #(bs,flow,N,c) = x.shape
    (bs,N,c)=x.shape
    #x = x.transpose(1,2).contiguous().view((bs,N,c)).numpy()
    
    A = np.zeros((bs,N,N),dtype=np.float32)
    for i in range(bs):
        A[i] = np.absolute(np.corrcoef(x[i]))
        A[i] = scaled_Laplacian(A[i])
        D = np.array(np.sum(A[i],axis=-1))
        D = np.matrix(np.diag(D))
        A[i] = D**-1*A[i]
        A[i][np.isnan(A[i])] = 0.
    print(A)
    return torch.from_numpy(A).cuda()


def c_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def p_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.zeros(np.ones(attn_shape)).astype('uint8')
    subsequent_mask[:,:,0] = 1
    return torch.from_numpy(subsequent_mask) == 0
