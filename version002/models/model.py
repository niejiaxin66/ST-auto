import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
from version002.models.spatial import GraphConvolution
from version002.models.spatial import gcnSpatial,Spatial
from version002.tools.spatial_utils import getadj,getA_cosin,getA_corr
from version002.models.autoformer import autoformer
from version002.tools.spatial_utils import geodistance, caldis, distance_matrix, getA_cosin,getA_corr,getadj,get_adj,scaled_Laplacian
class Fusion(nn.Module):
    def __init__(self,dim_in):
        super(Fusion,self).__init__()
        self.weight2 = nn.Linear(dim_in*2,dim_in)

    def forward(self,x1,x2=None):
        if(x2 is not None):
            out = self.weight2(torch.cat([x1,x2],dim=-1))
        else:
            out = x1
        return out

'''
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

'''

class T_STGCN(nn.Module):
    def __init__(self,path,seq_len, external_size, N, k,spatial,s_model_d,dim_hid=16, drop_rate=0.1):
    #多加了spatial,s_model_d
        super(T_STGCN,self).__init__()
        
        if(spatial=='gcn'):
            self.spatial = gcnSpatial(seq_len,dim_hid,seq_len,dropout=0.1)
        else:
            self.spatial = Spatial(seq_len,k,N,s_model_d)
            
        #self.spatial = Spatial(path,seq_len,dim_hid,seq_len,dropout=0.1)
        #self.spatial_f = Spatial(path,seq_len,dim_hid,seq_len,dropout=0.1)
        '''if(spatial=='gcn'):
            self.spatial_f = gcnSpatial(seq_len,dim_hid,seq_len,dropout=0.1)
        else:
            self.spatial_f = Spatial(seq_len,k,N,s_model_d)'''
            
        self.autoformer=autoformer
        self.k = k
        self.fusion=Fusion(seq_len)


    def forward(self,path,x_enc, x_mark_enc, x_dec, x_mark_dec,opt):
        '''initial data size
        x_enc: bs*seq_len*N
        '''
        '''spatial output
        sx_c: bs*N*seq_len
        '''
        '''temporal output
        sq_c: bs*seq_len*N
        '''
        '''fused output
        bs*closeness*N
        '''
        #x_c=batch_x
        #x_t=batch_x
        #x_t=x_t[:,:,:,None]  #bs*seq_len*N*1
        bs = len(x_enc)
        N = x_enc.shape[-1]
        seq_len = x_enc.shape[1]
        x_spatial = None
        sq_t = None
        #print('x_c\n',x_c)

        #get adj
        adj= distance_matrix(path)
        adj=torch.from_numpy(adj)
        index = torch.argsort(adj,dim=-1,descending=True)[:,0:self.k]
        #if(s):
            #spatial
        x_spatial = self.spatial(path,x_enc,adj,index)
            #print('spatial:',x_spatial[0])

        #temporal
        #if(c):
            # encoder - decoder
        if opt.use_amp:
            with torch.cuda.amp.autocast():
                if opt.output_attention:
                    outputs = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec,opt)[0].cuda()
                else:
                    outputs = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec,opt).cuda()
        else:
            if opt.output_attention:
                outputs = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec,opt)[0].cuda()

            else:
                outputs = self.autoformer(x_enc, x_mark_enc, x_dec, x_mark_dec,opt).cuda()

           # f_dim = -1 if self.opt.features == 'MS' else 0

        #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        pred=outputs(x_enc,x_mark_enc,x_dec,x_mark_dec,opt).detach().cpu().numpy()
        #batch_y = batch_y.detach().cpu().numpy()

        # outputs.detach().cpu().numpy()  # .squeeze()
        #true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
        pred=torch.from_numpy(pred)
        #sq_t = F.sigmoid(pred.permute(0,2,1)) #bs*N*seq_len
        x_temporal = pred.cuda() #(bs,seq_len,N)
        
        #if(FS):
        #x_temporal,_ = self.spatial_f(x_enc,x_temporal.transpose(1,2).unsqueeze(-2).unsqueeze(1),opt.mode,adj,index)

        #fusion
        pred = self.fusion(x_temporal.permute(0,2,1),x_spatial)
        return pred.transpose(1,2)
