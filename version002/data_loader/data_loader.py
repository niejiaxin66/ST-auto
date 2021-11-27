import numpy as np
from version002.tools.MinMaxNorm import MinMaxNorm01
from version002.tools.time_features import time_features
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def load_data(path, test_size,group_type,split,seq_len,label_len,pred_len,batch_size):

    #test_size=opt.test_size
    #path='/content/drive/MyDrive/Covid19/Alabama_covid_19_confirmed_us.xlsx'
    df_raw=pd.read_excel(path,sheet_name=2) #dataframe(66,670)
    df_raw=df_raw.T #dataframe(670,66)
    df_data=np.array(df_raw)[:,:,np.newaxis] #ndarray(670,66,1)
    
    data_train = df_data[:-test_size]  #ndarray:data_train ndarray with shape(655,66,1)
    mmn=MinMaxNorm01()
    mmn.fit(data_train)
    data_all = [df_data]
    data_all_mmn=[]
    for data in data_all:
        data_all_mmn.append(mmn.transform(data))


    data=data_all_mmn[0][:,:,0] #ndarray with shape (670,66)


    df_stamp=(pd.read_excel(path,sheet_name=0)).columns[11:] # 670items
    
    data_stamp=[]
    for i in range(len(df_stamp)):
        stamp=time_features(df_stamp[i])
        data_stamp.append(stamp)
    data_stamp=np.array(data_stamp) #ndarray with shape(670,3)
    df_stamp=np.array(df_stamp)[:,np.newaxis] #ndarray with shape(670,1)
    #这里的data_type和len_test,split都可以放进parse里
    border_train=int(np.floor((len(data)-test_size)*split)) #11664
    #我的思路是：直接把数据集先分好，然后再送下来进行进一步处理
    data_dict={
        'all':[data,df_stamp,data_stamp],
        'train':[data[:border_train],df_stamp[:border_train],data_stamp[:border_train]],
        'valid':[data[border_train:-test_size],df_stamp[border_train:-test_size],data_stamp[border_train:-test_size]],
        'test':[data[-test_size:],df_stamp[-test_size:],data_stamp[-test_size:]],
    }
    #Data_sliced=data_dict['All']

    class Dataset_pems04_h():
        def __init__(self, size=None,features='M',data_type='train'):
            # size [seq_len, label_len, pred_len]
            # info
            if size == None:
                self.seq_len = 24 * 4 * 4
                self.label_len = 24 * 4
                self.pred_len = 24 * 4
            else:
                self.seq_len = size[0]
                self.label_len = size[1]
                self.pred_len = size[2]
            # init

            self.features = features
            #self.freq = freq
            self.data_type=data_type
            self.__read_data__()

        def __read_data__(self):
            Data_sliced=data_dict[self.data_type]
            data=Data_sliced[0]
            df_stamp=Data_sliced[1]
            data_stamp=Data_sliced[2]
            self.data_x=data
            self.data_y=data
            self.data_stamp=data_stamp
        def __getitem__(self,index):
            s_begin=index
            s_end=s_begin+self.seq_len
            r_begin=s_end-self.label_len
            r_end=r_begin+self.label_len+self.pred_len

            seq_x=self.data_x[s_begin:s_end]
            seq_y=self.data_y[r_begin:r_end]
            seq_x_mark=self.data_stamp[s_begin:s_end]
            seq_y_mark=self.data_stamp[r_begin:r_end]
            #这是不是说明，seq_x和seq_y一个是用来预测，一个是用来放真实值的？然后seq_x_mark和seq_y_mark是用来放时间特征的
            return seq_x,seq_y,seq_x_mark,seq_y_mark

        def __len__(self):
            return len(self.data_x) - self.seq_len - self.pred_len + 1

          

    train_data_set=Dataset_pems04_h(size=[seq_len, label_len, pred_len],features='M',data_type='train')
    valid_data_set=Dataset_pems04_h(size=[seq_len, label_len, pred_len],features='M',data_type='valid')
    test_data_set=Dataset_pems04_h(size=[seq_len, label_len, pred_len],features='M',data_type='test')

    train_shuffle_flag = True
    train_drop_last = True

    valid_shuffle_flag=True
    valid_drop_last=True

    test_shuffle_flag=False
    test_drop_last=True
    #在ST-Tran中还有一个SubsetRandomSampler的过程，这里没加
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=batch_size,
        shuffle=train_shuffle_flag,
        #num_workers=10,
        drop_last=train_drop_last) #DataLoader with 358 itmes

    valid_data_loader = DataLoader(
        valid_data_set,
        batch_size=batch_size,
        shuffle=valid_shuffle_flag,
        #num_workers=10,
        drop_last=valid_drop_last) #DataLoader with 34 itmes

    test_data_loader = DataLoader(
        test_data_set,
        batch_size=batch_size,
        shuffle=test_shuffle_flag,
        #num_workers=10,
        drop_last=test_drop_last) #DataLoader with 3 itmes
    
    return train_data_loader,valid_data_loader,test_data_loader,mmn



