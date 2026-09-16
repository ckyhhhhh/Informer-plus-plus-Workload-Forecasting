import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')
class Dataset_Workload(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='workload.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dir_path = os.path.join(self.root_path, self.data_path)
        df_raw = self.__get_custome_dataset__(dir_path)
        slice_dct = self.__get_idx__(len(df_raw))

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[slice_dct['train'][0]:slice_dct['train'][1]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

        else:
            data = df_data.values
        cur_begin = slice_dct[self.flag][0]
        cur_end = slice_dct[self.flag][1]
        df_stamp = df_raw[['date']][cur_begin:cur_end]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[cur_begin:cur_end]
        if self.inverse:
            self.data_y = df_data.values[cur_begin:cur_end]
        else:
            self.data_y = data[cur_begin:cur_end]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]

        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



    def __get_custome_dataset__(self, dir_path):
        observed_values = pd.read_csv(dir_path)
        dataset_lgt = len(observed_values)
        df_raw = pd.DataFrame(observed_values, columns=['date', 'd1', 'd2', 'd3', 'd4'])
        df_raw[['d1', 'd2', 'd3', 'd4']] = df_raw[['d1', 'd2', 'd3', 'd4']].astype(float)
        return df_raw

    def __get_idx__(self, dataset_lgt, seed=1):
        num_train = int(dataset_lgt*0.6)
        num_test = int(dataset_lgt*0.2)
        num_vali = dataset_lgt - num_train - num_test
        border1s = [0, num_train-self.seq_len, dataset_lgt-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, dataset_lgt]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        dct = {'train': (border1s[0], border2s[0]), 'val': (border1s[1], border2s[1]), 'test':(border1s[2], border2s[2])}
        return dct
      