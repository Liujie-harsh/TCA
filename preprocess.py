import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class PowerConsumptionDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 0]  # 只预测Global_active_power
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def preprocess_data(data_path, seq_len=96, pred_len=24):
    # 读取数据
    df = pd.read_csv(data_path, sep=';', low_memory=False, na_values='?')
    
    # 合并Date和Time为datetime列并设为索引
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('datetime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # 选择特征列
    feature_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df = df[feature_columns]
    
    # 处理缺失值
    df = df.dropna()
    
    # 数据类型转换
    df = df.astype(float)
    
    # 划分训练/验证/测试集
    total_len = len(df)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    
    train_data = df[:train_len]
    val_data = df[train_len:train_len+val_len]
    test_data = df[train_len+val_len:]
    
    # 标准化
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)
    
    # 创建数据集
    train_dataset = PowerConsumptionDataset(train_data_scaled, seq_len, pred_len)
    val_dataset = PowerConsumptionDataset(val_data_scaled, seq_len, pred_len)
    test_dataset = PowerConsumptionDataset(test_data_scaled, seq_len, pred_len)
    
    return train_dataset, val_dataset, test_dataset, scaler

if __name__ == '__main__':
    data_path = '/home/lj/Documents/TCN/data/3/household_power_consumption.txt'
    train_dataset, val_dataset, test_dataset, scaler = preprocess_data(data_path)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    xs=[]
    ys=[]

    for i in range(len(test_dataset)):
        x, y = test_dataset[i]
        xs.append(x)
        ys.append(y)
    x=torch.stack(xs)
    y=torch.stack(ys)

    print(x.shape)
    print(y.shape)