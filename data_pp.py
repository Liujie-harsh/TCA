
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle

def load_and_preprocess_data(file_path=None, window_size=30, train_ratio=0.7, val_ratio=0.15):
    if file_path is None:
        file_path = '/home/lj/Documents/TCN/data/3/household_power_consumption.txt'
    
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, sep=';', na_values=['?'], low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件未找到，请检查路径：{file_path}")
    
    # 2. 处理时间列并设为索引
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    
    # 填充/删除缺失值
    df.dropna(inplace=True)
    
    # 3. 按小时重采样（求均值）
    df_hourly = df.resample('h').mean().dropna()
    # 提取时间信息（利用Datetime索引）
    df_hourly['hour'] = df_hourly.index.hour
    df_hourly['dayofweek'] = df_hourly.index.dayofweek  # 0=周一, 6=周日

    # # 周期性编码（避免23点与0点数值断裂）
    # df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
    # df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)
    # df_hourly['dow_sin'] = np.sin(2 * np.pi * df_hourly['dayofweek'] / 7)
    # df_hourly['dow_cos'] = np.cos(2 * np.pi * df_hourly['dayofweek'] / 7)

    # # 可选：是否周末（二值特征，增强周末模式识别）
    # df_hourly['is_weekend'] = (df_hourly['dayofweek'] >= 5).astype(float)

    # 清理临时列
    df_hourly.drop(columns=['hour', 'dayofweek'], inplace=True)

    # 4. 特征与目标分离
    target_col = 'Global_active_power'
    feature_cols = [col for col in df_hourly.columns if col != target_col]
    
    features = df_hourly[feature_cols].values
    target = df_hourly[[target_col]].values
    
    # 5. 数据标准化
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    features_scaled = scaler_X.fit_transform(features)
    target_scaled = scaler_Y.fit_transform(target)
    
    # 6. 构造滑动窗口
    def create_sliding_windows(X, y, window_size):
        Xs, ys = [], []
        for i in range(len(X) - window_size):
            Xs.append(X[i : i + window_size])
            ys.append(y[i + window_size])
        return np.array(Xs), np.array(ys)
    
    X_seq, Y_seq = create_sliding_windows(features_scaled, target_scaled, window_size=window_size)
    
    # 7. 数据集划分
    total_samples = len(X_seq)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X_seq[:train_end], Y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], Y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], Y_seq[val_end:]

    os.makedirs('/home/lj/Documents/TCN/scaler', exist_ok=True)
    scaler_X_path = os.path.join('/home/lj/Documents/TCN/scaler', 'scaler_X.pkl')
    scaler_Y_path = os.path.join('/home/lj/Documents/TCN/scaler', 'scaler_Y.pkl')
    
    # 保存scaler_X
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    
    # 保存scaler_Y
    with open(scaler_Y_path, 'wb') as f:
        pickle.dump(scaler_Y, f)
    
    print(f"标准化器已保存：")
    print(f"- scaler_X: {scaler_X_path}")
    print(f"- scaler_Y: {scaler_Y_path}")
    # 打印数据信息（可选，方便调试）
    print("="*50)
    print(f"数据预处理完成！")
    print(f"特征列: {feature_cols}")
    print(f"训练集 X: {X_train.shape}, Y: {y_train.shape}")
    print(f"验证集 X: {X_val.shape}, Y: {y_val.shape}")
    print(f"测试集 X: {X_test.shape}, Y: {y_test.shape}")
    print("="*50)
    
    # 返回所有需要的变量
    return (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, target_col)

if __name__ == "__main__":
    file_path = '/home/lj/Documents/TCN/data/3/household_power_consumption.txt'
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, target_col = load_and_preprocess_data(file_path)
