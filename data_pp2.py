
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pickle
from sklearn.model_selection import train_test_split

# files = ['file1.csv', 'file2.csv', ...]  # 替换为实际文件名
# for f in files:
#     df = pd.read_csv(f, nrows=5)
#     print(f"\n=== {f} ===")
#     print(f"列数: {len(df.columns)}")
#     print(df.columns.tolist())
#     print(df.head(2))


# 文件路径
data_dir = '/home/lj/Documents/TCN/data/2/'
files = {
    'Floor1': '2018Floor1.csv',
    'Floor2': '2018Floor2.csv',
    'Floor3': '2018Floor3.csv',
    'Floor4': '2018Floor4.csv',
    'Floor5': '2018Floor5.csv',
    'Floor6': '2018Floor6.csv',
    'Floor7': '2018Floor7.csv'
}

def process_floor1(df):
    """处理Floor1：只有z1-z4的电表数据"""
    result = {'Date': df['Date']}
    
    # 映射规则
    mappings = {
        'z1_Light(kW)': ('z1', 'light_power'),
        'z1_Plug(kW)': ('z1', 'plug_power'),
        'z2_AC1(kW)': ('z2', 'ac_power'),
        'z2_AC2(kW)': ('z2', 'ac_power'),
        'z2_AC3(kW)': ('z2', 'ac_power'),
        'z2_AC4(kW)': ('z2', 'ac_power'),
        'z2_Light(kW)': ('z2', 'light_power'),
        'z2_Plug(kW)': ('z2', 'plug_power'),
        'z3_Light(kW)': ('z3', 'light_power'),
        'z3_Plug(kW)': ('z3', 'plug_power'),
        'z4_Light(kW)': ('z4', 'light_power')
    }
    
    # 聚合空调功率（z2有4台AC）
    df['z2_ac_power'] = df[['z2_AC1(kW)', 'z2_AC2(kW)', 'z2_AC3(kW)', 'z2_AC4(kW)']].sum(axis=1)
    
    # 构建结果
    for col, (zone, var) in mappings.items():
        if col in df.columns:
            col_name = f'{zone}_{var}'
            if col_name not in result:
                result[col_name] = df[col]
            else:
                # 累加（如多个空调列）
                result[col_name] += df[col]
    
    # 添加z2_ac_power
    result['z2_ac_power'] = df['z2_ac_power']
    
    return pd.DataFrame(result)

def process_floor2(df):
    """处理Floor2：z2有14台空调，有传感器"""
    result = {'Date': df['Date']}
    
    # 处理z1
    result['z1_ac_power'] = df['z1_AC1(kW)']  # z1只有1台AC
    result['z1_light_power'] = df['z1_Light(kW)']
    result['z1_plug_power'] = df['z1_Plug(kW)']
    result['z1_temperature'] = df['z1_S1(degC)']
    result['z1_humidity'] = df['z1_S1(RH%)']
    result['z1_illuminance'] = df['z1_S1(lux)']
    
    # 处理z2：14台空调求和
    ac_cols = [f'z2_AC{i}(kW)' for i in range(1, 15)]
    result['z2_ac_power'] = df[ac_cols].sum(axis=1)
    result['z2_light_power'] = df['z2_Light(kW)']
    result['z2_plug_power'] = df['z2_Plug(kW)']
    result['z2_temperature'] = df['z2_S1(degC)']
    result['z2_humidity'] = df['z2_S1(RH%)']
    result['z2_illuminance'] = df['z2_S1(lux)']
    
    # 处理z3
    result['z3_light_power'] = df['z3_Light(kW)']
    result['z3_plug_power'] = df['z3_Plug(kW)']
    result['z3_temperature'] = df['z3_S1(degC)']
    result['z3_humidity'] = df['z3_S1(RH%)']
    result['z3_illuminance'] = df['z3_S1(lux)']
    
    # 处理z4
    result['z4_ac_power'] = df['z4_AC1(kW)']  # z4只有1台AC
    result['z4_light_power'] = df['z4_Light(kW)']
    result['z4_plug_power'] = df['z4_Plug(kW)']
    result['z4_temperature'] = df['z4_S1(degC)']
    result['z4_humidity'] = df['z4_S1(RH%)']
    result['z4_illuminance'] = df['z4_S1(lux)']
    
    return pd.DataFrame(result)

def process_floor_standard(df, floor_num):
    """处理Floor3-7：完全匹配真实数据结构"""
    result = {'Date': df['Date']}
    
    for zone in ['z1', 'z2', 'z3', 'z4', 'z5']:
        # ===================== 空调功率（完全按真实结构）=====================
        if zone == 'z1':
            # 特殊处理：Floor6 的 z1 只有 1 台空调，其他楼层 z1 有 4 台
            if floor_num == 6:
                ac_col = 'z1_AC1(kW)'
                result[f'{zone}_ac_power'] = df[ac_col] if ac_col in df.columns else 0
            else:
                ac_cols = [f'z1_AC{i}(kW)' for i in range(1, 5)]
                result[f'{zone}_ac_power'] = df[ac_cols].sum(axis=1) if all(c in df.columns for c in ac_cols) else 0

        elif zone == 'z2':
            # 所有楼层 z2 只有 1 台空调
            ac_col = 'z2_AC1(kW)'
            result[f'{zone}_ac_power'] = df[ac_col] if ac_col in df.columns else 0

        elif zone == 'z3':
            # z3 无空调
            result[f'{zone}_ac_power'] = 0

        elif zone == 'z4':
            # z4 所有楼层都有 4 台空调
            ac_cols = [f'z4_AC{i}(kW)' for i in range(1, 5)]
            result[f'{zone}_ac_power'] = df[ac_cols].sum(axis=1) if all(c in df.columns for c in ac_cols) else 0

        elif zone == 'z5':
            # z5 所有楼层 1 台空调
            ac_col = 'z5_AC1(kW)'
            result[f'{zone}_ac_power'] = df[ac_col] if ac_col in df.columns else 0

        # ===================== 照明 & 插座（全部正确，无需修改）=====================
        light_col = f'{zone}_Light(kW)'
        plug_col = f'{zone}_Plug(kW)'
        result[f'{zone}_light_power'] = df[light_col] if light_col in df.columns else 0
        result[f'{zone}_plug_power'] = df[plug_col] if plug_col in df.columns else 0

        # ===================== 传感器（z3 没有，正确）=====================
        if zone != 'z3':
            temp_col = f'{zone}_S1(degC)'
            humid_col = f'{zone}_S1(RH%)'
            lux_col = f'{zone}_S1(lux)'
            
            result[f'{zone}_temperature'] = df[temp_col] if temp_col in df.columns else np.nan
            result[f'{zone}_humidity'] = df[humid_col] if humid_col in df.columns else np.nan
            result[f'{zone}_illuminance'] = df[lux_col] if lux_col in df.columns else np.nan
        else:
            result[f'{zone}_temperature'] = np.nan
            result[f'{zone}_humidity'] = np.nan
            result[f'{zone}_illuminance'] = np.nan

    return pd.DataFrame(result)

def process_data2():
    # 主处理流程
    all_data = []
    print("开始处理各楼层数据...")

    for floor_name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        floor_num = int(floor_name.replace('Floor', ''))
        
        print(f"\n处理 {floor_name}...")
        if floor_name == 'Floor1':
            processed = process_floor1(df)
        elif floor_name == 'Floor2':
            processed = process_floor2(df)
        else:
            processed = process_floor_standard(df, floor_num)  # 传入整数
        
        processed['floor'] = floor_num
        all_data.append(processed)
        print(f"  {floor_name} 处理后列数: {len(processed.columns)}")

    # 合并所有数据
    print("\n合并所有楼层数据...")
    merged_df = pd.concat(all_data, axis=0, ignore_index=True)

    # 按时间排序
    merged_df.sort_values('Date', inplace=True)

    # 去除重复时间戳（保留第一条）
    merged_df.drop_duplicates(subset=['Date', 'floor'], keep='first', inplace=True)

    print(f"\n合并完成！")
    print(f"总行数: {len(merged_df)}")
    print(f"时间范围: {merged_df['Date'].min()} 到 {merged_df['Date'].max()}")
    print(f"\n数据列:\n{merged_df.columns.tolist()}")

    # 查看数据统计
    print(f"\n数据统计:")
    print(merged_df.describe())

    # 保存处理后的数据
    output_path = '/home/lj/Documents/TCN/data/2/datapp/processed_building_data.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\n数据已保存至: {output_path}")

    # 可选：按区域保存单独文件
    for zone in ['z1', 'z2', 'z3', 'z4', 'z5']:
        zone_cols = ['Date', 'floor'] + [col for col in merged_df.columns if col.startswith(zone)]
        zone_df = merged_df[zone_cols]
        zone_path = f'/home/lj/Documents/TCN/data/2/zone/zone_{zone}_data.csv'
        zone_df.to_csv(zone_path, index=False)
        print(f"区域 {zone} 数据已保存: {zone_path}")

    print("\n数据处理完成！")

def split_by_floor_temporal(df, train_ratio=0.7, val_ratio=0.15,
                            target_col='z1_ac_power',
                            feature_cols=None,
                            window_size=None,
                            standardize=True):
    """
    为每个楼层独立按时间顺序划分训练/验证/测试集，并合并结果。

    参数：
        df: DataFrame，必须包含 'Date', 'floor' 列，以及所有特征列和目标列。
        train_ratio: 训练集比例（基于每个楼层的时间序列长度）。
        val_ratio: 验证集比例（测试集 = 1 - train_ratio - val_ratio）。
        target_col: 目标列名称。
        feature_cols: 特征列名称列表，若为 None 则自动使用除 Date, floor, target_col 外的所有列。
        window_size: 若指定，则构造滑动窗口样本；否则返回原始样本（每个时间步一个样本）。
        standardize: 是否对特征和目标进行标准化（使用训练集的均值和标准差）。

    返回：
        X_train, y_train, X_val, y_val, X_test, y_test: 若 window_size 不为 None，则为窗口样本；
                                                        否则为原始样本矩阵。
        scaler_X, scaler_Y: 标准化器（若 standardize=True），否则 None。
        feature_cols: 使用的特征列。
    """
    # 1. 确定特征列
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['Date', 'floor', target_col]]
    
    # 2. 按楼层分组处理
    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    test_X_list, test_y_list = [], []
    
    for _, group in df.groupby('floor'):
        # 确保按时间排序
        group = group.sort_values('Date').reset_index(drop=True)
        
        # 提取特征和目标
        X = group[feature_cols].values
        y = group[target_col].values.reshape(-1, 1)
        
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 时间顺序划分
        X_train_floor, y_train_floor = X[:train_end], y[:train_end]
        X_val_floor, y_val_floor = X[train_end:val_end], y[train_end:val_end]
        X_test_floor, y_test_floor = X[val_end:], y[val_end:]
        
        # 构造滑动窗口（如果需要）
        if window_size is not None:
            def create_windows(X, y, window_size):
                Xs, ys = [], []
                for i in range(len(X) - window_size):
                    Xs.append(X[i:i+window_size])
                    ys.append(y[i+window_size])
                return np.array(Xs), np.array(ys)
            
            if len(X_train_floor) > window_size:
                X_train_floor, y_train_floor = create_windows(X_train_floor, y_train_floor, window_size)
                X_val_floor, y_val_floor = create_windows(X_val_floor, y_val_floor, window_size)
                X_test_floor, y_test_floor = create_windows(X_test_floor, y_test_floor, window_size)
            else:
                # 该楼层样本不足，跳过
                continue
        
        # 收集
        train_X_list.append(X_train_floor)
        train_y_list.append(y_train_floor)
        val_X_list.append(X_val_floor)
        val_y_list.append(y_val_floor)
        test_X_list.append(X_test_floor)
        test_y_list.append(y_test_floor)
    
    # 3. 合并所有楼层的对应部分
    X_train = np.concatenate(train_X_list, axis=0) if train_X_list else np.array([])
    y_train = np.concatenate(train_y_list, axis=0) if train_y_list else np.array([])
    X_val = np.concatenate(val_X_list, axis=0) if val_X_list else np.array([])
    y_val = np.concatenate(val_y_list, axis=0) if val_y_list else np.array([])
    X_test = np.concatenate(test_X_list, axis=0) if test_X_list else np.array([])
    y_test = np.concatenate(test_y_list, axis=0) if test_y_list else np.array([])
    
    # 4. 标准化（仅使用训练集）
    scaler_X, scaler_Y = None, None
    if standardize and len(X_train) > 0:
        # 特征标准化
        scaler_X = StandardScaler()
        # 如果 X_train 是 3D (样本, 时间步, 特征)，需要 reshape 为 2D 拟合，然后转换回去
        if X_train.ndim == 3:
            n_samples, n_steps, n_features = X_train.shape
            X_train_2d = X_train.reshape(-1, n_features)
            X_train_2d_scaled = scaler_X.fit_transform(X_train_2d)
            X_train = X_train_2d_scaled.reshape(n_samples, n_steps, n_features)
            
            # 对验证集和测试集做相同变换
            if len(X_val) > 0:
                X_val_2d = X_val.reshape(-1, n_features)
                X_val = scaler_X.transform(X_val_2d).reshape(X_val.shape)
            if len(X_test) > 0:
                X_test_2d = X_test.reshape(-1, n_features)
                X_test = scaler_X.transform(X_test_2d).reshape(X_test.shape)
        else:
            X_train = scaler_X.fit_transform(X_train)
            if len(X_val) > 0:
                X_val = scaler_X.transform(X_val)
            if len(X_test) > 0:
                X_test = scaler_X.transform(X_test)
        
        # 目标标准化
        scaler_Y = StandardScaler()
        y_train = scaler_Y.fit_transform(y_train)
        if len(y_val) > 0:
            y_val = scaler_Y.transform(y_val)
        if len(y_test) > 0:
            y_test = scaler_Y.transform(y_test)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler_X, scaler_Y, feature_cols, target_col)

def preprocess_building_data_multi_floor(file_path=None, 
                                         window_size=24,
                                         train_ratio=0.7,
                                         val_ratio=0.15,
                                         target_zone='z1',
                                         target_type='ac_power',
                                         use_sensors=True,
                                         resample_freq='h',
                                         include_lag_features=True):
    if file_path is None:
        file_path = '/home/lj/Documents/TCN/data/2/datapp/processed_building_data.csv'
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # 基础特征列（不含滞后）
    base_feature_cols = ['hour', 'dayofweek', 'month', 'is_weekend']
    other_energy_cols = []
    for zone in ['z1', 'z2', 'z3', 'z4', 'z5']:
        if zone != target_zone:
            for var in ['ac_power', 'light_power', 'plug_power']:
                other_energy_cols.append(f'{zone}_{var}')
    sensor_cols = []
    if use_sensors:
        for zone in ['z1', 'z2', 'z3', 'z4', 'z5']:
            for sensor in ['temperature', 'humidity', 'illuminance']:
                sensor_cols.append(f'{zone}_{sensor}')
    base_feature_cols = base_feature_cols + other_energy_cols + sensor_cols

    # 滞后列名（如果启用）
    target_col = f'{target_zone}_{target_type}'
    lag_cols = []
    if include_lag_features:
        lag_cols = [f'{target_col}_lag{lag}' for lag in [1, 2, 3, 24]]
    
    # 最终特征列（基础+滞后）
    final_feature_cols = base_feature_cols + lag_cols
    # 去重（确保没有重复）
    final_feature_cols = list(dict.fromkeys(final_feature_cols))

    processed_dfs = []

    for floor_num in df['floor'].unique():
        print(f"\n特征工程楼层 {floor_num}...")
        floor_df = df[df['floor'] == floor_num].copy()
        floor_df.set_index('Date', inplace=True)
        floor_df.sort_index(inplace=True)
        floor_df.drop(columns=['floor'], inplace=True)

        # 添加时间特征
        floor_df['hour'] = floor_df.index.hour
        floor_df['dayofweek'] = floor_df.index.dayofweek
        floor_df['month'] = floor_df.index.month
        floor_df['is_weekend'] = (floor_df.index.dayofweek >= 5).astype(int)

        # 确保所有基础特征列都存在（填充NaN）
        for col in base_feature_cols:
            if col not in floor_df.columns:
                floor_df[col] = np.nan

        if target_col not in floor_df.columns:
            print(f"  楼层 {floor_num} 没有目标列 {target_col}，跳过")
            continue

        # 缺失值插值（特征和目标）
        floor_df[base_feature_cols] = floor_df[base_feature_cols].interpolate(method='linear', limit_direction='both')
        floor_df[target_col] = floor_df[target_col].interpolate(method='linear', limit_direction='both')

        # 重采样
        if resample_freq != 'raw':
            floor_df = floor_df.resample(resample_freq).mean().dropna()

        # 滞后特征（可选）
        if include_lag_features:
            for lag in [1, 2, 3, 24]:
                lag_col = f'{target_col}_lag{lag}'
                floor_df[lag_col] = floor_df[target_col].shift(lag)
            # 删除因滞后产生的 NaN 行
            floor_df.dropna(inplace=True)

        # 确保所有最终特征列都存在（滞后列已经创建）
        for col in final_feature_cols:
            if col not in floor_df.columns:
                floor_df[col] = np.nan

        # 重置索引，添加楼层标识
        floor_df = floor_df.reset_index()
        floor_df['floor'] = floor_num
        processed_dfs.append(floor_df)

    if not processed_dfs:
        raise ValueError("没有成功处理任何楼层的数据")

    # 合并所有楼层清洗后的数据
    merged_clean = pd.concat(processed_dfs, axis=0, ignore_index=True)

    # 调用划分函数
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_X, scaler_Y, _, _) = split_by_floor_temporal(
        df=merged_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        target_col=target_col,
        feature_cols=final_feature_cols,
        window_size=window_size,
        standardize=True
    )

    # 保存标准化器
    os.makedirs('/home/lj/Documents/TCN/scaler2', exist_ok=True)
    with open('/home/lj/Documents/TCN/scaler2/building_scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('/home/lj/Documents/TCN/scaler2/building_scaler_Y.pkl', 'wb') as f:
        pickle.dump(scaler_Y, f)

    # 打印统计
    print("\n" + "="*60)
    print("预处理完成！")
    print(f"总样本数（窗口后）: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print("="*60)

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler_X, scaler_Y, final_feature_cols, target_col)

if __name__ == '__main__':

    file_path = '/home/lj/Documents/TCN/data/2/datapp/processed_building_data.csv'
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler_X, scaler_Y, feature_cols, target_col) = preprocess_building_data_multi_floor(file_path)
