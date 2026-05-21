from model import TimeSeriesCNN,TimeSeriesLSTM
import data_pp
from utils import evaluate_attack
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from Paano import PatchEncoder
from tcn import TemporalConvNet
import swanlab

# ===================== 第一步：定义训练函数（训练CNN模型） =====================
# ===================== Adam优化器 =====================
# ===================== MSE损失函数 =====================
def train_cnn_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, lr=0.001):
    """
    训练TimeSeriesCNN模型，用验证集监控损失，防止过拟合
    """
    # 1. 转换为Tensor并移到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # 2. 构建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 定义优化器和损失函数（回归任务用MSE）
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 4. 训练循环
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        # 计算平均损失
        train_loss_avg = train_loss / len(X_train)
        val_loss_avg = val_loss / len(X_val)
        
        # ===================== SwanLab 记录指标 =====================
        swanlab.log({
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "epoch": epoch + 1
        }, step=epoch + 1)  # 显式指定step便于对齐
        
        # 保存最优模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_cnn_model.pth")
            run.config.update({"best_val_loss": best_val_loss}, allow_val_change=True)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")
    
    # 加载最优模型
    model.load_state_dict(torch.load("best_cnn_model.pth"))
    model.eval()  # 切换到评估模式（关闭Dropout/BatchNorm等）
    print(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")
    return model

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, lr=0.001, clip_grad=1.0):
    """
    训练TimeSeriesCNN模型，用验证集监控损失，防止过拟合
    """
    # 1. 转换为Tensor并移到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    # 2. 构建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 定义优化器和损失函数（回归任务用MSE）
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 4. 训练循环
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        
        # 计算平均损失
        train_loss_avg = train_loss / len(X_train)
        val_loss_avg = val_loss / len(X_val)
        
        # ===================== SwanLab 记录指标 =====================
        swanlab.log({
            "train_loss": train_loss_avg,
            "val_loss": val_loss_avg,
            "epoch": epoch + 1
        }, step=epoch + 1)  # 显式指定step便于对齐
        
        # 保存最优模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_lstm_model.pth")
            run.config.update({"best_val_loss": best_val_loss}, allow_val_change=True)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")
    
    # 加载最优模型
    model.load_state_dict(torch.load("best_lstm_model.pth"))
    model.eval()  # 切换到评估模式（关闭Dropout/BatchNorm等）
    print(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")
    return model

class TCNForecaster(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        """
        input_size: 输入特征维度（11）
        output_size: 输出维度（1）
        num_channels: 每层通道数列表，例如 [64, 64, 128]
        kernel_size: 卷积核大小
        dropout: Dropout比率
        """
        super(TCNForecaster, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        # 最后的线性层将 TCN 输出映射到预测值
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # TCN 需要 (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)          # (batch, input_size, seq_len)
        out = self.tcn(x)               # (batch, num_channels[-1], seq_len)
        # 取最后一个时间步的输出
        out = out[:, :, -1]             # (batch, num_channels[-1])
        out = self.linear(out)          # (batch, output_size)
        return out

def train_tcn_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.001, clip_grad=1.0, run=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 转换为 Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)          # (batch, 1)
            loss = criterion(output, batch_y)
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                loss = criterion(output, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        train_loss_avg = train_loss / len(X_train)
        val_loss_avg = val_loss / len(X_val)

        # 记录到 SwanLab
        if run is not None:
            swanlab.log({
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "epoch": epoch + 1
            }, step=epoch + 1)
            if val_loss_avg < best_val_loss:
                run.config.update({"best_val_loss": val_loss_avg}, allow_val_change=True)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_tcn_model.pth")

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")

    model.load_state_dict(torch.load("best_tcn_model.pth"))
    model.eval()
    print(f"\n训练完成！最优验证损失: {best_val_loss:.6f}")
    return model

if __name__ == "__main__":
    # 数据加载
    file_path = "/home/lj/Documents/TCN/data/3/household_power_consumption.txt"
    window_size = 30
    train_ratio = 0.7
    val_ratio = 0.15
    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, target_col) = data_pp.load_and_preprocess_data(file_path, window_size, train_ratio, val_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TCN 超参数
    input_size = X_train.shape[2]          # 11
    output_size = 1
    num_channels = [64, 64, 128]           # TCN 每层通道数，可根据需要调整
    kernel_size = 3                        # 卷积核大小
    dropout = 0.2
    lr = 0.0005
    batch_size = 64
    epochs = 50
    clip_grad = 1.0

    # 初始化模型
    model = TCNForecaster(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )

    # SwanLab 初始化
    run = swanlab.init(
        project="time-series-tcn",
        name=f"tcn_lr{lr}_bs{batch_size}_ch{num_channels}_k{kernel_size}",
        config={
            "model": "TCNForecaster",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss_fn": "MSELoss",
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_channels": num_channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "clip_grad": clip_grad
        },
        notes="TCN 时间序列预测"
    )

    # 训练
    best_model = train_tcn_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr,
        clip_grad=clip_grad, run=run
    )
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = best_model(X_test_t)
        test_loss = nn.MSELoss()(pred, y_test_t).item()
    print(f"测试集 MSE: {test_loss:.6f}")
    swanlab.log({"test_loss": test_loss})
    swanlab.finish()

   
