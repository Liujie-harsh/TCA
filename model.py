import torch.nn.functional as F
from torch import nn

class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim, seq_len):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 60, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(60, 60, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(60, 60, kernel_size=3, padding=1)
        self.fc = nn.Linear(60 * seq_len, 1)  # 预测单步
        
    def forward(self, x):
        # x shape: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_units=100, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_units, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_units, 1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])  # 仅预测下一步


if __name__ == "__main__":
    # 简单测试
    model1 = TimeSeriesCNN(input_dim=6, seq_len=30)
    print(model1)
    model2 = TimeSeriesLSTM(input_dim=6)
    print(model2)
    # =============== 方法1：打印所有参数名称 + 形状 ===============
    print("\n===== 1. 模型所有参数（名称+形状）=====")
    for name, param in model1.named_parameters():
        print(f"{name:<30} | shape: {param.shape}")

    # =============== 方法2：统计总参数 / 可训练参数 ===============
    print("\n===== 2. 模型参数统计 =====")
    total_params = sum(p.numel() for p in model1.parameters())
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)

    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")



