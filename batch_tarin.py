"""
批量训练 LSTM 和 TCN 模型（6 维输入）

前提：data_pp.py 中已注释掉时间编码特征 (hour_sin/cos, dow_sin/cos, is_weekend)，
     输出的特征维度直接就是 6。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import swanlab

import data_pp
from model import TimeSeriesLSTM
from tcn import TemporalConvNet


# ===================== TCN 包装 =====================
class TCNForecaster(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)        # (batch, input_size, seq_len)
        out = self.tcn(x)             # (batch, num_channels[-1], seq_len)
        out = out[:, :, -1]
        return self.linear(out)


# ===================== 通用训练函数 =====================
def train_one_model(model, model_name, save_path,
                    X_train, y_train, X_val, y_val,
                    epochs=50, batch_size=64, lr=1e-3,
                    clip_grad=1.0, run=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            if clip_grad and clip_grad > 0:
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
        val_loss_avg   = val_loss / len(X_val)

        if run is not None:
            swanlab.log({
                "train_loss": train_loss_avg,
                "val_loss": val_loss_avg,
                "epoch": epoch + 1,
            }, step=epoch + 1)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), save_path)
            if run is not None:
                run.config.update({"best_val_loss": best_val_loss},
                                  allow_val_change=True)

        print(f"[{model_name}] Epoch [{epoch+1}/{epochs}] | "
              f"Train: {train_loss_avg:.6f} | Val: {val_loss_avg:.6f}")

    model.load_state_dict(torch.load(save_path))
    model.eval()
    print(f"\n[{model_name}] 训练完成，最优验证损失: {best_val_loss:.6f}\n")
    return model, best_val_loss


# ===================== 主流程 =====================
if __name__ == "__main__":
    # ---- 数据 ----
    file_path = "/home/lj/Documents/TCN/data/3/household_power_consumption.txt"
    window_size = 30
    train_ratio = 0.7
    val_ratio = 0.15

    (X_train, y_train, X_val, y_val, X_test, y_test,
     feature_cols, target_col) = data_pp.load_and_preprocess_data(
        file_path, window_size, train_ratio, val_ratio
    )

    INPUT_DIM = X_train.shape[2]   # 应为 6
    print(f"\n>>> 输入维度: {INPUT_DIM}, 特征: {feature_cols}")
    print(f">>> X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}\n")
    assert INPUT_DIM == 6, f"期望 6 维输入，实际为 {INPUT_DIM}，请检查 data_pp.py"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = "/home/lj/Documents/TCN/checkpoints_6d"
    os.makedirs(ckpt_dir, exist_ok=True)

    EPOCHS = 50
    BATCH_SIZE = 64
    CLIP_GRAD = 1.0

    TRAIN_CONFIGS = [
        {
            "name": "LSTM_6d",
            "build": lambda: TimeSeriesLSTM(input_dim=INPUT_DIM,
                                            hidden_units=100,
                                            num_layers=3),
            "lr": 1e-3,
            "save_path": os.path.join(ckpt_dir, "best_lstm_6d.pth"),
            "config_extra": {
                "model": "TimeSeriesLSTM",
                "hidden_units": 100,
                "num_layers": 3,
                "dropout": 0.2,
            },
        },
        {
            "name": "TCN_6d",
            "build": lambda: TCNForecaster(input_size=INPUT_DIM,
                                           output_size=1,
                                           num_channels=[64, 64, 128],
                                           kernel_size=3,
                                           dropout=0.2),
            "lr": 5e-4,
            "save_path": os.path.join(ckpt_dir, "best_tcn_6d.pth"),
            "config_extra": {
                "model": "TCNForecaster",
                "num_channels": [64, 64, 128],
                "kernel_size": 3,
                "dropout": 0.2,
            },
        },
    ]

    results = {}
    for cfg in TRAIN_CONFIGS:
        name = cfg["name"]
        print("=" * 60)
        print(f"开始训练: {name}")
        print("=" * 60)

        model = cfg["build"]()

        run = swanlab.init(
            project="time-series-6d",
            name=f"{name}_lr{cfg['lr']}_bs{BATCH_SIZE}",
            config={
                **cfg["config_extra"],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": cfg["lr"],
                "optimizer": "Adam",
                "loss_fn": "MSELoss",
                "input_dim": INPUT_DIM,
                "window_size": window_size,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "clip_grad": CLIP_GRAD,
                "feature_cols": feature_cols,
            },
            notes=f"{name} 6维输入批量训练",
            reinit=True,
        )

        best_model, best_val = train_one_model(
            model, name, cfg["save_path"],
            X_train, y_train, X_val, y_val,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=cfg["lr"],
            clip_grad=CLIP_GRAD, run=run, device=device,
        )

        # 测试集评估
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = best_model(X_test_t)
            test_loss = nn.MSELoss()(pred, y_test_t).item()
        print(f"[{name}] 测试集 MSE: {test_loss:.6f}\n")

        swanlab.log({"test_loss": test_loss})
        swanlab.finish()

        results[name] = {"best_val_loss": best_val, "test_loss": test_loss}

    print("=" * 60)
    print("所有模型训练完成，结果汇总：")
    print("=" * 60)
    for name, r in results.items():
        print(f"{name:<10} | best_val={r['best_val_loss']:.6f} | test={r['test_loss']:.6f}")