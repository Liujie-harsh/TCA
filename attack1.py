from sympy import epath
from model import TimeSeriesCNN,TimeSeriesLSTM
import torch
import torch.nn.functional as F
from utils import evaluate_attack, cosine_similarity
from data_pp import load_and_preprocess_data
from tcn import TemporalConvNet, TCNForecaster
import swanlab


def untargeted_tca(model, X, Y, epsilon, alpha, iterations=200,
            log_to_swanlab=False, log_prefix="attack", log_interval=10):
    """
    X: 原始时间序列 [batch, seq_len, features]
    Y: 真实值 [batch, 1]
    """
    X_adv = X.clone().detach().requires_grad_(True)
    
    for i in range(iterations):
        # 1. 计算梯度
        model.zero_grad()
        Y_pred = model(X_adv)
        loss = F.mse_loss(Y_pred, Y)  # 论文用L2损失
        loss.backward()
        
        # 2. 迭代扰动
        grad = X_adv.grad.data
        X_tilde = X_adv + alpha * torch.sign(grad)
        
        # 3. 裁剪确保在ε范围内
        X_tilde = torch.min(torch.max(X_tilde, X - epsilon), X + epsilon)
        

        perturbation_sign = torch.sign(X_tilde - X)
        X_boundary = X + perturbation_sign * epsilon
        cos_tilde=cosine_similarity(X,X_tilde)
        cos_boundary=cosine_similarity(X,X_boundary)
        
        # 4. 余弦相似度约束 (核心创新)
        if cosine_similarity(X, X_tilde) > cosine_similarity(X, X_boundary):
            X_adv = X_tilde
            chosen = "X_tilde"
        else:
            X_adv = X_boundary  # 保持最大扰动但满足相似性
            chosen = "X_boundary"
        
        # 5. === SwanLab 记录 ====
        if log_to_swanlab and (i % log_interval == 0 or i == iterations - 1):
            try:
                # 安全转换标量值
                cos_tilde_val = cos_tilde.item() if torch.is_tensor(cos_tilde) else float(cos_tilde)
                cos_boundary_val = cos_boundary.item() if torch.is_tensor(cos_boundary) else float(cos_boundary)
                

                swanlab.log({
                    f"{log_prefix}/loss": loss.item(),
                    f"{log_prefix}/cos_tilde": cos_tilde_val,
                    f"{log_prefix}/cos_boundary": cos_boundary_val,
                    f"{log_prefix}/chosen_X_tilde": 1 if chosen == "X_tilde" else 0,  # 二值指标便于可视化
                    f"{log_prefix}/iteration": i
                }, step=i)
            except Exception as e:
                print(f"[SwanLab Warning] Iter {i} logging failed: {e}")

        X_adv = X_adv.detach().requires_grad_(True)
    
    return X_adv.detach()

def targeted_tca(model, X, Y, Y_target, epsilon=0.1, alpha=0.001, 
                iterations=200, direction=1, margin=1.0):
    """
    direction: 1=overforecasting, -1=underforecasting
    margin: 攻击幅度参数m
    """
    # 设置目标序列
    Y_star = Y + direction * margin
    
    X_adv = X.clone().detach().requires_grad_(True)
    
    for i in range(iterations):
        model.zero_grad()
        Y_pred = model(X_adv)
        loss = F.mse_loss(Y_pred, Y_star)  # 最小化与目标的差异
        loss.backward()
        
        # 关键区别: 沿梯度下降方向更新 (而非上升)
        grad = X_adv.grad.data
        X_tilde = X_adv - alpha * torch.sign(grad)
        X_tilde = torch.min(torch.max(X_tilde, X - epsilon), X + epsilon)
        
        # 余弦相似度约束 (有方向性)
        perturbation_sign = torch.sign(X_tilde - X)
        X_boundary = X + perturbation_sign * epsilon
        # 4. 余弦相似度约束 (核心创新)
        if cosine_similarity(X, X_tilde) > cosine_similarity(X, X_boundary):
            X_adv = X_tilde
        else:
            X_adv = X_boundary  # 保持最大扰动但满足相似性
    
    return X_adv.detach()

def untargeted_fgsm(model, X, Y, epsilon, 
                log_to_swanlab=False, log_prefix="fgsm_attack"):
    """
    单步FGSM攻击
    X: [batch, seq_len, features], Y: [batch, 1]
    """
    X_adv = X.clone().detach().requires_grad_(True)
    
    # 前向+梯度计算
    model.zero_grad()
    Y_pred=model(X_adv)
    loss = F.mse_loss(Y_pred, Y)
    loss.backward()
    
    # 生成对抗样本（核心单步逻辑）
    gard = X_adv.grad.sign()
    X_tilde = X_adv + epsilon * gard
    
    X_tilde=torch.min(torch.max(X_tilde,X-epsilon),X+epsilon)
    
    # === SwanLab日志（结构对齐TCA）===
    if log_to_swanlab:
        try:
            perturbation = X_tilde - X
            cos_sim = cosine_similarity(X, X_tilde).mean().item()
            swanlab.log({
                f"{log_prefix}/loss": loss.item(),
                f"{log_prefix}/cos_sim": cos_sim,
                f"{log_prefix}/epsilon": epsilon,
                f"{log_prefix}/perturbation_norm": gard.abs().mean().item(),
                f"{log_prefix}/perturbation_linf": perturbation.abs().max().item()
            })
        except Exception as e:
            print(f"[SwanLab Warning] FGSM logging failed: {e}")
    
    return X_tilde.detach()

def untargeted_BIM(model, X, Y, epsilon, alpha,iterations=200,
                log_to_swanlab=False, log_prefix="BIM_attack", log_interval=10):
    X_adv = X.clone().detach().requires_grad_(True)
    
    for i in range(iterations):
        # 1. 计算梯度
        model.zero_grad()
        Y_pred = model(X_adv)
        loss = F.mse_loss(Y_pred, Y)  # 论文用L2损失
        loss.backward()
        
        # 2. 迭代扰动
        grad = X_adv.grad.data
        X_tilde = X_adv + alpha * torch.sign(grad)
        
        # 3. 裁剪确保在ε范围内
        X_adv = torch.min(torch.max(X_tilde, X - epsilon), X + epsilon)
        
        
        # 5. === SwanLab 记录 ====
        if log_to_swanlab and (i % log_interval == 0 or i == iterations - 1):
            try:
                perturbation = X_adv - X
                cos_sim = cosine_similarity(X, X_adv).mean().item()
                swanlab.log({
                    f"{log_prefix}/loss": loss.item(),
                    f"{log_prefix}/cos_sim": cos_sim,
                    f"{log_prefix}/perturbation_linf": perturbation.abs().max().item(),
                    f"{log_prefix}/iteration": i
                }, step=i)
            except Exception as e:
                print(f"[SwanLab Warning] Iter {i} logging failed: {e}")

        X_adv = X_adv.detach().requires_grad_(True)
    
    return X_adv.detach()

def untargeted_PGD(model, X, Y, epsilon, alpha,iterations=200,
                log_to_swanlab=False, log_prefix="PGD_attack", log_interval=10):
    delta = torch.empty_like(X).uniform_(-epsilon, epsilon)
    X_adv = (X + delta).clamp(X.min(), X.max())
    X_adv = torch.min(torch.max(X_adv, X - epsilon), X + epsilon)
    X_adv = X_adv.detach().requires_grad_(True)
    
    for i in range(iterations):
        # 1. 计算梯度
        model.zero_grad()
        Y_pred = model(X_adv)
        loss = F.mse_loss(Y_pred, Y)  # 论文用L2损失
        loss.backward()
        
        # 2. 迭代扰动
        grad = X_adv.grad.data
        X_tilde = X_adv + alpha * torch.sign(grad)
        
        # 3. 裁剪确保在ε范围内
        X_adv = torch.min(torch.max(X_tilde, X - epsilon), X + epsilon)
        X_adv = X_adv.clamp(X.min(), X.max())
        
        # 5. === SwanLab 记录 ====
        if log_to_swanlab and (i % log_interval == 0 or i == iterations - 1):
            try:
                perturbation = X_adv - X
                cos_sim = cosine_similarity(X, X_adv).mean().item()
                swanlab.log({
                    f"{log_prefix}/loss": loss.item(),
                    f"{log_prefix}/cos_sim": cos_sim,
                    f"{log_prefix}/perturbation_linf": perturbation.abs().max().item(),
                    f"{log_prefix}/iteration": i
                }, step=i)
            except Exception as e:
                print(f"[SwanLab Warning] Iter {i} logging failed: {e}")

        X_adv = X_adv.detach().requires_grad_(True)
    
    return X_adv.detach()

if __name__ == "__main__":

    # model = TemporalConvNet(input_dim=11)

  
    file_path = "/home/lj/Documents/TCN/data/3/household_power_consumption.txt"
    window_size = 30
    train_ratio = 0.7
    val_ratio = 0.15
    (X_train, y_train, X_val, y_val,
     X_test, y_test, 
     feature_cols, target_col) = load_and_preprocess_data(file_path,
                                 window_size, train_ratio, val_ratio,)
    
    X = torch.from_numpy(X_test).float()
    Y = torch.from_numpy(y_test).float()

    input_size = X.shape[2]                # 11
    output_size = 1
    # 必须与训练时的超参数相同，否则加载权重会出错
    num_channels = [64, 64, 128]           # 与训练代码中的配置一致
    kernel_size = 3                        # 与训练代码中的 kernel_size 一致
    dropout = 0.2
    model = TCNForecaster(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    )
    model.load_state_dict(torch.load("/home/lj/Documents/TCN/best_tcn_model.pth", map_location="cpu"))
    model.eval()
    model.to("cpu")
    for param in model.parameters():
        param.requires_grad = False

#===============epsilon修改=====================
    epsilon = 0.2
    run = swanlab.init(
        project="TCA_Attack_Experiments",
        name=f"TCN_epsilon_{epsilon}",
        config={
            "model": "TCN",
            "attack": "untargeted_tca",
            "epsilon": epsilon,
            "alpha": 0.001,
            "iterations": 200,
            "dataset": "household_power",
            "device": "cpu"  # 或 torch.cuda.is_available()
        }
    )
    # ===================攻击切换=====================
    X_adv_untargeted = untargeted_tca(model, X, Y, swanlab.config["epsilon"], swanlab.config["alpha"],
                                            log_to_swanlab=True, log_prefix="untargeted_tca",
                                         )  
    # 评估攻击效果
    metrics = evaluate_attack(model, X, X_adv_untargeted, Y)
    rmse = metrics.get('RMSE', metrics.get('rmse', 0))
    xsim = metrics.get('XSIM', metrics.get('xsim', 0))
    sim = metrics.get('SIM', metrics.get('sim', 0))
    
    # 计算扰动强度（补充关键指标）
    perturbation = X_adv_untargeted - X
    avg_l2 = torch.norm(perturbation, p=2, dim=(1, 2)).mean().item()
    avg_linf = torch.max(torch.abs(perturbation)).item()
    
    # 记录最终结果到SwanLab
    print(f"✅ 攻击完成 | RMSE: {rmse:.4f} | XSIM: {xsim:.4f} | SIM: {sim:.4f}")
    
    swanlab.log({
        "final/RMSE": metrics["RMSE"],
        "final/XSIM": metrics["XSIM"],
        "final/SIM": metrics["SIM"],
        "final/avg_l2": avg_l2,
        "final/avg_linf": avg_linf
    })
    swanlab.finish()  # 显式结束实验
