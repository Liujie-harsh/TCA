# ---------------评估指标计算----------------
import torch
import torch.nn.functional as F


def cosine_similarity(x1, x2):
    """计算时间序列的余弦相似度"""
    x1_flat = x1.reshape(x1.size(0), -1)
    x2_flat = x2.reshape(x2.size(0), -1)
    return F.cosine_similarity(x1_flat, x2_flat, dim=1).mean()

def cosine_similarity_per_sample(x1, x2):
    """逐样本，返回 shape [batch]。用于 TCA 的逐样本决策。"""
    x1_flat = x1.reshape(x1.size(0), -1)
    x2_flat = x2.reshape(x2.size(0), -1)
    return F.cosine_similarity(x1_flat, x2_flat, dim=1)

def evaluate_attack(model, X_original, X_adv, Y_true):
    """计算论文中的4个关键指标"""
    # 1. RMSE
    Y_pred_original = model(X_original).detach()
    Y_pred_adv = model(X_adv).detach()
    rmse = torch.sqrt(F.mse_loss(Y_pred_adv, Y_true))
    
    # 2. XSIM (样本相似度)
    xsim = cosine_similarity(X_original, X_adv).item()
    
    # 3. SIM (预测值与真实值余弦相似度)
    sim = cosine_similarity(Y_true.unsqueeze(1), Y_pred_adv).item()
    
    # 4. SFM (综合指标 - 需要先计算基准值)
    # 注意: 需要在完整实验后计算，参考论文公式(8)
    
    return {
        'RMSE': rmse.item(),
        'XSIM': xsim,
        'SIM': sim
    }

#---------------攻击基线----------------
# BIM (迭代FGSM)
def bim_attack(model, X, Y, epsilon, alpha=0.001, iters=200):
    X_adv = X.clone().detach().requires_grad_(True)
    for _ in range(iters):
        loss = F.mse_loss(model(X_adv), Y)
        loss.backward()
        X_adv = X_adv + alpha * X_adv.grad.sign()
        X_adv = torch.min(torch.max(X_adv, X-epsilon), X+epsilon)
        X_adv = X_adv.detach().requires_grad_(True)
    return X_adv.detach()