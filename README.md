# 时间序列预测与对抗鲁棒性评估 (Time Series Forecasting & Adversarial Robustness Evaluation)

本项目提供了一个基于 PyTorch 的完整框架，用于在多元时间序列数据上训练深度学习模型，并评估其抵御各种对抗性攻击的鲁棒性。该项目支持监督预测以及对比预训练，并与 SwanLab 无缝集成以进行实验跟踪和指标可视化。

## 📌 项目概述

代码库的设计涵盖了数据处理、神经网络架构训练（以预测未来数值），并对其漏洞进行了严格的测试。项目深入探讨了模型如何应对通过基于梯度的对抗攻击（包括专门的时间序列对比攻击 TCA）生成的微小输入扰动。

## 🗂 项目结构

该项目被模块化为数据处理、模型架构、训练流程和对抗性攻击评估四个主要部分：

### 数据预处理 (Data Preprocessing)

- **data_pp.py**: 负责处理家庭用电量数据集。它执行每小时重采样、去除缺失值、标准归一化以及生成滑动窗口。
- **data_pp2.py**: 处理多层建筑的 CSV 传感器数据。它映射特定区域和传感器的数据，生成时间与滞后特征 (lag features)，并按楼层在时间序列上划分数据集。

### 模型架构 (Model Architectures)

- **model.py**: 定义了 `TimeSeriesCNN`（使用一维卷积）和 `TimeSeriesLSTM` 网络架构。
- **Paano.py**: 实现了 `PatchEncoder` 架构。该网络采用了一维 CNN，并配备了可逆实例归一化（`RevIN1d`）和自适应平均池化机制。

### 训练流程 (Training Pipelines)

- **modeltra.py**: 为 CNN、LSTM 和 TCN 模型提供独立的训练循环，使用 MSE 损失和 Adam 优化器。它还集成了 SwanLab 以跟踪训练和验证损失。
- **batch_tarin.py**: 一个专门的脚本，用于在 6 维输入特征上批量训练 LSTM 和 TCN 模型。
- **paanotrain.py**: 对 `PatchEncoder` 执行自监督/对比预训练。它结合了三元组损失 (triplet loss) 和预设的分类任务，并使用了余弦退火学习率。

### 对抗性攻击 (Adversarial Attacks)

- **attack.py, attack1.py, attack2.py**: 评估经过训练的模型的鲁棒性（分别针对 CNN、TCN 和 PatchEncoder）。这些脚本使用多种算法生成对抗性扰动，并将 RMSE、XSIM 和 SIM 等最终评估指标记录到 SwanLab。

## 🛡️ 实现的对抗性攻击

该框架支持多种专门针对时间序列数据设计的白盒对抗攻击。

| **攻击方法**                    | **描述**                                                     |
| ------------------------------- | ------------------------------------------------------------ |
| **FGSM**                        | 快速梯度符号法 (Fast Gradient Sign Method)：一种利用梯度符号的单步攻击。 |
| **BIM**                         | 基本迭代法 (Basic Iterative Method)：FGSM 的迭代扩展版本，通过多次应用微小扰动生成对抗样本。 |
| **PGD**                         | 投影梯度下降法 (Projected Gradient Descent)：在 epsilon 范围内使用随机噪声初始化，然后执行迭代更新与裁剪。 |
| **无目标 TCA (Untargeted TCA)** | 时间序列对比攻击：迭代地扰动样本，同时强制执行余弦相似度约束 (Cosine Similarity)，以最大程度保留原始时间序列模式。 |
| **有目标 TCA (Targeted TCA)**   | 结合余弦相似度约束，并沿着梯度下降方向更新，驱使模型的预测值达到特定的目标幅度（如过度预测或预测不足）。 |

## 🚀 快速开始

### 环境依赖

请确保您的环境中安装了以下核心依赖库：

- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `swanlab`
- `tqdm`

### 1. 数据准备

通过运行预处理脚本来准备您的数据集。这将生成必要的滑动窗口并保存数据标准化器 (Scalers)。

Bash`# 处理家庭用电量数据python data_pp.py# 处理多层建筑传感器数据python data_pp2.py`

### 2. 模型训练

训练标准的监督预测模型，或对表示编码器 (Encoder) 进行预训练。

Bash`# 训练监督模型 (CNN, LSTM, TCN)python modeltra.py# 或者在 6 维特征上进行批量训练python batch_tarin.py# 对 PatchEncoder 进行对比预训练python paanotrain.py`

### 3. 对抗评估

针对已训练的模型生成对抗样本，以评估其鲁棒性。

Bash`# 使用 PGD 评估 CNN 的鲁棒性python attack.py# 使用 TCA 评估 TCN 的鲁棒性python attack1.py# 使用 TCA 评估 PatchEncoder/LSTM 的鲁棒性python attack2.py`

## 📊 实验跟踪

本项目使用 **SwanLab** 进行全面的实验跟踪。在训练和攻击评估期间，诸如 `train_loss`、`val_loss`、`cos_sim` 以及扰动幅度（`avg_l2`、`avg_linf` 等）都会自动同步到您的 SwanLab 仪表板以便于可视化与对比。请在相应的脚本配置字典中自定义您的项目名称 (Project) 和实验名称 (Name)。