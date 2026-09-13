"""
dcor_privacy.py
================
在 Split Learning / Split Federated Learning (SFL-V1 / SFL-V2) 中,
用距离相关性 (Distance Correlation, DCOR) 衡量 / 优化"隐私损失"的工具函数。

思路参考 NoPeek (Vepakomma et al., 2020):
    - DCOR(X, Z) 衡量原始输入 X 与切割层激活值 Z 之间的统计依赖程度,
      数值在 [0, 1] 之间,越接近 0 说明 Z 中和 X 相关的信息越少 (泄露风险越低)。
    - 这个函数是可微的,可以直接当作一个训练时的正则项加进 loss;
      也可以在 no_grad 模式下只用来做"事后监控/评估",不参与训练。

本文件提供三样东西:
    1. dist_corr(X, Y)       —— 可微分的 DCOR 计算 (PyTorch)
    2. PrivacyMeter          —— 只做监控用的轻量封装 (不影响训练,记录每轮的 DCOR)
    3. ReconstructionAttackTester —— 一个简单的"攻击者"重建网络,
                                      用来经验性地衡量隐私泄露程度 (补充 DCOR 这种统计量)

依赖: torch >= 1.8
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 可微分的距离相关性 (Distance Correlation)
# ---------------------------------------------------------------------------

def _pairwise_dist(A: torch.Tensor) -> torch.Tensor:
    """计算 A 中样本两两之间的欧氏距离矩阵。A: [n, d]"""
    # ||a_i - a_j||^2 = ||a_i||^2 - 2*a_i.a_j + ||a_j||^2
    r = torch.sum(A * A, dim=1, keepdim=True)          # [n, 1]
    D2 = r - 2.0 * (A @ A.t()) + r.t()                  # [n, n]
    D2 = torch.clamp(D2, min=0.0)                       # 数值误差可能导致极小负数
    return torch.sqrt(D2 + 1e-12)


def dist_corr(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    计算 batch 内 X 与 Y 的距离相关性 DCOR(X, Y),返回一个标量 (可反传梯度)。

    X, Y: [batch_size, ...] 任意形状,函数内部会自动 flatten 成 [batch, -1]。
          例如 X 是原始图像 [B, C, H, W],Y 是切割层激活 [B, C', H', W']。

    注意:
        - batch_size 建议 >= 8~16,否则统计量噪声较大。
        - 对 X 建议做归一化 (比如像素值缩放到 [0,1] 或标准化),
          避免不同量纲主导距离计算。
    """
    n = X.size(0)
    assert Y.size(0) == n, "X 和 Y 的 batch size 必须一致"

    X = X.reshape(n, -1)
    Y = Y.reshape(n, -1)

    a = _pairwise_dist(X)
    b = _pairwise_dist(Y)

    # 双中心化 (double-centering)
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

    dcov_xy = torch.sqrt(torch.clamp((A * B).sum() / (n ** 2), min=eps))
    dvar_xx = torch.sqrt(torch.clamp((A * A).sum() / (n ** 2), min=eps))
    dvar_yy = torch.sqrt(torch.clamp((B * B).sum() / (n ** 2), min=eps))

    dcor = dcov_xy / torch.sqrt(dvar_xx * dvar_yy + eps)
    return dcor


# ---------------------------------------------------------------------------
# 2. 仅用于监控/评估的轻量封装 (不参与训练,不影响模型权重)
# ---------------------------------------------------------------------------

class PrivacyMeter:
    """
    在训练过程中定期调用 update(),记录 DCOR(X, Z) 的历史值,
    用来画出"隐私泄露程度随训练变化"的曲线 (类似论文 Fig. 3 / 11 / 13),
    但完全不参与反向传播、不影响模型训练。
    """

    def __init__(self):
        self.history: list[float] = []

    @torch.no_grad()
    def update(self, X: torch.Tensor, Z: torch.Tensor) -> float:
        value = dist_corr(X, Z).item()
        self.history.append(value)
        return value

    def latest(self) -> float | None:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# 3. 简单的重建攻击测试台 (经验性地衡量隐私泄露,补充 DCOR 这种统计量)
# ---------------------------------------------------------------------------

class SimpleDecoder(nn.Module):
    """
    一个通用的小型反卷积解码器,尝试从激活值 Z 重建原始输入 X。
    仅作为"能力有限的攻击者"参考实现,按你的输入/激活形状调整层数和通道数。
    """

    def __init__(self, in_channels: int, out_channels: int = 3, out_size: int = 32):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return F.interpolate(out, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class ReconstructionAttackTester:
    """
    用法:
        tester = ReconstructionAttackTester(decoder)
        tester.fit(activation_loader)          # 攻击者用一批 (Z, X) 训练重建模型
        leakage_score = tester.evaluate(held_out_loader)   # 在留出集上算平均 L2 / SSIM

    leakage_score 越低,说明攻击者越难从激活值还原原图,隐私保护越好。
    这个指标和 DCOR 是互补关系: DCOR 是训练时的可微代理指标,
    重建误差是"真刀真枪跑一次攻击"的经验性指标,两者建议一起报告。
    """

    def __init__(self, decoder: nn.Module, device: str = "cpu", lr: float = 1e-3):
        self.decoder = decoder.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

    def fit(self, activation_x_pairs, epochs: int = 10):
        """activation_x_pairs: 可迭代对象,每次产出 (Z_batch, X_batch)"""
        self.decoder.train()
        for _ in range(epochs):
            for Z, X in activation_x_pairs:
                Z, X = Z.to(self.device), X.to(self.device)
                self.optimizer.zero_grad()
                X_hat = self.decoder(Z)
                loss = F.mse_loss(X_hat, X)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, activation_x_pairs) -> float:
        """返回测试集上的平均 L2 (MSE) 重建误差,数值越大隐私保护越好"""
        self.decoder.eval()
        total, n = 0.0, 0
        for Z, X in activation_x_pairs:
            Z, X = Z.to(self.device), X.to(self.device)
            X_hat = self.decoder(Z)
            total += F.mse_loss(X_hat, X, reduction="sum").item()
            n += X.numel()
        return total / max(n, 1)
