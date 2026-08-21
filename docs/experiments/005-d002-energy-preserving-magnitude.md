# E005：D002 空间隔离幅度恢复的能量保持损失

> 日期：2026-08-22
>
> 状态：代码完成，等待服务器训练
>
> 前置实验：[E004 同场景空间隔离幅度泛化](004-d002-spatial-holdout-magnitude.md)

## 实验问题

E004 在441个未见验证 patch 上实现了 `100%` 的 RMSE 胜率，并改善了相关性、PSNR
和 SSIM；但预测线性幅度 RMS ratio 的中位数只有 `0.6267`。E005 回答：

> 在不改变数据、网络和训练协议的前提下，加入一个直接约束线性幅度 RMS 的损失，
> 能否消除系统性能量压缩，同时保留 E004 已经获得的结构泛化？

## 唯一实验变量

E004 的损失为对数幅度域 Charbonnier：

```text
L_charb = mean(sqrt((prediction_log - target_log)^2 + epsilon^2))
```

E005 只增加每个 patch 的线性幅度 RMS 对数比惩罚：

```text
prediction_mag = expm1(clamp(prediction_log, 0, 20))
target_mag     = expm1(clamp(target_log, 0, 20))
L_energy       = abs(log(RMS(prediction_mag) / RMS(target_mag)))
L_total        = L_charb + 0.10 * L_energy
```

使用对数比而不是 `abs(RMS_pred - RMS_target)`，可以使过强与过弱的相同比例偏差受到
对称惩罚，也不会让不同 patch 的绝对幅度尺度改变损失权重。`0.10` 在训练前固定：当
E004 的典型 ratio 为 `0.6267` 时，该项对总损失贡献约 `0.047`，足以影响能量但不应
压过逐像素结构损失。

## 严格保持不变的变量

- 输入和标签仍为 Echo-RMS 归一化的 `log1p` 幅度；
- train/guard/validation 仍为 `15714/520/441`，manifest 规则完全相同；
- SwinIR 结构、参数量、`512×512` 输入输出和 `drop_path_rate=0` 不变；
- Adam、学习率 `2e-4`、batch size 1、随机种子42不变；
- 从头初始化，不加载 E004 权重，避免把续训时间作为第二个变量；
- 最多150000步、每5000步验证、10次无改善提前停止不变；
- raw model、按 mean normalized log RMSE 选择 `best.pt` 的规则不变；
- E004 的七项成功检查及阈值全部不变。

## 日志与判定

训练日志同时记录：

- `loss`：加权后的总损失；
- `charb`：原始结构损失；
- `rms_log_ratio`：未加权的能量偏差。

实验只有在最佳 raw checkpoint 同时满足原七项检查时通过。最关键的联合条件是：

1. median linear-magnitude RMS ratio 进入 `0.90～1.10`；
2. mean/median RMSE 仍比 Echo identity 至少改善10%；
3. RMSE 胜率至少75%，且相关性、PSNR、SSIM 仍全部改善。

如果能量 ratio 恢复但结构指标明显退化，说明权重过强；如果结构结果复现 E004 而
ratio 仍显著低于0.9，说明单个全局能量约束不足，下一实验再考虑高响应区域加权，不能
在本次运行中临时调参。
