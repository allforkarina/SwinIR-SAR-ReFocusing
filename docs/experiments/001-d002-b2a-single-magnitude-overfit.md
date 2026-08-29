# E001：D002-B2-A 单样本对数幅度过拟合

> 日期：2026-08-18
>
> 状态：实验通过；入口因仍提供当前相位主线使用的共享函数而保留
>
> 所属决策：[D002：Echo→Image 通用配对图像恢复](../decisions/002-generic-image-restoration.md)

## 实验问题

在排除复数相位之后，当前 SwinIR 能否把单个 Echo patch 的散焦幅度结构拟合为对应 Image patch 的聚焦幅度结构？

该实验只检验单样本可学习性，不检验多样本统一映射，也不检验跨场景泛化。

## 唯一实验变量

相对于已有复数单样本实验，本实验只改变数据表示和输出通道：

```text
input  = log1p(abs(Echo)  / rms(Echo))
target = log1p(abs(Image) / rms(Echo))
```

- Echo 和 Image 共用由 Echo 计算的 RMS；
- 不使用 Image 峰值或 Image RMS 归一化，避免标签信息泄漏；
- SwinIR 输入和输出均为一个通道；
- 模型结构规模、Adam、学习率、EMA 和训练步数保持原单样本实验口径；
- 第一轮训练损失只有对数幅度 Charbonnier，不加入 SSIM、梯度或感知损失。

## 对照与指标

报告同时保存：

- 全零输出基线；
- Echo identity 基线；
- raw 模型和 EMA 模型；
- normalized log RMSE；
- log-magnitude correlation；
- 还原到线性幅度后的 RMS ratio；
- log-magnitude PSNR 和 SSIM；
- 每个保存步的对比图和 MAT 预测。

raw 模型连续三次评估同时满足以下条件才记为通过：

| 指标 | 门槛 |
| --- | ---: |
| normalized log RMSE | `<= 0.10` |
| log-magnitude correlation | `>= 0.95` |
| magnitude RMS ratio | `0.90～1.10` |
| PSNR | `>= 30 dB` |
| SSIM | `>= 0.95` |

EMA 只作观察，不参与通过判定，因为高衰减 EMA 在单样本短训练中通常明显滞后。

## 运行约定

推荐样本继续使用已完成复数单样本验证的：

`patch_row_17500_col_9400_2.mat`

正常输出目录：

`runs/scene4_single_magnitude_row17500_col9400`

如果训练中断，使用同一组参数和 `checkpoints/interrupted.pt` 恢复。脚本会严格比对配置和样本 SHA-256，防止错误断点被接入。

## 后续判定

- 若本实验通过：执行 B2-A 的 16 样本联合过拟合；
- 若单样本都不能通过：先检查幅度表示、残差输出范围和学习率，不进入 16 样本；
- 若单样本通过而 16 样本失败：优先判断是否仍存在样本间算子冲突或上下文不足；
- 无论结果如何，本实验都不能单独证明 SAR 复数重聚焦成功。
