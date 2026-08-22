# E004：D002 同场景空间隔离幅度泛化

> 日期：2026-08-20
>
> 状态：已完成；未通过预注册总判定，但结构泛化成立
>
> 前置实验：[E003 16样本训练预算扩展](003-d002-b2a-budget-extension.md)

## 实验问题

E003 已证明当前 SwinIR 可以同时记忆同一 parent scene 中16对局部对数幅度映射。
E004 不再增加过拟合样本或训练预算，而是回答：

> 只在 Scene4 的训练空间区域学习后，模型能否在没有参与训练、并且与训练 patch
> 不共享原始大图像素的连续验证区域中，稳定优于直接输出 Echo 的基线？

本实验仍然只是同一 parent scene 内的经验恢复实验，不是跨场景泛化，也不恢复复数
相位。

## 为什么不能从 E003 权重继续

E003 的锚点 `patch_row_17500_col_9400_2.mat` 位于 E004 的验证区域内。如果加载
E003 权重，验证标签已经被模型用于训练，结果将发生泄漏。因此 E004：

- 继承 E003 的网络结构、输入表示、损失、优化器和学习率；
- 不继承 E003 的模型、EMA 或优化器状态；
- 使用固定随机种子从头初始化。

## 空间划分

Scene4 配对网格共有 `16675` 个 `512×512` patch，起始坐标步长为100。划分使用
patch 左上角的全局坐标：

| split | 坐标规则 | 数量 | 用途 |
| --- | --- | ---: | --- |
| validation | row `16400～18400`，col `7700～9700` | `441` | 连续未见区域 |
| guard | row `15888～18912`，col `7188～10212`，扣除 validation | `520` | 禁止训练 |
| train | guard 外的所有配对 patch | `15714` | 参数更新 |

保护带由验证区四边向外扩展512像素。由于实际坐标间隔为100，最靠近验证区的训练
patch 起点至少相差600像素，因此任意训练 patch 与任意验证 patch 的半开原始像素
窗口都不相交。

脚本对 split 数量进行严格校验；目录缺文件、坐标变化或误用其他 scene 时会在训练前
失败。

## 固定实验变量

- 输入：`log1p(abs(Echo) / rms(Echo))`；
- 标签：`log1p(abs(Image) / rms(Echo))`；
- 单通道 `512→512` SwinIR，`drop_path_rate=0`；
- Adam，学习率 `2e-4`，常数学习率；
- magnitude Charbonnier loss；
- batch size 为1；
- 最多 `150000` 次有效参数更新；
- 每 `5000` 步评估全部441个验证 patch；
- raw model 是验证和最佳检查点的权威模型；
- 按验证集平均 normalized log RMSE 保存 `best.pt`；
- 连续10次验证没有至少 `1e-4` 的改善时提前停止。

EMA 仍被保存用于后续消融，但不参与本实验的成功判定，避免相对于 E003 同时更换
权威模型。

## 基线与成功标准

Echo identity 基线是把归一化后的 Echo 对数幅度直接作为预测。模型必须同时满足：

1. 验证集平均 normalized log RMSE 相对基线降低至少 `10%`；
2. 验证集 RMSE 中位数相对基线降低至少 `10%`；
3. 至少 `75%` 的验证 patch 的 RMSE 优于各自 Echo identity；
4. 平均相关性、PSNR 和 SSIM 均优于 Echo identity；
5. 预测线性幅度 RMS ratio 的验证集中位数位于 `0.90～1.10`。

这些条件避免仅凭少数样本、平均误差、锐化或低幅度退化解宣布成功。

## 产物和恢复契约

运行目录包含：

- `split_manifest.json`：全部文件的 train/guard/validation 归属与指纹；
- `echo_identity_baseline.json`：441个验证样本的固定基线；
- `metrics.jsonl` 和 `train.log`：训练、验证和比较轨迹；
- `checkpoints/best.pt`：验证平均 RMSE 最佳状态；
- `checkpoints/latest.pt`、`final.pt`、定期归档和中断检查点；
- `report.json`：最佳验证结果、逐样本指标和最终判定。

断点恢复会严格比较完整配置和数据 manifest 指纹，并恢复 raw、EMA、Adam、学习率
调度器、混合精度、采样位置、early stopping 和所有 RNG 状态。

## 服务器结果

最佳 raw model 出现在 `step=95000`。随后验证平均 RMSE 不再改善，训练在
`step=145000` 因连续10次验证未改善而提前停止。

| 指标 | 最佳结果 |
| --- | ---: |
| 验证样本数 | 441 |
| mean normalized log RMSE | 0.531840 |
| mean RMSE 相对 Echo 改善 | 28.52% |
| median RMSE 相对 Echo 改善 | 24.46% |
| RMSE 胜率 | 100% |
| mean correlation | 0.254021 |
| mean PSNR | 21.8556 dB |
| mean SSIM | 0.273944 |
| median linear-magnitude RMS ratio | 0.626695 |

七项预注册检查中，RMSE、胜率、相关性、PSNR 和 SSIM 六类检查全部通过；唯一失败项
是线性幅度 RMS ratio 未进入 `0.90～1.10`。其分布为：均值 `0.5957`、中位数
`0.6267`、P05 `0.3674`、P95 `0.7549`、最大值 `0.8146`，因此不是少数异常样本，
而是整个验证区域都存在系统性的输出能量压缩。

## 结论与下一步

E004 证明局部 `512×512` Echo 幅度包含可迁移到同场景未见区域的恢复信息：441个
验证 patch 的 RMSE 全部优于各自 Echo identity，排除了“只能记忆16个样本”的解释。
但仅用逐像素 log-domain Charbonnier 会偏向大量低幅度背景，不能可靠保持少量高幅度
散射点所主导的线性幅度能量。

因此下一步不增加上下文、不扩大模型，也不延长已经平台化的训练；E005 只增加一个与
失败判据直接对应的线性幅度 RMS 约束，继续使用相同划分、初始化、网络和评估协议。

## 最佳 checkpoint 人工审查

`scripts/visualize_spatial_holdout_checkpoint.py` 从 `best.pt` 中读取 raw model 和最佳验证
步的逐样本指标，只在441个 validation patch 中进行确定性抽样。默认12个样本包括
RMSE、RMS ratio、相关性和 SSIM 的高低端与中位样本，并补入空间分散样本，避免只挑选
视觉效果较好的结果。

每个样本输出一张 `2×3` 图：

- 上排对 Echo、prediction、Image 分别按各自峰值归一化，用于观察轮廓和结构；
- 下排三者统一按 Image 峰值缩放，用于观察绝对相对幅度和能量压缩；
- 标题同时报告模型与 Echo identity 的 RMSE、相关性、RMS ratio、PSNR 和 SSIM；
- `audit_manifest.json` 记录选择原因、坐标、重算指标和 checkpoint 原始指标。

人工审查时应分别回答：预测是否保留正确轮廓、是否比 Echo 更接近 Image、是否产生虚假
亮点或过度平滑，以及共享尺度下预测是否整体偏暗。上排只能证明结构相似，不能用来否定
E004 已测得的幅度压缩。
