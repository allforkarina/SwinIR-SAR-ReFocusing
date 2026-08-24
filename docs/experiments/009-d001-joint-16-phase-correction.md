# E009：D001监督式16样本联合相位校正门禁

> 日期：2026-08-24
>
> 状态：代码完成，等待服务器训练
>
> 前置实验：[E008单样本相位校正](008-d001-supervised-single-phase-correction.md)

## 实验问题

E008证明SwinIR能在单个训练样本上从Echo频谱预测接近unrestricted oracle的单位相位
校正。E009只增加样本数，回答：

> 同一个SwinIR能否同时记忆16个空间分散样本各自的内容条件相位校正？

本实验仍是训练集过拟合门禁，不使用验证集，不声称未见样本泛化。

## 固定数据与变量

- 复用E002/E003的锚点加确定性归一化最远点采样；
- 选择16个原大图坐标中互不重叠的`512×512`配对patch；
- 每对样本以自己的Echo RMS同时归一化Echo与Image；
- 输入、相位标签、模型、三项损失、Adam和学习率与E008保持一致；
- 从随机初始化开始，不加载E008单样本权重；
- physical batch size为1，以确定性乱序epoch采样，每16步覆盖全部样本一次；
- 默认64000步，相当于每个样本约4000次更新；每1600步评估全部16个样本。

唯一有意修正是EMA更新：第`n`次成功更新使用
`min(0.999, 1 - 1/n)`，前1000步逐渐进入目标衰减，避免E008短训练中的严重滞后。
Raw仍是唯一通过判据，EMA只提供辅助审查。

## 集合级成功条件

每个样本都单独计算自己的Echo identity与unrestricted phase oracle。Raw必须对16个样本
逐一满足E008的全部门槛：

- weighted phase alignment至少0.95；
- coherence至少达到自身Oracle的90%；
- SSIM增益至少达到自身Oracle增益的80%；
- edge correlation增益至少达到自身Oracle增益的75%；
- normalized complex RMSE比自身Oracle最多高0.08；
- 高频能量ratio位于0.75～1.25。

只有`16/16`连续出现三次才通过。平均值不参与通过判定，防止简单样本掩盖困难样本。
若某个Oracle相对Echo没有正SSIM或edge增益，则候选只需达到或超过该Oracle的绝对指标，
不对非正增益进行无定义的比例除法。

最佳checkpoint先按通过样本数最大选择；通过数相同时，再选择最差Oracle RMSE差最小者。

## 产物与人工审查

- `selected_samples.json`：16个样本坐标、顺序、指纹与互不重叠声明；
- `metrics.jsonl`：每轮16个样本的Raw/EMA逐样本与集合指标；
- `checkpoints/best.pt`、`latest.pt`、`final.pt`和中断检查点；
- 训练过程中保存锚点、最差RMSE和最差相位样本；
- 结束时为全部16个样本保存Echo、Raw、EMA、Oracle、Image五列双尺度审查图；
- `report.json`：基线、最终结果、最佳通过数和完整推理契约。

## 决策规则

- `16/16`通过且人工图均接近Oracle：进入E010空间隔离的未见patch泛化；
- 多数样本随预算稳定改善但未通过：只扩展训练预算，不改变模型和损失；
- 长期只有少数样本通过或不同样本反复互相替代：当前频域SwinIR存在容量或内容条件冲突，
  再比较更适合频域全局映射的结构；
- 相位alignment高但视觉普遍不及Oracle：检查固定Echo频谱幅度的上限和混合损失权重；
- 任何通过都只表示16个训练样本可共同记忆，不能当作泛化结论。
