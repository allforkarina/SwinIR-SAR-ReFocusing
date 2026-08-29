# E011-B：D001固定64训练样本受控相位过拟合

> 日期：2026-08-25
>
> 状态：训练进行中；step 147200保存时因存储I/O中断，可从step 140800恢复
>
> 前置实验：[E011-A训练集checkpoint学习信号诊断](011-a-d001-seen-training-checkpoint-progress.md)

## 实验问题

E011-A证明E010在step 50000早停时连训练probe也没有形成可测量的相位学习，但每个训练样本
平均只获得约3.18次更新，不能据此判断模型容量或联合训练上限。E011-B只回答：

> 将样本数从E009的16增加到64，在不使用验证集早停并给予充分、均匀训练曝光时，同一个
> SwinIR能否同时记忆64个内容条件相位校正？

本实验是训练集过拟合诊断，不提供未见样本泛化证据。

## 数据边界与选择

- 先按E010完全相同的区域构造15714 train、520 guard和441 validation记录；
- 候选集合严格只包含train split，guard和validation不进入选择、训练或指标；
- 锚点固定为训练区的`patch_row_10000_col_3000_2.mat`；
- 从train候选中使用“锚点加归一化最远点”确定性选择64个空间分散样本；
- 任意两个所选`512×512` patch不得发生像素重叠；
- `selected_samples.json`保存所选文件、坐标、内容指纹、E010数据清单指纹和split统计。

## 训练协议

- 从随机初始化开始，不加载E009或E010权重；
- 输入、Oracle相位标签、三项损失、模型结构、Adam与学习率均复用E009成功路径；
- physical batch size为1，确定性乱序epoch采样，每64 updates覆盖全部样本一次；
- 最大160000 updates，即最多2500 updates/sample；
- 每6400 updates（100个完整subset epochs）评估全部64个训练样本；
- 每32000 updates保存代表图和`latest.pt`；
- 不计算validation指标，不存在validation early stopping；
- Raw是通过权重，EMA使用E009预热策略并仅作辅助审查。

训练只在全部样本通过时提前停止，不会因为平均指标停滞而停止。若中断，可从`latest.pt`或
`interrupted.pt`恢复，但resolved config与固定样本指纹必须完全一致。

## 逐样本成功条件

与E009保持一致，每个Raw预测都必须满足：

- weighted phase alignment至少0.95；
- coherence至少达到自身Oracle的90%；
- SSIM增益至少达到自身Oracle增益的80%；
- edge增益至少达到自身Oracle增益的75%；
- normalized complex RMSE比自身Oracle最多高0.08；
- 高频能量ratio位于0.75～1.25。

只有`64/64`连续三次通过才将实验状态记为`passed`。最佳checkpoint先最大化通过样本数，
通过数相同时最小化集合最差Oracle RMSE差，避免平均值掩盖困难样本。

## 产物与人工审查

- `selected_samples.json`：固定64样本与空间split证据；
- `resolved_config.json`：完整训练、停止条件和推理契约；
- `metrics.jsonl`：每次全量训练集评估；
- `checkpoints/best.pt`、`latest.pt`、`final.pt`和中断checkpoint；
- 训练中保存锚点、最差相位和最差RMSE代表图；
- 结束时为全部64个样本输出Echo、Raw、EMA、Oracle、Image五列双尺度图；
- `report.json`：最终和最佳集合结果、逐样本指标与完整产物路径。

## 决策规则

- `64/64`通过且人工图显示真实结构重聚焦：扩展到256样本；
- 大多数样本持续改善但预算结束前未全通过：先检查学习曲线，再决定扩展预算；
- 通过样本长期互相替代、集合最差值不改善：支持梯度干扰或容量限制假设；
- 64样本仍整体无学习信号：检查大集合训练数值稳定性、损失聚合和模型优化，不进入256；
- 无论是否通过，都不能把训练集记忆解释为未见区域泛化。

## 当前服务器进展（2026-08-28）

训练从随机初始化持续到step 147200，64个样本的集合最差指标仍在总体改善：最低weighted
phase alignment达到`0.7037`，最大Oracle RMSE差降至`0.1928`，最低Oracle coherence比例
达到`0.7584`，最低SSIM和edge增益比例分别达到`0.5726`和`0.3802`。严格门槛仍为`0/64`
通过，因此当前只能说明模型正在学习，不能宣布64样本过拟合成功。

step 147200在原子保存新`best.pt`的临时文件阶段发生存储I/O错误。当时`/home`使用率为
99%。原有`best.pt`经CPU加载验证完整，对应step 140800；`latest.pt`对应更旧状态并可删除。
后续从step 140800恢复，并在训练前保证原子保存所需的额外磁盘空间。
