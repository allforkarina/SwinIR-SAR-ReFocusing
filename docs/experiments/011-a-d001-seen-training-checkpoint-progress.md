# E011-A：D001训练集checkpoint学习信号诊断

> 日期：2026-08-25
>
> 状态：代码完成，等待服务器评测与人工审查
>
> 前置实验：[E010相位校正空间留出泛化门禁](010-d001-phase-spatial-holdout.md)

## 实验问题

E010的final模型在441个空间留出验证样本上没有学习到重聚焦结构，但训练集从未使用相同的
Oracle相对指标进行评估。E011-A只回答：

> E010从step 0训练到step 50000后，在它见过的训练样本上是否出现了可测量且可见的相位
> 重聚焦学习信号？

本实验不更新模型参数，不证明泛化，也不直接证明完整15714个训练样本均可过拟合。

## 固定输入与隔离边界

- 初始模型：E010的`best.pt`，其权重和验证状态均为step 0；
- 最终模型：E010的`final.pt`，其权重和`last_validation`均为step 50000；
- 两个checkpoint必须具有完全相同的resolved config和数据清单指纹；
- 数据来源只允许E010 manifest中的train split，不包含520个guard或441个validation样本；
- 默认从15714个训练样本坐标中，用确定性归一化最远点采样选取441个空间分散probe；
- 两个checkpoint在完全相同的probe上计算指标；
- 模型输入始终只有Echo频谱，Image只用于离线Oracle、重建指标和人工审查。

选择结果写入`selected_train_samples.json`并带有独立指纹，保证重复运行可以核验样本一致性。

## 评测与诊断门槛

每个训练probe分别计算step 0与step 50000的完整E010指标，主要比较：

- mean与median weighted phase alignment变化；
- mean Echo到Oracle RMSE差距闭合比例变化；
- mean Oracle coherence比例变化；
- SSIM与edge gain变化；
- step 50000 RMSE优于step 0的样本比例；
- 高频能量是否仍然合理。

自动状态`training_signal_supported`是低门槛诊断信号，而不是模型成功结论。默认要求：

- mean和median phase alignment均至少提高0.05；
- mean RMSE差距闭合比例至少提高0.05；
- mean coherence Oracle比例至少提高0.05；
- 至少75%的probe最终RMSE优于初始化。

所有检查通过后仍必须人工审查，确认变化属于聚焦结构恢复，而不是随机亮点、噪声重排或显示
尺度差异。

## 可视化

根据训练probe上的指标变化选择最好、最差和空间分散的12个样本，每个样本输出五列双尺度图：

```text
Echo | Initial prediction (step 0) | Final prediction (step 50000) | Oracle phase | Image
```

上排各列独立峰值归一化，下排统一使用Image峰值。联系表统一使用Image峰值，便于同时观察结构
恢复和能量分布。

## 决策规则

- 指标和人工图都显示训练集明显接近Oracle：E010主要是训练样本记忆但无法空间泛化；
- step 50000与step 0在训练probe上同样无结构：当前大规模联合训练本身没有学会；
- 只有弱改善：训练曝光不足、样本梯度冲突和容量限制仍无法区分；
- 若训练学习信号不成立，进入E011-B的`64 → 256 → 1024`受控逐级过拟合；
- 不直接对15714个样本复制E009每样本约2500次曝光，因为对应约3930万次更新，诊断成本过高。

## 产物

- `selected_train_samples.json`：probe坐标、split和选择指纹；
- `report.json`：两个checkpoint逐样本指标、集合统计、delta、检查项和诊断状态；
- `samples/*.png`：五列双尺度代表样本；
- `audit_012_training_samples.png`或分页联系表。
