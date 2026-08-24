# E010：D001监督式相位校正空间留出泛化门禁

> 日期：2026-08-25
>
> 状态：代码完成，等待服务器训练与人工审查
>
> 前置实验：[E009监督式16样本联合相位校正](009-d001-joint-16-phase-correction.md)

## 实验问题

E009已经证明一个共享SwinIR能够同时拟合16个训练样本的内容条件相位校正。E010第一次
检验泛化：

> 模型只读取Echo频谱，在从未用于更新参数、且与训练patch没有像素重叠的空间留出区域，
> 能否预测接近每个样本unrestricted phase oracle的相位校正？

通过意味着当前表示在同一原大图内部具有空间泛化能力；仍不能替代后续完整scene holdout。

## 数据隔离

复用E004/E005已核验的空间划分：

- validation：`row=16400..18400`且`col=7700..9700`，共441对；
- guard：`row=15888..18912`且`col=7188..10212`，除validation外共520对；
- train：其余15714对；
- guard不参与训练、验证或指标计算；
- 任一训练patch与任一validation patch至少在一个轴向相隔512像素，因此没有像素重叠。

E009锚点`patch_row_17500_col_9400_2.mat`位于E010 validation区域。为防止标签信息通过参数
泄漏，E010强制随机初始化，代码没有加载E009 checkpoint的入口。

## 模型输入、监督和输出

数据流与E008/E009保持一致：

```text
X = fftshift(FFT2(Echo / RMS(Echo)))
Y = fftshift(FFT2(Image / RMS(Echo)))
P_target = unit(Y * conj(X))
P_pred = unit(SwinIR(real(X), imag(X)))
prediction = IFFT2(ifftshift(P_pred * X))
```

Image仅在训练样本上构造相位标签和三项损失。验证及部署前向的唯一输入是Echo频谱；
验证Image只用于离线指标和人工审查。网络输出单位相位校正，不能靠压低幅度通过门槛。

## 训练协议

- 随机种子42，不加载E009或其他预训练权重；
- physical batch size为1，Adam，固定学习率`2e-4`；
- 圆周相位、复数重建、log幅度损失权重为`1.0 : 0.25 : 0.25`；
- 最多150000次更新，每5000步在全部441个validation patch上评估；
- Raw是唯一判定权重；EMA使用早期预热，仅供辅助审查；
- 连续三次完整验证通过才写入`passed.pt`并提前结束；
- 早停只跟踪平均相位alignment，连续10次没有至少`1e-4`改善则停止。

最佳checkpoint首先优先选择已经满足完整通过条件的验证点，再在相同通过状态下选择平均相位
alignment更高的点。早停历史单独维护，不会让未通过的高分点覆盖已通过checkpoint。

## 泛化成功条件

每个validation patch分别计算Echo identity和使用该patch Image构造的unrestricted phase
oracle。Oracle仅作为离线性能上界，绝不输入模型。Raw在全部441对上的分布必须同时满足：

- 平均、median weighted phase alignment均至少0.50，p05至少0.20；
- 平均和median的Echo到Oracle RMSE差距闭合比例均至少0.50；
- 至少90%的样本RMSE优于Echo identity；
- 平均coherence达到Oracle的50%；
- 平均SSIM增益达到Oracle增益的50%；
- 平均edge correlation增益达到Oracle增益的40%；
- 高频能量ratio的median位于0.75～1.25。

这些指标同时约束相位、复数一致性、结构、高频能量和分布尾部，避免E004/E005中“只做降噪、
对比度增强或能量缩放也被判为成功”的假阳性。

## 产物与人工审查

- `split_manifest.json`：数据指纹、坐标与train/guard/validation归属；
- `validation_baselines.json`：441对Echo与unrestricted oracle逐样本基线；
- `metrics.jsonl`和`train.log`：训练及完整验证历史；
- `checkpoints/best.pt`、`latest.pt`、`passed.pt`（仅通过时）、归档和中断checkpoint；
- `report.json`：最佳/最终验证、完整逐样本指标、成功条件和推理契约；
- 独立审查脚本从`best.pt`选择最差、最好、中位及空间分散样本，输出
  Echo、Raw、Oracle、Image四列的独立峰值图和共享Image峰值图。

## 决策规则

- 指标连续三次通过且人工图显示Raw确实恢复Oracle中的聚焦结构：进入完整scene holdout；
- 指标通过但人工图仍只是降噪或对比度增强：判为失败，修订指标后重评，不扩大结论；
- 相位和Oracle相对指标稳定改善但未达门槛：先分析困难样本与学习曲线，再决定扩预算；
- validation长期接近Echo而训练损失下降：判定当前映射主要是记忆，转向更强的频域全局条件结构、
  场景参数条件化或物理先验；
- 无论结果如何，不允许用训练patch或E009 checkpoint替代E010的空间泛化证据。
