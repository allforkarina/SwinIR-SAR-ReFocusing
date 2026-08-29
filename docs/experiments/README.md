# Scene4 Echo→Image 实验总览与产物保留策略

> 更新日期：2026-08-29
>
> 当前主线：D001监督式单位相位校正

## 结论总览

| 实验 | 路线与问题 | 结果 | 对当前决策的影响 | 代码状态 | checkpoint策略 |
| --- | --- | --- | --- | --- | --- |
| [E001](001-d002-b2a-single-magnitude-overfit.md) | D002单样本幅度恢复 | 通过 | 证明单样本实现可拟合 | 保留为当前共享依赖 | 删除 |
| [E002](002-d002-b2a-joint-16-magnitude-overfit.md) | D002联合16样本幅度恢复 | 64000步未通过但仍改善 | 进入仅扩预算的E003 | 一次性入口已移除 | 删除 |
| [E003](003-d002-b2a-budget-extension.md) | D002延长16样本预算 | 通过，step 110400达到16/16 | 幅度联合拟合主要受预算影响 | 一次性入口已移除 | 删除 |
| [E004](004-d002-spatial-holdout-magnitude.md) | D002空间留出幅度恢复 | 指标未通过，人工失败 | 输出能量压缩且只恢复低频包络 | 训练共享依赖保留，审查入口移除 | 删除 |
| [E005](005-d002-energy-preserving-magnitude.md) | D002增加能量约束 | 数值通过，人工失败 | 只修复能量，仍未形成聚焦结构；停止D002主线 | 训练共享依赖保留，审查入口移除 | 删除 |
| [E006](006-d001-shared-complex-frequency-filter.md) | D001共享复数滤波 | 失败 | 否定跨patch共享空间不变滤波器 | 保留评价依赖 | 无训练checkpoint |
| [E007](007-d001-per-patch-phase-oracle.md) | D001逐样本相位Oracle | unrestricted有效，二次相位无效 | 确立“只改相位”的可恢复上限 | 保留 | 无训练checkpoint |
| [E008](008-d001-supervised-single-phase-correction.md) | D001单样本相位预测 | 通过 | 验证Echo-only前向与监督链路 | 保留共享实现 | 删除，已被E009覆盖 |
| [E009](009-d001-joint-16-phase-correction.md) | D001联合16样本相位预测 | 通过，step 40000达到16/16 | 当前最小成功重聚焦基线 | 保留 | 仅保留`best.pt` |
| [E010](010-d001-phase-spatial-holdout.md) | D001空间留出泛化 | 失败 | 约3.18 updates/sample不足，验证与训练均未学习 | 保留，后续需重跑 | 删除本轮全部checkpoint |
| [E011-A](011-a-d001-seen-training-checkpoint-progress.md) | E010训练checkpoint离线诊断 | 无学习信号 | 证明E010早停权重尚未记忆训练集 | 一次性入口已移除 | 无新增checkpoint |
| [E011-B](011-b-d001-controlled-64-train-overfit.md) | 固定64训练样本充分曝光 | 进行中；step 147200保存中断 | 指标持续改善，但尚未达到64/64 | 保留 | 当前仅保留可读`best.pt` |

当前不能声称模型已经泛化。已经确认的是：相位Oracle能够恢复聚焦结构，SwinIR能够记忆
1个和16个训练样本；64样本仍在受控过拟合诊断中；E010尚未得到充分训练后的空间留出证据。

## 代码保留边界

项目保留E007–E011当前相位主线、数据审计、通用复数训练基础和必要的共享评价函数。
E001–E005及E011-A中不再被当前主线引用的一次性入口已经删除。部分文件虽然最初来自早期
实验，仍被相位主线导入，因此暂时保留，避免为目录整洁进行高风险共享代码迁移。

被移除的入口仍可通过Git历史恢复；实验文档、配置、日志格式和结论不删除。

## checkpoint保留原则

checkpoint不是实验结论的唯一证据。每个运行应永久保留`report.json`、`resolved_config.json`、
`metrics.jsonl`、`train.log`、样本清单和少量人工审查图，但按以下规则控制大型权重：

1. 指标和人工审查均失败的实验删除全部checkpoint；
2. 已被后续实验覆盖的单样本或代理任务删除全部checkpoint；
3. 当前成功基线E009只保留一个可读的`best.pt`；
4. 活跃实验E011-B只保留当前可恢复的`best.pt`，定期删除更旧的`latest.pt`和归档权重；
5. 任何删除前先加载保留权重确认step，再运行dry-run并核对预计释放空间；
6. `/home`使用率达到95%时暂停训练并清理，避免原子保存新旧checkpoint共存时写入失败。

仓库提供`scripts/prune_experiment_checkpoints.py`。它只处理显式指定运行目录下的
`.pt/.pth/.ckpt`，默认只预览；只有增加`--apply`才真正删除。它不会删除报告、日志、配置、
图片或MAT预测。

E011-B的可读`best.pt`可通过`scripts/visualize_phase_train_subset_checkpoint.py`
重新导出完整训练子集审查结果。该入口验证checkpoint步数、保存指标、固定样本及其文件指纹
一致后，输出每个样本的Echo、RAW、EMA、Oracle和Image双尺度对比图，按8个样本分页的
共享Image峰值汇总图，以及包含现场重算指标的`audit_manifest.json`。

## 后续顺序

1. 从E011-B的可读`best.pt(step 140800)`恢复并完成既定160000步预算；
2. 根据完整学习曲线和64样本人工审查判断是否扩预算，而不是仅依据`64/64`硬门槛；
3. 若64样本形成稳定重聚焦，再逐级扩大训练子集；
4. 训练集容量得到支持后，重新设计并运行E010式空间留出泛化门禁；
5. 获得多原始场景数据后，最终使用scene-level holdout验证泛化。
