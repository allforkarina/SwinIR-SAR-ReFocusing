# E014：精细相位邻域与可靠频率分析

## 目的

在不训练模型、不划分数据集、不修改源 `.mat` 文件的前提下，回答两个问题：

1. Oracle 单位相位校正在 row/col 方向上是否随空间距离呈稳定衰减；
2. 相对能量 hard mask 和 soft weight 能否在保留重聚焦能力的同时，减少低能量频点对相位监督的干扰。

## 固定设计

- 距离：100、200、400、800、1600；
- 方向：row、col；
- 每个“方向 × 距离”最多均匀抽取 100 对，总计约 1000 对；
- 当前基线：`cross_energy > 1e-6`，相位权重为 `sqrt(cross_energy)`；
- 相对 hard mask：相对每个 patch 的 cross-energy 第 99 百分位，门限为 -20、-30、-40、-50 dB；
- soft weight 指数：0、0.25、0.5、1；
- 每个 patch 对在汇总中权重相同；Echo 能量仅用于 low/mid/high 分层，不参与加权；
- 另抽取最多 128 个涉及的 patch，计算不同 hard mask 下的 phase-only Oracle 恢复损失。

## 输出

- `summary.json`：参数、数据配对、抽样计数和只读契约；
- `pair_metrics.csv`：每个空间对、每个 mask/weight profile 的相位相似度；
- `profile_summary.csv`：按方向、距离、能量层和 profile 汇总；
- `oracle_mask_tradeoff.csv`：可靠频率保留率与 Oracle 恢复能力的权衡；
- `figures/`：距离衰减曲线及 mask 恢复权衡图。

该实验只用于选择后续消融设置，不能单独证明从 Echo 可识别相位，也不产生 train/validation/test 划分。
