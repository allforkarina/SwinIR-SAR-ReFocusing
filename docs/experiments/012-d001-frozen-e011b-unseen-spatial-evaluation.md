# E012：冻结E011-B checkpoint的未见空间patch评估

> 日期：2026-08-30
>
> 状态：完成；同场景未见patch零训练泛化失败
>
> checkpoint：E011-B `best.pt`，step 140800

## 实验问题

冻结64样本训练得到的E011-B权重，不进行训练或checkpoint选择，直接在E010预注册的441个
validation patch上评价Raw和EMA，以区分训练集记忆与同场景空间泛化。

模型前向只读取Echo频谱。Image仅用于构造离线unrestricted phase oracle、计算指标和生成
Echo/Raw/EMA/Oracle/Image五列审查图。Raw是判定权重，EMA只作辅助。

## 完整性约束

- checkpoint必须属于`E011-B-D001-controlled-64-train-overfit`且step为140800；
- 当前数据manifest必须与checkpoint训练时的manifest指纹一致；
- checkpoint中的64个训练样本逐一校验文件SHA256和大小；
- validation patch不得与64个训练patch发生像素重叠；
- 评估入口不创建优化器，也不保存或修改checkpoint。

实际运行使用checkpoint记录的严格配对目录：

- `/home/sy/capella_scene4_paired/echo`；
- `/home/sy/capella_scene4_paired/image`。

## 结果

| 指标 | Raw | EMA |
| --- | ---: | ---: |
| mean phase alignment | 0.000027 | 0.000073 |
| median phase alignment | -0.000060 | 0.000055 |
| p05 phase alignment | -0.002531 | -0.002217 |
| mean Oracle RMSE gap closed | -0.000731 | -0.000702 |
| RMSE优于Echo比例 | 0.4626 | 0.4739 |
| mean Oracle coherence fraction | 0.002498 | 0.002458 |
| mean SSIM gain fraction | -0.02873 | -0.02891 |
| mean edge gain fraction | -0.02661 | -0.02616 |
| median high-frequency energy ratio | 1.0153 | 1.0151 |

Raw和EMA均只通过高频能量ratio检查，其余预注册门槛全部失败。高频能量保持来自单位相位
输出不改变Echo频谱幅度，不能解释为重聚焦成功。相位alignment约为零，Oracle gap没有闭合，
结构指标略微恶化，说明该checkpoint在441个未见patch上没有形成可测量的相位泛化。

本结果支持“step 140800权重主要记忆64个训练样本”，但不能单独区分缺少全局坐标、上下文
不足、训练样本不足或映射不可识别等原因，也不能外推到跨场景泛化。

## 产物

服务器运行目录：`runs/scene4_e012_frozen_e011b_unseen_holdout_step140800`。

- `report.json`：441个样本的Raw/EMA逐样本和汇总指标；
- `split_manifest.json`与`resolved_config.json`：数据和评估契约；
- `representative_samples/`：12个代表样本的五列双尺度图。
