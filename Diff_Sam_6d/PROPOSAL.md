# Proposal：基于 SAM-6D ISM 与扩散模型的 YCB-Video 六自由度位姿估计

**文档版本**：与当前仓库 `Diff_Sam_6d` 实现一致（含 `ism/` 与 `diff/` 管线）。  
**定位**：研究/工程 proposal，可用于开题、周报、合作说明或论文「系统与方法」初稿骨架。

---

## 1. 摘要（Executive Summary）

本项目在 **BOP YCB-Video** 数据上，采用与 SAM-6D 等两阶段方法一致的思路：**上游实例分割（ISM）** 提供每帧实例级 mask 与类别；**下游** 训练一个 **条件 DDPM**，在 **6D 连续旋转（vec9）+ 平移** 的位姿空间中对噪声去噪，条件包含 **整图 RGB 下采样、可选相机内参归一化、可选 ISM 对齐后的实例裁剪 RGB、可选物体 ID（标量或 Embedding）**。评测与 **BOP 元数据** 对齐，并支持导出 **BOP19** 格式结果以对接 `bop_toolkit` 官方指标（VSD / MSSD / MSPD）。

---

## 2. 背景与问题陈述

### 2.1 背景

- 六自由度物体位姿估计在机器人抓取、AR、工业检测中需求明确；**YCB-Video** 是 BOP 中常用基准。
- **SAM-6D** 将「检测/分割」与「位姿」解耦；本仓库聚焦 **在 ISM 预测存在的前提下，用生成式模型学习位姿分布**，探索条件设计（几何、语义、分割质量）对扩散位姿头的影响。

### 2.2 拟解决的核心问题

1. 如何在 **计算开销可控** 的条件下，将 **ISM 实例信息** 注入扩散模型（整图条件 vs 实例裁剪条件 vs 物体 ID）？
2. 如何用 **连续旋转表示（vec9）** 替代或补充传统 12 维表示，并保持训练稳定与评测可复现？
3. 如何构建 **可扩展数据管线**（`train_real` 大规模训练、`test` 评测），并处理 **真实推理产物**（大规模 `.npz`、偶发截断文件）？

---

## 3. 技术方案（与代码一一对应）

### 3.1 总体架构

| 阶段 | 组件 | 代码入口 / 说明 |
|------|------|-----------------|
| 上游 | SAM-6D 官方 ISM（Hydra + PyTorch Lightning） | `../SAM-6D-official/.../Instance_Segmentation_Model/`，本仓库用 `ism/run_ycbv_train_real_ism.sh` 等封装 |
| 桥接 | ISM `.npz` 定位、按类 IoU 匹配、损坏文件容错 | `diff/ism_bridge.py`；训练集 `diff/dataset.py` 中 `crop_mask_for_gt_instance` |
| 下游 | 条件 DDPM + `EpsMLP` 噪声预测网络 | `diff/diffusion.py`、`diff/model.py`、`diff/train_pose6d.py` |
| 评测 | 全 test 逐帧采样、自定义 trans/rot/score；BOP19 CSV 导出 | `diff/eval_full_test.py`、`diff/export_bop19_results.py` |

### 3.2 位姿表示与扩散目标

- **主路径**：`train_pose6d.py` 使用 **vec9**（6D 旋转 + 平移缩放），`pose_dim=9`，与 `diff/geometry.py` 中 `pose_to_vec9` / `vec9_to_pose` 一致。
- **训练目标**：标准 DDPM 的 **噪声回归**（`F.mse_loss(eps_hat, noise)`），时间步嵌入为 **正弦 + MLP**（`sinusoid_time_embedding` + `time_mlp`）。

### 3.3 条件向量（Cond）

由 `diff/dataset.py` 中 `build_pose_cond_vector` 拼接，维度由 `total_cond_dim` / checkpoint 元数据 `infer_cond_setup_from_ckpt` 推断：

| 成分 | 维度（典型） | 开关 / 说明 |
|------|----------------|-------------|
| 整图 RGB 8×8 | 192 | 始终启用 |
| ISM 对齐实例裁剪 RGB 8×8 | +192 | `--ism-crop-cond`；裁剪框来自与 GT 同类 IoU 最大的 ISM mask（失败则回退 GT mask） |
| 相机内参 `fx/W, fy/H, cx/W, cy/H` | +4 | 默认启用；`--no-cam-k` 关闭（兼容旧 ckpt） |
| 物体 ID 标量归一化 | +1 | `--cond-obj-id` |
| 物体 ID Embedding | 不进入 cond 向量，拼在 MLP 输入 | `--obj-emb-dim N` + `EpsMLP.obj_emb` |

### 3.4 数据集与实例策略

- **`YcbvIsmPoseDataset`**：`split` 对应 `Data/BOP/ycbv/{split}_metaData.json`；`--pred-dir` 指向 ISM 输出目录（如 `Data/ISM_npz/ycbv_train_real` 符号链接）。
- **`--require-ism`**：仅保留该帧存在可加载 npz 的样本（与部署「有分割再估位姿」一致）。
- **`instance_mode`**：在启用 obj_id / ism_crop / embedding 时为 `random_visible`，否则可为 `largest`（与 `eval_full_test` 最大可见实例策略对齐类说明见 `diff/EXPERIMENTS.md`）。

### 3.5 工程与稳定性（已实现）

- **WSL / 单卡**：ISM 侧 `machine/trainer=local_gloo`、`+machine.trainer.devices=1`（`ism/run_ycbv_train_real_ism.sh`）。
- **跳过已有预测**：`ISM_SKIP_EXISTING_NPZ`；周期性 `torch.cuda.empty_cache`（`ISM_EMPTY_CUDA_EVERY`）。
- **损坏 npz**：`load_ism_npz` 返回 `None` 时回退 GT mask；`ism/scan_bad_ism_npz.py` 全量校验并可 `--delete` 后补跑 ISM。
- **ISM 转 JSON**：官方 `convert_npz_to_json` 已加固，避免单文件损坏导致整池崩溃（`SAM-6D-official/.../model/utils.py`）。

---

## 4. 实验设计建议（与 `diff/EXPERIMENTS.md` 对齐）

### 4.1 公平对比（Fair）

- **同一 split、同一 ISM 覆盖、同一 `require-ism`** 下对比不同位姿头或条件消融，避免与 `train_real` 泛化实验混表。

### 4.2 泛化设定

- **`train_real` 训练 / `test` 评测**：更贴近「训练场景 ≠ 测试场景」；需保证 test 侧 ISM 或关闭 `require-ism` 策略与论文表述一致。

### 4.3 指标层次

1. **仓库内快速指标**：`eval_full_test.py` 的 trans/rot/score（便于迭代）。
2. **BOP19 官方召回**：`export_bop19_results.py` + `bop_toolkit`；无显示环境需 OSMesa / C++ 渲染器路径（详见 `diff/EXPERIMENTS.md`）。

### 4.4 一键训练示例（当前脚本）

- 大规模 train_real + ISM crop + obj embedding：`diff/run_train_trainreal_ismcrop_large.sh`（`BS` / `STEPS` / `RESUME` 可调）。

---

## 5. 实施路线（里程碑）

| 阶段 | 内容 | 验收 |
|------|------|------|
| M1 | BOP 数据与 `train_real` / `test` 元数据就绪 | 元数据 JSON 可读、RGB 路径有效 |
| M2 | ISM 在目标 split 上推理完成 | `result_ycbv_*` 下 npz 数量与元数据帧一致或可解释差异 |
| M3 | 损坏 npz 扫描与补跑 | `scan_bad_ism_npz.py` 损坏数为 0 或已隔离 |
| M4 | 扩散训练收敛与 ckpt | `eps_mlp_pose6d_*.pt` 含 `cond_dim`、`pose_repr` 等元数据 |
| M5 | 全量评测与可视化 | `eval_full_test` CSV/JSON；可选 `eval_vis_best_worst.py` |
| M6 | BOP19 提交级结果 | CSV + `scores_bop19.json` |

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| ISM 推理中断导致截断 npz | 扫描删除 + 补跑；训练侧 `load_ism_npz` 回退 GT |
| WSL 上 NCCL/多卡异常 | ISM 使用 gloo + 单设备配置 |
| `train_real` 全量 + JSON 导出耗时长 | `ISM_SKIP_EXISTING_NPZ`；JSON 阶段异常跳过单文件 |
| 论文口径与内部 score 不一致 | 区分「内部 eval」与「BOP19」两栏报告 |

---

## 7. 仓库结构速查（撰写附录用）

```
Diff_Sam_6d/
  ism/                    # ISM 启动、软链、坏文件扫描
  diff/
    train_pose6d.py       # vec9 DDPM 主训练
    dataset.py            # 条件构造、ISM 裁剪 mask
    ism_bridge.py         # npz 与 IoU 匹配
    model.py              # EpsMLP
    diffusion.py          # 调度与采样
    eval_full_test.py     # 全量 test 评估
    export_bop19_results.py
    EXPERIMENTS.md        # 实验约定与 BOP19 流程
  Data/BOP/ycbv/          # BOP 数据根（用户自备）
  Data/ISM_npz/           # 指向 ISM 输出的符号链接（推荐）
```

---

## 8. 结论

本 proposal 概括了当前代码实现的 **两阶段（ISM + 条件扩散）位姿学习管线**：在 **vec9** 位姿空间上做 DDPM，条件融合 **几何（K）**、**外观（RGB / ISM 裁剪）** 与 **物体身份（标量或 Embedding）**，并配套 **BOP 对齐评测与 BOP19 导出**。后续工作可在 **ISM 质量感知损失**、**更强骨干（U-Net / SE(3)）**、**多实例 per-target 采样** 等方向扩展，而不改变现有数据契约与脚本入口。

---

*若需英文版或「论文方法节」缩写版，可在本文件结构上拆成 `PROPOSAL_en.md` 或 `METHOD_section.md`。*
