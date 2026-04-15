# 实验设定与论文表述（ISM + Diffusion Pose）

## 1. 系统流程（可写进论文方法）

与 SAM-6D 等两阶段工作一致，整体为：

1. **Instance Segmentation（ISM）**：对 YCB-Video test 图像得到实例 mask / `category_id`（官方或自跑推理，输出 `.npz`）。
2. **Pose 扩散模型**：以 **RGB 下采样 + 归一化相机内参**（`fx/W, fy/H, cx/W, cy/H`，共 4 维，默认开启）为条件，对 **位姿向量**（x12 或 vec9）做 DDPM 去噪训练；评测时在 test 上与 BOP GT 对齐。旧权重无 `cond_dim` 字段时按 **仅 RGB（192 维）** 加载；新训练默认 **196 维**。训练可加 `--no-cam-k` 与旧设定对齐。

当前实现里，训练时 **`--require-ism` 只保证该帧存在 ISM 预测文件**（与部署管线一致）；`mask_iou` 已算入 `dataset` 但未写入 loss（若后续要强调「利用 ISM 质量」，可再接一项辅助监督或拼到条件向量里）。

## 2. 公平对比（同一数据管线）

要在论文里 **同一设定** 下比较 **x12** 与 **vec9（pose6d）**：

| 设定 | 数据 | ISM | 说明 |
|------|------|-----|------|
| Fair A | `split=test`，`--require-ism` | 有 npz 的 20738 帧 | 与 `eps_mlp_ism_20k.pt` 一致 |
| Fair B | 同上 | 同上 | `train_pose6d.py` 训出的 vec9 模型 |

训练命令见 `diff/tools/train_pose6d_test_ism_fair.sh`（输出如 `eps_mlp_pose6d_test_ism_20k.pt`）。

**注意**：在 test 上训练、在 test 上评，数字偏乐观；更严谨可改为 **train_real 训练 / test 评测**，但 test 侧需 **自行对 train 场景跑 ISM** 才能开 `require-ism`，否则只能不用 ISM 筛选。

## 3. 泛化实验（与 Fair 分开写）

| 设定 | 训练 | 评测 | 典型用途 |
|------|------|------|----------|
| `train_real`，不加 `require-ism` | train 场景 | test | 域泛化，数值通常低于 Fair |

对应权重示例：`eps_mlp_pose6d_trainreal_20k.pt`。

## 4. 论文里如何写「加 ISM 的结果」

- **方法段**：说明 ISM 为上游模块，提供与 BOP 评测一致的 **test 帧上的预测**；本工作在该设定下训练/评估 diffusion pose。
- **实验段**：报告 **ISM + 本文 pose 模块** 在 YCB-Video test 上的指标（与全 test 或 BOP 子集一致）；与 **仅几何/无 ISM 筛选** 的消融分开列表（若做了）。
- **公平性**：对比 **x12 vs vec9** 时用 **同一 split + 同一 require-ism**，避免与 train_real 混在同一表。

## 5. 一键复现脚本

- `diff/tools/train_pose6d_test_ism_fair.sh`：Fair 设定下训练 vec9。
- `diff/tools/run_fair_eval_compare.sh`：对多个 checkpoint 跑 test 全量评估并打印对比表。

## 6. BOP19 官方指标（bop_toolkit）

仓库内 `eval_full_test.py` 的 trans/rot/score **不是** BOP 的 VSD/MSSD/MSPD 与 `bop19_average_recall`。要与 leaderboard 口径一致需：

1. **导出结果 CSV**（BOP19 格式，与 [bop_toolkit `inout.load_bop_results`](https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/inout.py) 一致）：

   ```bash
   cd /path/to/Diff_Sam_6d
   python -m diff.export_bop19_results \
     -c diff/output/eps_mlp_pose6d_test_ism_20k.pt \
     --require-ism \
     --method diffsam
   ```

   默认读取 `Data/BOP/ycbv/test_targets_bop19.json`，写出 `diff/output/diffsam_ycbv-test.csv`，并在同目录写 `.meta.json` 记录条数与策略。

   **策略**：**`--policy match_largest`（默认）** 与 `eval_full_test` 一致，每帧一次 DDPM，仅当最大可见实例的 `obj_id` 与某条 target 一致时写该行。**`--policy per_target`** 对 `test_targets_bop19.json` 中每条 target 各做一次采样（整图条件不变，seed 按 target 区分），并用 GT 确认该帧存在对应 `obj_id`（多实例同 id 时取 mask 面积最大者作 oracle 过滤）；与 BOP 行数对齐，利于涨 recall，**无需重训**；导出仍依赖 GT 过滤，非完全盲测。**`--policy duplicate`** 仅调试。

2. **克隆并配置 [thodan/bop_toolkit](https://github.com/thodan/bop_toolkit)**，在 `bop_toolkit_lib/config.py`（或环境变量）中设置 `datasets_path` 指向含 `ycbv` 的 BOP 根目录、`results_path` 指向你放置 CSV 的目录。

3. 将 `diffsam_ycbv-test.csv` 放到 `results_path` 下，运行：

   ```bash
   python scripts/eval_bop19_pose.py --dataset ycbv --result_filenames diffsam_ycbv-test.csv
   ```

   具体参数以该脚本 `--help` 为准；评测产物通常为 `eval_path` 下的 `scores_bop19.json`（含 `bop19_average_recall` 等）。

   **无显示 / WSL（要算完整 VSD + 官方 `bop19_average_recall`）**：Vispy 默认 EGL 在纯 WSL 常失败。推荐编译 [thodan/bop_renderer](https://github.com/thodan/bop_renderer)（系统 `gcc/g++` + 系统 `libosmesa6-dev`），例如：

   ```bash
   cd bop_renderer   # 仓库根目录
   rm -rf build && CXX=/usr/bin/g++ CC=/usr/bin/gcc cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
     -DOSMESA_LIBRARY=/usr/lib/x86_64-linux-gnu/libOSMesa.so \
     -DOSMESA_INCLUDE_DIR=/usr/include \
     -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   cmake --build build -j$(nproc)
   ```

   评测前设置 `BOP_RENDERER_PATH` 指向 **`bop_renderer/build`**（内含 `bop_renderer.cpython-*.so`），并改用 C++ 渲染：

   ```bash
   export BOP_RENDERER_PATH=/path/to/bop_renderer/build
   python scripts/eval_bop19_pose.py --result_filenames YOUR_ycbv-test.csv --renderer_type cpp --num_workers 1
   ```

   完整三项跑完后会生成 `eval_path/<result_name>/scores_bop19.json`，其中 **`bop19_average_recall`** 为 VSD/MSSD/MSPD 三项平均（单数据集 YCB-V；与 BOP Core **AR_core** 多集平均仍不是同一列）。

   **仅 MSSD+MSPD（无 VSD）**：`eval_calc_errors.py` 设 `--error_type=mssd` 与 `mspd`，再用 `python -m diff.bop19_mssd_mspd_scores ...`（见上文）。结果 CSV 须符合 `parse_result_filename`：**仅一个下划线** 分隔方法名与 `ycbv-test`。

若未来模型能按 **每个 target 物体** 分别预测位姿，再改为对 `test_targets_bop19` 中每条 target 各写一行即可与「每物体一估计」的设定对齐。
