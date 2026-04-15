# sam6d（Diff + ISM）

本目录包含：

- **`Diff_Sam_6d/`**：条件扩散位姿训练/评测、ISM 启动脚本、BOP 工具集成等（见 `Diff_Sam_6d/PROPOSAL.md`）。
- **`SAM-6D-official/`**：官方 SAM-6D 仓库副本（含对 ISM 的本地修改，如 `model/utils.py`、`model/detector.py` 等）。

**数据与权重不在仓库中。** 克隆后请将 BOP 数据放到 `Diff_Sam_6d/Data/`，并按各子项目 README 下载 ISM 检查点；或建立与本地一致的 `checkpoints` 符号链接。

上传到 GitHub 的步骤与所需账号/Token：**见 [GITHUB_SETUP.md](./GITHUB_SETUP.md)**。
