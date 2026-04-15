# 将本仓库上传到 GitHub（不含数据）

仓库根目录：`/home/yan/work/sam6d`（包含 `Diff_Sam_6d` 与 `SAM-6D-official` 的本地改动）。

`.gitignore` 已排除：**BOP 数据、ISM npz 与 log、训练 checkpoint/日志、SAM 预训练权重目录** 等大文件。

---

## 你需要准备什么

### 1. GitHub 账号

在 [github.com](https://github.com) 注册并登录。

### 2. 在 GitHub 上新建一个空仓库

1. 右上角 **+** → **New repository**  
2. **Repository name**：例如 `sam6d-diff-ism`（自定，勿与已有仓库重名）  
3. 选 **Public** 或 **Private**  
4. **不要**勾选 “Add a README / .gitignore / license”（本地已有代码时避免冲突）  
5. 点 **Create repository**

记下页面上的地址，二选一：

- **HTTPS**：`https://github.com/<你的用户名>/<仓库名>.git`  
- **SSH**：`git@github.com:<你的用户名>/<仓库名>.git`（需先配置 SSH 公钥）

### 3. 本机身份验证（推送时用）

任选一种：

| 方式 | 你需要做的 |
|------|------------|
| **HTTPS** | 在 GitHub → **Settings → Developer settings → Personal access tokens** 创建 **Fine-grained** 或 **classic** token，勾选 `repo`；推送时密码处粘贴 token。 |
| **SSH** | 本机 `ssh-keygen`，把 `~/.ssh/id_ed25519.pub` 内容贴到 GitHub → **Settings → SSH and GPG keys**；测试 `ssh -T git@github.com`。 |

GitHub 已逐步弃用「账号密码推送」，**推荐 SSH 或 HTTPS + PAT**。

### 4. 本地不要提交的东西（已由 .gitignore 处理）

- `Diff_Sam_6d/Data/`（约百 GB 级 BOP、模板、渲染模型等）  
- `Diff_Sam_6d/diff/output/`（checkpoint、log、可视化）  
- `SAM-6D-official/.../Instance_Segmentation_Model/log/`  
- `SAM-6D-official/.../Instance_Segmentation_Model/checkpoints/`（SAM / DINOv2 等权重）

克隆仓库的人需要：按各子目录 README 自行下载 **BOP 数据** 与 **ISM 权重**。

---

## 嵌套 Git 仓库（bop_renderer / bop_toolkit / SAM-6D-official）

若子目录里已有 `.git`，外层 `git add` 只会加入一个 **gitlink 指针**，别人克隆后子目录是空的。  
若你希望 **把所有源码当作一个仓库提交**（推荐、简单），在首次 `commit` **之前**执行一次：

```bash
cd /home/yan/work/sam6d
rm -rf Diff_Sam_6d/bop_renderer/.git
rm -rf Diff_Sam_6d/bop_toolkit/.git
rm -rf SAM-6D-official/.git
```

（只删除子模块的 **Git 元数据**，不删文件内容。）之后重新 `git add .`。

若你 **刻意** 用官方远程跟踪 `bop_toolkit`，可改回 `git submodule`，此处不展开。

---

## 首次提交与推送（在 `/home/yan/work/sam6d` 下执行）

若本机 **尚未** `git init`，先执行一次 `git init`。若已存在 `.git`，从下面 **配置邮箱姓名** 开始即可。

```bash
cd /home/yan/work/sam6d

# 1) 提交者身份（GitHub 要求有合法 name/email）
git config user.email "你的邮箱@example.com"
git config user.name "你的名字或 GitHub 昵称"

# 2) 索引与提交
git add .
git status   # 确认没有 Data/、diff/output/、ISM log/、checkpoints 实体等大路径
git commit -m "Initial commit: Diff_Sam_6d + SAM-6D-official (no data/weights)"

# 3) 默认分支名（与 GitHub 新建仓库一致）
git branch -M main

# 4) 远程与推送
git remote add origin https://github.com/<用户名>/<仓库名>.git
# 或: git remote add origin git@github.com:<用户名>/<仓库名>.git

git push -u origin main
```

若远程仓库创建时带了 README 导致拒绝推送，可用：

`git pull origin main --rebase` 再 `git push`，或按 GitHub 页面提示操作。

---

## 可选：只上传 Diff_Sam_6d

若希望 **SAM-6D 完全用上游官方仓库**、只托管自己的 `Diff_Sam_6d`：

```bash
cd /home/yan/work/sam6d/Diff_Sam_6d
# 把 .gitignore 复制进来或单独写一份，忽略 Data/ 与 diff/output/
git init
# ...
```

注意：当前脚本里 `ISM_ROOT` 默认指向旁边的 `SAM-6D-official`，克隆者需自行克隆官方 SAM-6D 并设置环境变量 `ISM_ROOT`。

---

## 大文件提醒

若将来需要把 **单个超过 ~100MB** 的文件纳入 Git，应使用 [Git LFS](https://git-lfs.github.com/)，否则 GitHub 会拒绝推送。权重与数据集应继续放在网盘 / Release 资产 / 说明链接中，不要直接进普通 Git。
