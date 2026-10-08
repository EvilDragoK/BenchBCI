# 上传到 GitHub + Windows 环境配置（完整操作手册）

> 目标：把本机（macOS）的代码推到 GitHub，然后在 **Windows** 电脑上克隆、配环境、跑起来。
> 全文命令都可**直接复制粘贴**：macOS/Linux 用「终端」，Windows 用「PowerShell」（个别处标注 cmd）。
> 项目路径：本机为 `~/Desktop/APP/BCIAPP`，下文用 `BCIAPP` 代指，请按需替换。

---

## 0. 30 秒看懂全流程

```
① 本机(macOS)          git init → commit → push        →  GitHub 仓库
② GitHub              （网页建空仓库 / 或 gh 一条命令）
③ Windows             git clone → venv → pip install   →  设 BCI_DATA_ROOT → 跑自检 → 启动 GUI
```

| 步骤 | 谁做 | 耗时 | 关键点 |
|---|---|---|---|
| 1. 上传前检查 | 本机 | 3 分钟 | **不要上传 `.venv/`、`processed/`、`experiments/`、`configs/llm.yaml`**（已在 `.gitignore`） |
| 2. 推到 GitHub | 本机 | 5 分钟 | 用 PAT 或 SSH 认证；仓库建**空仓库**（不勾 README） |
| 3. 克隆 | Windows | 2 分钟 | 放在**短路径**（如 `D:\code\BCIAPP`），避免 260 字符限制 |
| 4. 环境 | Windows | 10–30 分钟 | `python -m venv .venv` + `pip install -r requirements.txt`（torch 最大） |
| 5. 数据 | Windows | 拷贝即可 | 只需一个环境变量 `BCI_DATA_ROOT`，**不用改任何代码** |
| 6. 验证 | Windows | 5 分钟 | 4 条命令：环境体检 / 单测 / GUI 自检 / 功能演示 |

---

## 1. 上传前检查（3 分钟，避免把数据、密钥、垃圾推上去）

### 1.1 已经排除的东西（`.gitignore` 逐条说明）

| 被排除 | 为什么 | 队友拿到后怎么办 |
|---|---|---|
| `.venv/` | 虚拟环境含上万文件、几百 MB 且与平台强绑定（mac 的 wheel 在 Windows 不能用） | 自己 `python -m venv .venv` 重建 |
| `processed/` | 预处理产物（Epochs/tensor），可用配置文件重新生成 | 跑一次预处理 |
| `experiments/` | 实验记录（权重 `.pt`、指标、截图）体积大 | 自己跑实验生成 |
| `configs/llm.yaml` | **含 API 密钥** | 复制 `configs/llm.example.yaml` 自己填 |
| `__pycache__/` `.pytest_cache/` `*.pyc` `*.egg-info/` | 缓存 | 自动重建 |
| `.DS_Store` `Thumbs.db` `.idea/` `.vscode/` `.codebuddy/` | 系统/IDE 垃圾 | — |
| `*.log` | 日志 | — |

> `.gitattributes` 已把换行符统一为 LF（Windows 检出也不会把 `.py` 改成 CRLF），跨平台协作不会出现"整文件 diff"的假改动。

### 1.2 手动确认没有密钥

```bash
cd ~/Desktop/APP/BCIAPP

# ① 密钥文件是否存在（存在也没关系，它在 .gitignore 里，不会上传）
ls configs/llm.yaml 2>/dev/null && echo "存在 → 下面 git status 时确认它没出现"

# ② 全项目搜一遍可能写死的密钥（排除 .venv / 数据与实验产物）
grep -rn "sk-[A-Za-z0-9]\{16,\}" --exclude-dir=.venv --exclude-dir=processed \
     --exclude-dir=experiments . | head
```

### 1.3 确认体积（GitHub 单文件硬上限 100 MB，单仓库建议 < 1 GB）

```bash
du -sh .                      # 本机 2.6G —— 但绝大部分是 .venv/processed/experiments（都不上传）
find . -type f -size +1M -not -path "./.venv/*" -not -path "./processed/*" \
       -not -path "./experiments/*" -not -path "./.git/*"    # 应该没有输出
```

**实际会进仓库的只有代码本体**（实测）：

```
会进仓库的文件数 = 128
总体积         = 2.43 MB
顶层           = core/ ui/ tests/ docs/ configs/ models_custom/ + main.py README.md
                .gitignore .gitattributes icon.png
```

> 也就是"2.6 G 的项目其实只推 2.4 MB"——前提是 `.venv/ processed/ experiments/` 保持被忽略。

### 1.4 想清楚这些文件要不要分享

| 文件/目录 | 默认 | 说明 |
|---|---|---|
| `models_custom/`（你的自定义模型 .py/.json） | **会上传** | 想自己留着就加进 `.gitignore` |
| `docs/demo_output.txt`（实测输出） | 会上传 | 内含本机绝对路径，无隐私，留着方便对比 |
| `icon.png` | 会上传 | 应用图标 |

---

## 2. 本机（macOS）：初始化仓库并推送到 GitHub

### 2.1 先在 GitHub 建一个**空仓库**

1. 打开 <https://github.com/new>；
2. Repository name 填 `BCIAPP`（或任意名）；
3. **Public / Private 自选**（含未发表实验建议 Private）；
4. **不要**勾 "Add a README file / .gitignore / license"（否则首次 push 会冲突，需要 `git pull --rebase`）；
5. 点 Create repository，记下页面显示的地址：`https://github.com/<你的用户名>/BCIAPP.git`。

### 2.2 提交并推送（复制粘贴即可）

```bash
cd ~/Desktop/APP/BCIAPP

# 1) 初始化（默认分支直接叫 main）
git init
git branch -M main

# 2) 配置提交身份（只需一次；--global 表示以后所有仓库都用）
git config --global user.name  "你的名字"
git config --global user.email "你的邮箱@example.com"

# 3) 看一下到底会提交哪些文件（重点：不该有 .venv/processed/experiments/密钥）
#    想逐条核对某文件是被哪条规则忽略的：git check-ignore -v configs/llm.yaml
git add -A
git status --short | head -40
git status --short | wc -l          # 本仓库实测：128 个文件（约 2.4 MB）

# 4) 提交
git commit -m "feat: BCI Lab —— EEG 预处理/训练/分析平台（初版）"

# 5) 关联远端并推送（把 <用户名> 换成你的）
git remote add origin https://github.com/<用户名>/BCIAPP.git
git push -u origin main
```

> 若第 3 步发现不该提交的文件：`git rm -r --cached 文件或目录` 后再重新 `git add -A`。

### 2.3 认证方式（GitHub 不接受账号密码）

**方式 A：Personal Access Token（HTTPS，最省事）**

1. 打开 <https://github.com/settings/personal-access-tokens/new>（细粒度 token）；
2. Repository access 选 `Only select repositories` → 选 `BCIAPP`；
3. Permissions → Repository permissions → **Contents: Read and write**；
4. 生成后**立刻复制** token（`github_pat_...`，只显示一次）；
5. 回到终端 `git push`，用户名填 GitHub 用户名，**密码处粘贴 token**；
6. 让 macOS 记住它：`git config --global credential.helper osxkeychain`。

**方式 B：SSH 密钥（一次配好，长期免密）**

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"     # 一路回车
pbcopy < ~/.ssh/id_ed25519.pub                      # 复制公钥
# 粘贴到 https://github.com/settings/keys → New SSH key
ssh -T git@github.com                               # 看到 "Hi 用户名!" 即成功
git remote set-url origin git@github.com:<用户名>/BCIAPP.git
git push -u origin main
```

**方式 C：`gh` 一条命令建仓库并推送（最快）**

```bash
brew install gh          # 或 https://cli.github.com
gh auth login            # 选 GitHub.com → HTTPS → 浏览器登录
cd ~/Desktop/APP/BCIAPP
gh repo create BCIAPP --private --source=. --remote=origin --push
```

### 2.4 以后每次更新（日常三连）

```bash
cd ~/Desktop/APP/BCIAPP
git status                 # 看改了啥
git add -A
git commit -m "fix: 论文超参抽取支持 PDF 补充材料"
git push
```

### 2.5 上传常见报错对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `remote origin already exists` | 已加过远端 | `git remote set-url origin <新地址>` |
| `Updates were rejected ... fetch first` | 远端有本地没有的提交（建仓库时勾了 README） | `git pull --rebase origin main` 再 `git push` |
| `Support for password authentication was removed` | 用了账号密码 | 用 §2.3 的 token 或 SSH |
| `Authentication failed` / 403 | token 权限不足或过期 | 重新生成 **Contents: Read and write** 的 token |
| `file is 123.45 MB; this exceeds GitHub's file size limit of 100.00 MB` | 传了大文件 | `git rm --cached 大文件`，加进 `.gitignore`；已进历史则用 `git filter-repo` |
| `RPC failed; curl 18 transfer closed` | 网络中断 | `git config --global http.postBuffer 524288000` 后重推；或改用 SSH |
| `fatal: not a git repository` | 不在项目目录 / 没 `git init` | `cd` 到项目根目录 |
| push 很慢/超时（国内） | 网络 | 走 SSH，或 `git config --global http.proxy http://127.0.0.1:7890`（按你的代理改） |

### 2.6 需要连数据/权重一起传？（一般不必要）

```bash
# 只有在"确实要版本化大文件"时才用（Git LFS 免费额度 1 GB 存储 / 1 GB 月流量）
brew install git-lfs && git lfs install
git lfs track "*.pt" "*.mat" "*.nii.gz"     # 会写入 .gitattributes
git add .gitattributes && git commit -m "chore: 启用 Git LFS"
```

> 建议：**数据集与实验产物不要进仓库**（体积大 + 被试隐私 + 许可限制），用"配置文件 + 环境变量"让对方自己生成。

### 2.65 公钥 vs 私钥：哪一个能上传？（务必分清）

| | 文件 | 能不能给 GitHub / 别人 | 说明 |
|---|---|---|---|
| **公钥** | `~/.ssh/id_ed25519.pub` | ✅ **必须**上传到 GitHub | 它只是"锁"，放在 GitHub 的 Settings → SSH keys 里，用来验证你的身份；泄露无害 |
| **私钥** | `~/.ssh/id_ed25519`（**没有 .pub**） | ❌ **绝对不能** | 它是"钥匙"。谁拿到就能以你的身份推代码；一旦泄露必须立刻作废重生成 |
| **LLM API Key** | `configs/llm.yaml` | ❌ 绝对不能 | 已在 `.gitignore` 里；模板见 `configs/llm.example.yaml` |

**自查命令（随时可跑，用来看"我到底传上去没有"）**

```bash
cd ~/Desktop/APP/BCIAPP

# 1) 仓库里有没有被跟踪的密钥/凭证类文件（应无输出）
git ls-files | grep -Ei "id_rsa|id_ed25519|\.pem$|\.ppk$|\.key$|\.pub$|llm\.yaml|\.env"

# 2) 内容级搜索：已提交的代码里有没有私钥特征串（应无输出）
git grep -n -I -E "BEGIN (OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY" HEAD

# 3) 常见 token 格式（应只命中手册里的占位示例，不能有真 key）
git grep -n -I -E "sk-[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}" HEAD

# 4) 整个历史（含以前所有提交）一起查
for c in $(git rev-list --all); do git grep -l -I "BEGIN OPENSSH PRIVATE KEY" "$c"; done

# 5) 工作区里是否有私钥文件（应该都在 ~/.ssh 下，不在项目里）
grep -rl "BEGIN OPENSSH PRIVATE KEY" . --exclude-dir=.venv --exclude-dir=.git
```

> 本仓库实测（提交 `a6f488b`）：1~5 全部通过——**没有任何私钥或真 token 进入仓库**。
> 曾有过的 `123456`（误生成的私钥）从未 `git add`，已移到 `~/.ssh/unused_backup/`。

**万一哪天真把私钥推上去了（应急三步，越快越好）**

```bash
# ① 先在 GitHub 上作废：Settings → SSH and GPG keys → 删除那把 key（泄露的私钥立即失效）
#    然后本机重新生成一把并重新登记：
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "你的邮箱" -N ""
pbcopy < ~/.ssh/id_ed25519.pub        # 粘到 https://github.com/settings/ssh/new

# ② 从仓库里删掉该文件（若只是最后一个提交且还没 push，amend 即可）
git rm --cached 泄露的文件 && git commit -m "chore: 移除误提交的密钥" && git push

# ③ 若已经 push（尤其 public 仓库）→ 历史里仍有，需要清历史后强推（会改写提交号，队友要重新 clone）：
#    推荐用 git-filter-repo：pip install git-filter-repo
git filter-repo --path 泄露的文件 --invert-paths
git push --force-with-lease origin main
```

> **判断标准很简单**：`skills/密钥/密码` 这类东西一旦推上 public 仓库，就当它**已经泄露**——
> 正确的处置是"作废重发"，而不是"删掉文件再推一次"。

---

### 2.7 推不上去怎么办？先诊断，再选通道（国内网络重点）

**三种典型报错 → 根因 → 处置**（都是网络/认证问题，**不是仓库或代码问题**）：

| 报错 | 根因 | 处置 |
|---|---|---|
| `Error in the HTTP2 framing layer` | HTTPS 链路**时通时断**（中间设备干扰 HTTP/2 分帧，TCP 与 TLS 有时能成、随后被切断） | ① 设 HTTP/1.1 后重试 ② **改 SSH**（稳） |
| `Failed to connect to github.com port 443 after 75000 ms` | 到 `github.com:443` 的 **TLS 被干扰**（TCP 三次握手能过、TLS 无响应；其它网站正常） | **改 SSH**（实测 SSH 22 与 443 都通）/ 开代理 |
| `fatal: could not read Username for 'https://github.com'` | 链路已正常，只是**没有凭证**（没配 token / 没配凭证助手） | 配 PAT（HTTPS）或 SSH key |

**一条命令诊断**（仓库自带，Windows/macOS/Linux 通用，无第三方依赖）：

```bash
python scripts/diagnose_github.py --url https://github.com/<用户名>/<仓库>.git
```

它会分四段实测并给出结论：

```
[1/4] TCP 端口连通性      github.com:443 / github.com:22 / ssh.github.com:443 …
[2/4] TLS（HTTPS）握手   github.com / ssh.github.com / www.baidu.com（对照组）
[3/4] SSH 协议握手        github.com:22 / ssh.github.com:443（"Permission denied" 也算可达）
[4/4] git 实际访问        默认(HTTP/2) / 强制 HTTP/1.1
结论：TCP=通 TLS=不通 SSH=通 → 到 GitHub 的 HTTPS 被干扰，改走 SSH
```

> 这类干扰常常**时通时断**：同一分钟内两次测试可能一次超时、一次成功。所以"偶然推成功"不代表稳定，
> 稳定方案是 **SSH**（或代理）。

#### 方案 A：改走 SSH（推荐，一次配好长期稳定）

```bash
# 1) 生成密钥（已有 ~/.ssh/id_ed25519 可跳过）
ssh-keygen -t ed25519 -C "你的邮箱@example.com"      # 一路回车；可留空口令

# 2) 复制公钥
pbcopy < ~/.ssh/id_ed25519.pub
#   Windows PowerShell:  Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub | Set-Clipboard
#   Windows cmd:         type %USERPROFILE%\.ssh\id_ed25519.pub | clip

# 3) 浏览器打开 https://github.com/settings/ssh/new → New SSH key → 粘贴 → Add SSH key

# 4) 验证（期望：Hi <用户名>! You've successfully authenticated...）
ssh -T git@github.com

# 5) 把远端换成 SSH 再推
cd ~/Desktop/APP/BCIAPP
git remote set-url origin git@github.com:<用户名>/<仓库>.git
git push -u origin main
```

**踩坑记录（都是真事）**

- **`pbcopy < ~/.ssh/id_ed25519.pub` 报 `no such file or directory`**
  → 说明你还没生成密钥（跳过了第 1 步），先执行 `ssh-keygen`。
- **`ssh-keygen` 问 "Enter file in which to save the key（…）：" 时不要输入别的名字**
  → 这里**直接回车**才会存到 `~/.ssh/id_ed25519`；如果输了 `123456` 之类，密钥会生成在**当前目录**
  （也就是项目里！），一旦 `git add -A` **私钥就进仓库了**。补救：
  ```bash
  mv 123456 ~/.ssh/unused_backup/ 2>/dev/null; mv 123456.pub ~/.ssh/unused_backup/ 2>/dev/null
  git status --short          # 确认项目里不再有密钥文件
  ```
  本仓库的 `.gitignore` 已对 `id_rsa / id_ed25519 / *.pem / *.ppk` 做保险。
- **怎么查 GitHub 上登记了哪些公钥**（不发私钥、不登录也能查公开列表）：
  ```bash
  curl -s https://api.github.com/users/<用户名>/keys | python -m json.tool | head -20
  ```
- **确认某把本地公钥是否已登记**（比对指纹，不用肉眼比字符串）：
  ```bash
  ssh-keygen -lf ~/.ssh/id_ed25519.pub          # 得到 SHA256:xxxx
  # 再把上面 API 返回的 key 内容存成 k.pub，ssh-keygen -lf k.pub 比对 SHA256
  ```
- **不要复用别的用途的密钥**：如果 `~/.ssh/` 里已有给云服务器/公司用的密钥，给 GitHub **单独生成一把**
  （那条私钥在别人机器上，等于别人能推你的仓库）。

若哪天 22 端口被挡，让 SSH 走 443 —— 编辑 `~/.ssh/config`（Windows 为 `%USERPROFILE%\.ssh\config`）：

```
Host github.com
  HostName ssh.github.com
  Port 443
  User git
```

#### 方案 B：HTTPS + Personal Access Token

```bash
# 1) 生成 token：https://github.com/settings/personal-access-tokens/new
#    Repository access → 选你的仓库；Permissions → Contents: Read and write
# 2) 让系统记住凭证（macOS 用钥匙串；Windows 用管理器）
git config --global credential.helper osxkeychain      # Windows: manager
# 3) 重试推送：用户名填 GitHub 用户名，密码处**粘贴 token**
git push -u origin main
# 4) 若仍报 HTTP2 framing（抖动）→ 强制 HTTP/1.1 并重试
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
```

#### 网络抖动时的重试循环（macOS / Linux / Git Bash）

```bash
cd ~/Desktop/APP/BCIAPP
for i in 1 2 3 4 5; do
  echo "第 $i 次推送…"
  git push -u origin main && { echo "✔ 成功"; break; }
  sleep 5
done
```

#### 方案 C：Gitee 中转（队友也在国内时最省心）

你本机连不上 GitHub 时，**GitHub 服务器本身能连 Gitee**，于是：

```bash
# 1) 在 https://gitee.com 建一个空仓库，然后推到 Gitee
git remote add gitee https://gitee.com/<你的Gitee名>/<仓库名>.git
git push gitee main

# 2) GitHub 网页 → New repository → 拉到最底 "Import a repository"
#    填 https://gitee.com/<你的Gitee名>/<仓库名>.git  → GitHub 服务端拉取（绕开你本机网络）
```

> 如果队友也在同一网络，**直接从 Gitee 克隆**往往比 GitHub 更快更稳。

#### 队友（Windows）侧注意

- 优先 **SSH 克隆**：`git clone git@github.com:<用户名>/<仓库>.git`
- 或从 Gitee 克隆；用 HTTPS 直连 GitHub 会碰到同样的抖动
- 详见 §3.2 与 §4 的排查表

#### 顺带：让提交显示在你头像下

`git config --global user.name/user.email` 只是**提交作者**，与登录账号无关。GitHub 只把
"邮箱属于该账号（或在其 Emails 里验证过）"的提交算到你名下：

```bash
# 方式一：把该邮箱加进 GitHub → Settings → Emails 并验证
# 方式二：用 GitHub 提供的匿名邮箱（在 Settings → Emails 页面能查到你的 ID）
git config --global user.email "<ID>+<用户名>@users.noreply.github.com"
```

---

## 3. Windows：克隆 + 环境配置 + 运行

### 3.1 装两样东西

| 软件 | 版本 | 下载 | 安装要点 |
|---|---|---|---|
| **Git for Windows** | 最新 | <https://git-scm.com/download/win> | 默认下一步即可（自带 Git Bash 与凭证管理器） |
| **Python** | 3.11 ~ 3.13，**64 位** | <https://www.python.org/downloads/windows/> | **务必勾选 "Add python.exe to PATH"** |

安装后验证（PowerShell）：

```powershell
git --version
python --version          # 应显示 Python 3.1x.x
python -c "import sys; print(sys.maxsize > 2**32 and '64-bit OK' or '32-bit 不行！')"
```

> 不要用 Microsoft Store 版 Python（路径隔离容易出怪问题）；用 python.org 的安装包。
> 不建议把项目放在 OneDrive 同步目录或含中文/空格的路径里。

### 3.2 克隆（短路径！）

```powershell
# 允许长路径（Windows 260 字符限制，PyTorch/MNE 的嵌套很深）
git config --global core.longpaths true

# 克隆到短路径（推荐 D:\code\BCIAPP，而不是 C:\Users\某某\Documents\...\BCIAPP）
mkdir D:\code -Force | Out-Null
cd D:\code
git clone https://github.com/<用户名>/BCIAPP.git
cd BCIAPP
```

第一次会弹出窗口让你登录 GitHub（Git Credential Manager），或让你粘贴 §2.3 的 token —— 登录成功后 Windows 会记住，以后 `git pull` 免密。

### 3.3 建虚拟环境 + 装依赖

```powershell
cd D:\code\BCIAPP

# 1) 建虚拟环境（克隆下来是没有 .venv 的，必须自己建）
python -m venv .venv

# 2) 激活（PowerShell）
.\.venv\Scripts\Activate.ps1
#    若报"因为在此系统上禁止运行脚本" → 执行下面这行后再激活：
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#    （或用 cmd：.venv\Scripts\activate.bat；或 Git Bash：source .venv/Scripts/activate）

# 3) 升级 pip 并安装依赖（torch 较大，耐心等）
python -m pip install -U pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**要用 NVIDIA 显卡训练**（默认 PyPI 的 torch 可能是 CPU 版）：

```powershell
# 先在终端确认显卡与驱动：nvidia-smi（看右上角 CUDA Version，如 12.4）
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 若输出 False，按 CUDA 版本重装（示例 cu124，去 https://pytorch.org/get-started/locally/ 生成你的版本）
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**可选增强（用不到可以不装）**：

```powershell
pip install openpyxl        # 读 Excel 变量表（否则请把表另存为 CSV）
pip install pyriemann       # 黎曼切空间特征
pip install moabb           # 公开数据集一键下载
pip install autoreject      # 逐 epoch 自动剔除
```

### 3.4 配置数据目录（本项目唯一需要"配置"的东西）

代码里的数据路径 = **环境变量 `BCI_DATA_ROOT` → 没设则用本机默认路径**。所以 Windows 上只要：

```powershell
# 永久生效（写进用户环境变量；设置后要新开一个终端）
setx BCI_DATA_ROOT "D:\BCI\data"

# 只在当前窗口生效（调试时更灵活）
$env:BCI_DATA_ROOT = "D:\BCI\data"
```

**要求的目录结构**（`BCI_DATA_ROOT` 是"数据集根目录"，下面再分数据集）：

```
D:\BCI\data\
├── BNCI2014001\            # BCI Competition IV 2a
│   ├── A01T.mat  A01E.mat  A02T.mat … A09E.mat
└── PhysioNet_MI\           # PhysioNet EEG Motor Movement/Imagery
    ├── S001\S001R01.edf … S001R14.edf
    └── S002\…
```

**数据从哪来**（三者任选，数据本身**不在仓库里**）：

| 方式 | 操作 |
|---|---|
| **① 从你 Mac 上拷**（最省事，也最快） | Mac 上：`rsync -avh ~/Desktop/DL/BCI/data/ /Volumes/U盘/BCI_data/`；Windows 上把 `BCI_data` 整个拷到 `D:\BCI\data` |
| ② 官方下载 BNCI2014001 | BCI Competition IV / BNCI Horizon 2020 数据集页（`001-2014`，含 `A01T.mat`…），解压后放到 `D:\BCI\data\BNCI2014001\` |
| ② 官方下载 PhysioNet MI | `https://physionet.org/files/eegmmidb/1.0.0/`（浏览器或 `wget -r -N -c -np <url>`），得到 `S001\S001R01.edf` 结构，放进 `D:\BCI\data\PhysioNet_MI\` |

**验证是否被识别**：

```powershell
python -c "from core.data_paths import describe; print(describe())"
# 期望看到：数据集根目录 = D:\BCI\data（来源：环境变量 BCI_DATA_ROOT）
#           · BNCI2014001: D:\BCI\data\BNCI2014001（存在）
```

**不想设环境变量的替代方案**（任选一种）：

```powershell
# ① 直接改配置文件（3 个文件的 root 行）——它们已写成 ${BCI_DATA_ROOT:-默认路径}
notepad configs\preprocess_bnci2014001.yaml
# ② 命令行临时覆盖（预处理支持 --root，训练支持 --processed-dir）
python -m core.pipeline configs\preprocess_bnci2014001.yaml --root D:\BCI\data\BNCI2014001
```

> 数据集缺失**不会崩**：相关单测自动 skip，界面「数据管理」页会显示"（不存在）"。

### 3.5 冒烟自检（4 条命令，建议按顺序）

```powershell
# ① 环境体检（依赖是否齐全 + 设备是否可见）
python -c "import mne, torch, numpy, scipy, sklearn, PyQt6; print('mne', mne.__version__, '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

# ② 单测（361 项；缺数据集时 test_bnci/test_physionet 会 skip，属正常）
python -m pytest tests\ -q

# ③ GUI 无头自检（不弹窗，返回码 0 = 0 错误）
$env:QT_QPA_PLATFORM = "offscreen"
python main.py --selftest
Remove-Item Env:QT_QPA_PLATFORM          # 自检完清掉，免得影响后面开界面

# ④ 功能演示（120 项检查，会真跑一遍各功能并打印结果）
python docs\demo_features.py
```

### 3.6 启动 GUI 与命令行

```powershell
# GUI（5 页：数据管理 → 预处理工作台 → 训练控制台 → 结果分析 → 信号分析）
python main.py

# 命令行：预处理（读原始数据 → 生成 Epochs/张量，输出到 processed\）
python -m core.pipeline configs\preprocess_bnci2014001.yaml

# 命令行：训练（传统基线，CPU 也能跑）
python -m core.cli train configs\train_csp_lda.yaml --device cpu

# 命令行：训练（深度学习，用 GPU）
python -m core.cli train configs\train_eegnet.yaml --device cuda --epochs 60

# 实验记录 / 报告
python -m core.cli list
python -m core.cli report experiments\<实验目录>
```

### 3.7 Windows 与 macOS 的差异清单（踩坑前先看）

| 方面 | macOS（本机） | Windows | 本项目的处理 |
|---|---|---|---|
| 训练设备 | `mps`（Apple Metal） | `cuda`（NVIDIA）/ `cpu` | 训练页「检测/选择…」自动列出可用设备；命令行 `--device` |
| 数据路径 | 默认 `~/Desktop/DL/BCI/data` | 需设 `BCI_DATA_ROOT` | `core/data_paths.py` 统一解析（**不用改代码**） |
| 路径分隔符 | `/` | `\` | 代码统一用 `pathlib.Path`，两种都认 |
| 换行符 | LF | 传统 CRLF | `.gitattributes` 统一 LF，`.bat/.cmd` 保持 CRLF |
| 中文字体 | Songti SC | **SimSun（宋体）** | `core/plot_fonts.py` 自动回退，无需设置 |
| 多进程 DataLoader | fork 启动 | spawn（需 `__main__` 保护） | 训练默认 `num_workers=0`，不受影响 |
| 大小写敏感 | 默认不敏感 | 不敏感 | 但 Linux CI 敏感 → 引用模块请照抄大小写 |
| 长路径 | 无限制 | 260 字符 | 克隆到短路径 + `core.longpaths true` |
| 数值结果 | MPS 与 CUDA 有微小差异 | — | 同一实验请固定 `device` 与 `seed` 再比较 |

> **已审计**：`core/` 与 `ui/` 里没有任何 POSIX 专属调用（`os.fork` / `fcntl` / `resource` /
> `termios` / `pty` 均未使用），路径统一走 `pathlib`，子进程用 `sys.executable` 调自身 ——
> 所以"同一份代码在 Windows 上跑"没有平台阻塞项，差异只集中在**设备**与**数据路径**两件事上。

### 3.8 日常同步（Windows 端）

```powershell
cd D:\code\BCIAPP
git pull                      # 拉取 Mac 上的新提交
# 如果本地也改了东西：
git stash                     # 先把手头改动收起来
git pull
git stash pop                 # 再放回来（可能有冲突，按提示改）
```

---

## 4. Windows 常见问题排查

| 症状 | 原因 | 解决 |
|---|---|---|
| `无法加载文件 ... Activate.ps1，因为在此系统上禁止运行脚本` | PowerShell 执行策略 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，或用 cmd 的 `activate.bat` |
| `'python' 不是内部或外部命令` | 没勾 Add to PATH | 重装 Python 勾选 PATH；或临时用 `py -3.13` 代替 `python` |
| `pip install` 很慢 / 超时 | 网络 | 加镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；或设 `pip config set global.index-url` |
| `torch.cuda.is_available()` 是 False | 装成了 CPU 版 / 驱动太旧 | 按 §3.3 用 `--index-url .../cu124` 重装；更新 NVIDIA 驱动 |
| 显存不足 `CUDA out of memory` | batch 太大 | 训练页把 batch 调小，或 `--device cpu` |
| `could not find or load the Qt platform plugin "windows"` | PyQt6 装坏 / 混装了 PyQt5 / 变量残留 | `pip install --force-reinstall PyQt6`；确认没设 `QT_QPA_PLATFORM=offscreen`；`pip list \| findstr PyQt` 只应有一个 |
| GUI 白屏/闪退 | 显卡驱动或远程桌面 | 更新驱动；`QT_OPENGL=software` 再启动：`$env:QT_OPENGL="software"; python main.py` |
| 中文显示成方框 | 缺中文字体 | Windows 自带来体（SimSun）；若被精简，装一份宋体或在 `core/plot_fonts.py` 里加你系统有的字体名 |
| `FileNotFoundError` 且路径很长 | 260 字符限制 | 项目放短路径 + `git config --global core.longpaths true` |
| 测试里有 `skipped` | 数据集不在（`BCI_DATA_ROOT` 没设或路径不对） | `python -c "from core.data_paths import describe; print(describe())"` 看解析结果 |
| `KeyError: 'data'` 读 .mat | 下载的数据集版本/结构不同 | 确认是 `001-2014`（BCI IV 2a）原始 `.mat`，不是预处理过的版本 |
| `matplotlib` 字体缓存警告 | 首次运行 | 可忽略；或 `python -c "import matplotlib.font_manager as f; f._get_fontconfig_fonts.cache_clear()"` |
| 论文抽取读 PDF 报 `pypdf` 相关错误 | 依赖缺失/损坏 PDF | `pip install -U pypdf`；扫描版 PDF 请先 OCR（界面会提示） |
| `git push` 要密码且一直失败 | 用了账号密码 | 用 token 或 SSH（§2.3），Windows 上 Git Credential Manager 首次会弹窗登录 |

---

## 5. 一页速查（Cheat Sheet）——按本仓库的实际配置写死

> **本仓库现状（已配好，命令可直接复制）**
> - macOS 路径：`~/Desktop/APP/BCIAPP`；Windows 建议：`D:\code\BCIAPP`
> - 远端：`git@github.com:EvilDragoK/BenchBCI.git`（**SSH**，密钥已配好，**push 不再需要 token**）
> - 分支：`main`（已跟踪 `origin/main`）
> - 仓库目前是 **public**：**clone 不需要凭证**，只有 push 需要（用 SSH key）

---

### 5.1 macOS：推送（你自己这台机器）

**首次推送（留档，已完成，仅换新机器时参考）**

```bash
cd ~/Desktop/APP/BCIAPP
git init && git branch -M main
git add -A && git status --short | wc -l        # 本仓库应为 128
git commit -m "feat: BCI Lab —— EEG 预处理/训练/分析平台（初版）"
git remote add origin git@github.com:EvilDragoK/BenchBCI.git
git push -u origin main

# 若还没配 SSH key（报 no such file or directory 就是这种情况）：
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -C "your@email.com" -N ""   # 文件名处务必直接回车
pbcopy < ~/.ssh/id_ed25519.pub        # 粘贴到 https://github.com/settings/ssh/new
ssh -T git@github.com                 # 期望：Hi EvilDragoK! You've successfully authenticated
```

**日常推送（最常用，改完代码三行搞定）**

```bash
cd ~/Desktop/APP/BCIAPP
git add -A && git commit -m "fix: 描述这次改了什么" && git push
```

**推荐版（先看清再提交，避免误传密钥/数据）**

```bash
cd ~/Desktop/APP/BCIAPP
git status --short                     # ① 改了哪些文件（不该出现 .venv/processed/experiments/密钥）
git diff --stat                        # ② 大致改了多少行
git add -A                             # ③ 全部暂存（只提交部分文件：git add core/llm.py ui/train_page.py）
git commit -m "feat: 从论文 PDF 抽取训练超参并绑定实验记录"
git push                               # ④ 推送（SSH，秒级）
git log --oneline -3                   # ⑤ 确认提交历史
```

**只想推某个文件 / 排除某个文件**

```bash
git add core/llm.py tests/test_llm.py                  # 只提交这两个
git add -A -- ':!docs/demo_output.txt'                 # 全部，但排除这个文件
git restore --staged 某文件                            # 撤销暂存（还没 commit 时）
```

**推送失败时（HTTPS 抖动 / 认证问题）**

```bash
python scripts/diagnose_github.py --url https://github.com/EvilDragoK/BenchBCI.git

# 已经用 SSH 的话通常不会失败；偶发网络抖动就重试：
for i in 1 2 3 4 5; do git push && break; sleep 5; done

# 确认远端到底收到了什么：
git ls-remote origin refs/heads/main
```

---

### 5.2 Windows：拉取（队友那台机器）

**首次：克隆 + 搭环境（复制整段，逐段执行）**

```powershell
# ① 一次性设置
git config --global core.longpaths true

# ② 克隆（二选一）
mkdir D:\code -Force | Out-Null; cd D:\code
git clone https://github.com/EvilDragoK/BenchBCI.git     # A. HTTPS：public 仓库，clone 不需要凭证
git clone git@github.com:EvilDragoK/BenchBCI.git         # B. SSH：更稳（队友需把自己的公钥加到 GitHub）
cd BenchBCI

# ③ 虚拟环境 + 依赖
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # 报"禁止运行脚本" → Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -U pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ④ 数据目录（只设一个环境变量，代码不用改）
setx BCI_DATA_ROOT "D:\BCI\data"     # 设完**新开一个终端**
python -c "from core.data_paths import describe; print(describe())"

# ⑤ 验证
python -m pytest tests\ -q                                  # 361 项（缺数据会 skip，正常）
$env:QT_QPA_PLATFORM="offscreen"; python main.py --selftest   # GUI 自检，退出码 0
python docs\demo_features.py                                # 120 项功能演示
python main.py                                              # 启动界面
```

**日常拉取（每次开工前先做）**

```powershell
cd D:\code\BCIAPP
git pull --rebase --autostash      # 拉最新代码；本地有未提交改动会自动暂存/恢复
```

**只拉取、不碰本地改动（更保守）**

```powershell
git stash push -m "本地临时改动"
git pull --rebase
git stash pop
```

---

### 5.3 后续修改后，怎么重新上传（macOS 端标准流程）

```bash
cd ~/Desktop/APP/BCIAPP

# 0) 先同步远端（多人协作时必做，避免 push 被拒）
git pull --rebase --autostash

# 1) 看一眼改了什么
git status --short
git diff --stat

# 2) 暂存（git add -A 会自动处理新增/修改/删除/改名）
git add -A

# 3) 提交（信息写清楚；建议前缀：feat / fix / docs / refactor / test / chore）
git commit -m "feat: 时频分析支持 ERD/ERS 归一化与频段地形图"

# 4) 推送
git push
```

**几种常见情况**

| 情况 | 命令 |
|---|---|
| 只想提交部分文件 | `git add 文件1 文件2` → `git commit -m "..."` |
| 删了文件/改了文件名 | `git add -A`（自动记录删除与改名） |
| 提交前发现漏了一个文件 | `git add 漏的文件` → `git commit --amend --no-edit`（**仅限还没 push**） |
| 提交信息写错了（还没 push） | `git commit --amend -m "新信息"` |
| 已经 push 了想撤销这次改动 | `git revert <commit>`（生成一次反向提交，安全）→ `git push` |
| 撤销某个文件的工作区改动（回到上次提交） | `git restore 文件`（**会丢掉未提交的修改**） |
| 确认某文件是否被忽略 | `git check-ignore -v processed/ configs/llm.yaml` |
| 想临时把文件排除出提交 | `git add -A -- ':!文件'` |

**push 被拒（远端有你没有的提交）**

```bash
git pull --rebase        # 把你的提交"叠"到远端最新之上
git push
```

**出现冲突（提示 `CONFLICT`）**

```bash
git status                       # 列出冲突文件
# 打开冲突文件，编辑 <<<<<<< ======= >>>>>>> 之间的内容，保留你要的
git add 冲突文件
git rebase --continue            # 若之前是 pull --rebase
git push
# 想放弃这次 rebase 回到原样：
git rebase --abort
```

**提交信息规范（建议）**

```
feat:  新功能（如 feat: 论文 PDF 拖拽抽取训练超参）
fix:   修 bug（如 fix: 随机种子的长连接词写法漏识别）
docs:  文档（如 docs: 补充 Windows 环境配置手册）
test:  测试（如 test: 新增 14 项路径可移植性测试）
chore: 杂项（如 chore: 更新 .gitignore 密钥保险）
```

**多个队友协作（可选的分支流程）**

```bash
git switch -c feature/timefreq-batch      # 开分支做新功能
# …改代码、提交…
git push -u origin feature/timefreq-batch # 推到远端分支
# 然后在 GitHub 网页点 "Compare & pull request" 发起 PR，合并后再 git switch main && git pull
```

---

### 5.4 常用命令小抄（记不住就抄）

```bash
git status                  # 当前改动概览
git status -sb              # 简版（含分支与领先/落后几个提交）
git diff                    # 未暂存的改动细节
git diff --staged           # 已暂存的改动细节
git log --oneline -10       # 最近 10 次提交
git show <hash> --stat      # 某次提交改了哪些文件
git remote -v               # 远端地址（本仓库应为 git@github.com:EvilDragoK/BenchBCI.git）
git branch -vv              # 分支与跟踪关系
git ls-remote origin        # 远端所有分支/标签
git fetch --all --prune     # 拉取远端信息但不动工作区
```

---

### 5.5 出问题时

```bash
python scripts/diagnose_github.py --url https://github.com/EvilDragoK/BenchBCI.git   # 网络三段诊断
```

- 推送报错 → §2.7（HTTPS 抖动 / 认证 / SSH 切换）
- Windows 报错 → §4（执行策略、pip 慢、CUDA、Qt 插件、路径过长…）
- 忘了命令 → 本节

---

## 6. 附录：为了让"换机器不用改代码"，本轮做了两处改动

| 改动 | 文件 | 作用 |
|---|---|---|
| **数据路径统一解析** | 新增 `core/data_paths.py`（`BCI_DATA_ROOT` → 默认路径兜底） | 界面「数据管理」页默认目录、`tests/test_bnci.py`、`tests/test_physionet.py`、`docs/demo_features.py` 全部改为读它；Windows 只设一个环境变量即可 |
| **配置支持 `${VAR:-默认}` 展开** | `core/pipeline.py` 的 `load_config` | 3 个 `configs/preprocess_*.yaml` 的 `root` 写成 `${BCI_DATA_ROOT:-~/Desktop/DL/BCI/data}/BNCI2014001`：本机不设变量照旧、别人设了变量即生效 |

配套测试：`tests/test_paths.py`（14 项：默认/覆盖/Windows 风格取值、`${VAR}`/`${VAR:-默认}` 展开、**3 个预处理配置必须可移植**、以及"代码里不允许残留未受保护的本机绝对路径"的回归守卫）。

```powershell
python -m pytest tests\test_paths.py -v      # Windows 上也能跑（不依赖数据集）
```

---

## 7. 隐私与安全清单（public 仓库必看）

### 7.1 上传前用这 5 条自查

```bash
cd ~/Desktop/APP/BCIAPP

# ① 有没有密钥/凭证类文件被跟踪（应无输出）
git ls-files | grep -Ei "id_rsa|id_ed25519|\.pem$|\.ppk$|\.key$|\.pub$|llm\.yaml|\.env"

# ② 内容里有私钥特征串吗（应无输出）
git grep -n -I -E "BEGIN (OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY" HEAD

# ③ 有真 token 吗（只应有手册里的占位示例）
git grep -n -I -E "sk-[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,}" HEAD

# ④ 有本机用户名/绝对家目录吗（应无输出，本项目已全部改为 ~ / Path.home()）
git grep -n -I -E "/Users/|/home/[a-z]" HEAD

# ⑤ 有没有被试数据/实验产物（应无 .npy/.mat/.fif/.pt/.db/.csv）
git ls-tree -r --name-only HEAD | grep -Ei "\.npy$|\.mat$|\.fif$|\.pt$|\.db$|\.csv$"
```

### 7.2 public 仓库会暴露什么（客观事实）

| 会被公开 | 说明 | 能否避免 |
|---|---|---|
| 代码与文档内容 | 这是目的本身 | — |
| **提交作者姓名 + 邮箱** | git 元数据，永久留在历史里 | 用 GitHub 匿名邮箱：`git config user.email "<ID>+<用户名>@users.noreply.github.com"`（仍会归属到你账号） |
| **文件里的绝对路径** | 会带出系统用户名（如 `/Users/xxx/…`） | 统一写成 `~` / `Path.home()`（本项目已做） |
| 仓库可见性 | `public` = 全网可见 + 可被搜索/爬取 | 改用 Private + 加协作者 |
| 你的账号名与显示名 | GitHub 账号本就是公开的 | — |

**不会**因为 push 而暴露：私钥、SSH 私钥、API Key、`processed/`、`experiments/`、被试数据、`configs/llm.yaml`（都在 `.gitignore`）。

### 7.3 两个彻底方案

**方案一：改成 Private（最省事，推荐先做）**

网页：仓库 → **Settings → General → 最下方 Danger Zone → Change visibility → Make private**；
然后 **Settings → Collaborators → Add people** 把 Windows 队友加进来（他们就能 `clone`/`pull`）。
> 注意：public → private 后，若队友之前已 clone，只需继续 `git pull` 即可（地址不变，改用 SSH/HTTPS 认证）。

**方案二：保持 public，但把历史里的痕迹也清掉**（本项目只有一个提交，很容易）

```bash
cd ~/Desktop/APP/BCIAPP

# ① 本仓库用 GitHub 匿名邮箱（`111974444+EvilDragoK@users.noreply.github.com` 会被 GitHub 认作你）
git config user.email "111974444+EvilDragoK@users.noreply.github.com"

# ② 把清理后的内容并入唯一那个提交，并重置作者信息
git add -A
git commit --amend --reset-author -m "feat: BCI Lab —— EEG 预处理/训练/分析平台（初版）"

# ③ 强推（历史被改写；目前只有你一个人在推、队友还没 clone，所以安全）
git push --force-with-lease origin main
```

> **已经有人 clone 过怎么办**：让对方在本地执行 `git fetch origin && git reset --hard origin/main`
> （或直接删掉重新 clone）。
> **已经被爬虫/缓存抓走怎么办**：public 仓库一旦公开就当"已泄露"——邮箱这类信息改历史也难保证彻底，所以
> 真正在意隐私就选**方案一（Private）**。

**强推后核对清单（照抄即可，全部应为期望值）**

```bash
cd ~/Desktop/APP/BCIAPP
git fetch origin
echo "远端 = $(git rev-parse origin/main)   本地 = $(git rev-parse HEAD)"        # 两者应相同
git log origin/main --format='%an <%ae>' -1                                    # 应是匿名邮箱
echo "含本机路径文件数 = $(git grep -l -I '/Users/[a-z]' origin/main | wc -l)"   # 0
echo "含邮箱文件数     = $(git grep -l -I '@163.com' origin/main | wc -l)"       # 0
echo "密钥类文件数     = $(git ls-tree -r --name-only origin/main | grep -Eci 'id_rsa|id_ed25519|\.pem$|\.ppk$|llm\.yaml|\.env|__pycache__')"   # 0
echo "数据类文件数     = $(git ls-tree -r --name-only origin/main | grep -Eci '\.npy$|\.mat$|\.fif$|\.pt$|\.db$|\.csv$')"                  # 0
git status -sb                                                                  # 不应有 ahead/behind
```

> **注意（改历史的局限）**：`git push --force` 之后，旧提交从分支历史里消失了，但 GitHub 仍会
> **按具体 hash 短期保留该对象**（`GET /repos/<owner>/<repo>/commits/<旧hash>` 可能仍返回 200），
> 第三方也可能早已缓存过。因此：
> - 一般隐私（本机路径、用户名）→ 改历史 + 强推即可；
> - 真正敏感的东西（私钥、API Key、被试数据）→ **必须先去源头作废/轮换**，再清历史；
> - 想连"悬挂对象"也不留 → 把仓库改成 **Private**（推荐），或**删库重建**（空仓库不含任何旧对象）。

### 7.4 方案 B：删库重建（最彻底，连"悬挂对象"一起清掉）

**适用**：仓库刚建不久、提交很少、没有 issue/PR/star、还没给别人长期 clone 过
（本仓库 2026-10-08 正是如此，代价极低）。

**第 0 步 本地准备（本机）**

```bash
cd ~/Desktop/APP/BCIAPP
git status --short                              # 应为空（无未提交改动）
git log --format='%h  %an <%ae>' -1             # 作者应是匿名邮箱
# 若还有未提交改动，并入唯一提交，保持"一个干净提交"：
git add -A && git -c user.email="111974444+EvilDragoK@users.noreply.github.com" \
    commit --amend --reset-author --no-edit
```

**第 1 步 删除 GitHub 上的仓库**

仓库页 → **Settings → 最下方 Danger Zone → Delete this repository** → 按提示输入 `EvilDragoK/BenchBCI` 确认。

**第 2 步 新建一个空仓库（同名，地址不变）**

<https://github.com/new> → Repository name 填 `BenchBCI` → 可见性选 **Private**（推荐，隐私可控）
或 Public → **不要勾** "Add a README / .gitignore / license" → Create repository。

**第 3 步 推送（空仓库 → 普通 push，不需要任何 force）**

```bash
cd ~/Desktop/APP/BCIAPP
git fetch --prune origin          # 清掉指向旧仓库的过期远程引用（可能报 404，正常）
git push -u origin main
```

> 同名仓库地址不变，无需改 `origin`。若换了名字：
> `git remote set-url origin git@github.com:<用户名>/<新仓库名>.git`

**第 4 步 核对（直接复制）**

```bash
cd ~/Desktop/APP/BCIAPP
echo "远端 = $(git ls-remote origin refs/heads/main | cut -f1)   本地 = $(git rev-parse HEAD)"
git ls-tree -r --name-only origin/main | wc -l                    # 129
git grep -l -I '/Users/[a-z]' origin/main | wc -l                 # 0
git grep -l -I '@163.com' origin/main | wc -l                     # 0
# 旧提交的悬挂对象应彻底消失（新仓库里不存在这个对象）
curl -s -o /dev/null -w "%{http_code}\n" \
  https://api.github.com/repos/EvilDragoK/BenchBCI/commits/a6f488b427ad8b3a364b2638ca0e2a5189984753   # 应 404
```

**第 5 步（Private 仓库才需要）加协作者**

Settings → **Collaborators** → Add people → 输入队友的 GitHub 用户名 → 对方接受邀请后即可 clone/pull。

**第 6 步 队友重新克隆**（旧仓库已删除，之前的 clone 全部失效）

```powershell
cd D:\code; git clone git@github.com:EvilDragoK/BenchBCI.git
```

> 提醒：删库会把该仓库的 **issue / PR / star / watch 一起删掉**（本仓库没有这些，无损失）。

### 7.4 顺带：本机杂项

```bash
# 本机 __pycache__ 里有绝对路径（未推送，但分享文件夹时会带出去）；可安全清理，会自动重建
find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

---
