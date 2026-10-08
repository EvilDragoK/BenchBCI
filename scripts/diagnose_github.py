#!/usr/bin/env python3
"""一条命令诊断"为什么 git push 到 GitHub 失败"（Windows / macOS / Linux 通用，无第三方依赖）。

它把到 GitHub 的连接拆成三段分别测试：
1. **TCP**：端口能不能建连（`socket.create_connection`）
2. **TLS**：HTTPS 能不能握手（`ssl.wrap_socket`，带 SNI=github.com）
3. **SSH**：SSH 协议能不能握手（`ssh -T`，返回 "Permission denied" 也算通）

典型结论：
- TCP 通 + TLS 超时 + SSH 通  → 到 GitHub 的 HTTPS 被网络干扰，**改用 SSH 推送**即可
- 三段全不通                    → 整体出网受限，需要用代理/VPN，或改用 Gitee 中转
- TLS 通但 git 报 HTTP2 framing → 多半是 HTTP/2 兼容问题，`git config --global http.version HTTP/1.1`

用法::

    python scripts/diagnose_github.py
    python scripts/diagnose_github.py --url https://github.com/<用户名>/<仓库>.git
"""
from __future__ import annotations

import argparse
import socket
import ssl
import subprocess
import sys
import time

TCP_TARGETS = [("github.com", 443), ("github.com", 22), ("ssh.github.com", 443),
               ("codeload.github.com", 443)]
TLS_TARGETS = ["github.com", "ssh.github.com"]
CONTROL = "www.baidu.com"          # 对照组：确认本机网络/TLS 栈本身是好的


def tcp(host: str, port: int, timeout: float = 8.0) -> tuple[bool, str]:
    t0 = time.perf_counter()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, f"{time.perf_counter() - t0:.1f}s"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}（{time.perf_counter() - t0:.1f}s）"


def tls(host: str, port: int = 443, timeout: float = 10.0) -> tuple[bool, str]:
    ctx = ssl.create_default_context()
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as ss:
                return True, f"{ss.version()}（{time.perf_counter() - t0:.1f}s）"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:60]}（{time.perf_counter() - t0:.1f}s）"


def ssh(host: str = "github.com", port: int = 22, timeout: float = 12.0) -> tuple[bool, str]:
    """返回 (是否可达, 说明)。"Permission denied (publickey)" 也算可达（只是没配密钥）。"""
    cmd = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
           "-o", f"ConnectTimeout={int(timeout)}", "-p", str(port), f"git@{host}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        out = (r.stdout + r.stderr).strip().splitlines()
        text = out[-1] if out else ""
        ok = "successfully authenticated" in text or "Permission denied" in text
        return ok, text[:80]
    except subprocess.TimeoutExpired:
        return False, f"超时（>{timeout:.0f}s）"
    except FileNotFoundError:
        return False, "没装 ssh 客户端"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def git_ls_remote(url: str | None, http_version: str | None = None) -> tuple[bool, str]:
    if not url:
        return False, "未提供 --url，跳过"
    cmd = ["git"]
    if http_version:
        cmd += ["-c", f"http.version={http_version}"]
    cmd += ["ls-remote", url]
    import os

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90, env=env)
        text = (r.stderr or r.stdout).strip().splitlines()
        tail = text[-1] if text else ""
        ok = r.returncode == 0
        # 私有仓库未认证时 "could not read Username" → HTTP 层其实是通的
        if "could not read Username" in tail or "Authentication failed" in tail:
            return True, f"HTTP 层可达（需要认证）：{tail[:60]}"
        return ok, tail[:90]
    except subprocess.TimeoutExpired:
        return False, "超时（>90s）"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:60]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="诊断到 GitHub 的连接（TCP / TLS / SSH 三段）")
    ap.add_argument("--url", help="仓库地址，如 https://github.com/user/repo.git（可选）")
    args = ap.parse_args(argv)

    print("=" * 74)
    print("GitHub 连接诊断（结果里出现 ✘ 的一段就是问题所在）")
    print("=" * 74)

    print("\n[1/4] TCP 端口连通性")
    res_tcp = {}
    for host, port in TCP_TARGETS:
        res_tcp[(host, port)] = tcp(host, port)
        ok, info = res_tcp[(host, port)]
        print(f"  {'✔' if ok else '✘'} {host}:{port:<4} {info}")

    print("\n[2/4] TLS（HTTPS）握手")
    res_tls = {}
    for host in TLS_TARGETS:
        res_tls[host] = tls(host)
        ok, info = res_tls[host]
        print(f"  {'✔' if ok else '✘'} {host:22s} {info}")
    ctrl = tls(CONTROL)
    print(f"  {'✔' if ctrl[0] else '✘'} {CONTROL:22s} {ctrl[1]}"
          f"   ← 对照组（它不通说明本机网络/TLS 有问题）")

    print("\n[3/4] SSH 协议握手（“Permission denied” 也算可达）")
    res_ssh = {}
    for host, port in (("github.com", 22), ("ssh.github.com", 443)):
        res_ssh[(host, port)] = ssh(host, port)
        ok, info = res_ssh[(host, port)]
        print(f"  {'✔' if ok else '✘'} {host}:{port:<4} {info}")

    print("\n[4/4] git 实际访问（HTTPS）")
    res_git = {}
    if args.url:
        for ver, label in ((None, "默认"), ("HTTP/1.1", "HTTP/1.1")):
            res_git[label] = git_ls_remote(args.url, ver)
            ok, info = res_git[label]
            print(f"  {'✔' if ok else '✘'} {label:9s} {info}")
    else:
        print("  （跳过：没给 --url，加 `--url https://github.com/<用户名>/<仓库>.git` 可测到 HTTP 层）")

    git_ok = any(v[0] for v in res_git.values()) if res_git else None

    # ---------------- 结论（全部复用上面的实测结果，不再重测，避免自相矛盾）----------------
    tcp_ok = res_tcp.get(("github.com", 443), (False, ""))[0]
    tls_ok = res_tls.get("github.com", (False, ""))[0]
    ssh_ok = any(v[0] for v in res_ssh.values())
    print("\n" + "=" * 74)
    print("结论与建议")
    print("=" * 74)
    print(f"· TCP(443)={'通' if tcp_ok else '不通'}  TLS(HTTPS)={'通' if tls_ok else '不通'}  "
          f"SSH={'通' if ssh_ok else '不通'}  本机对照网站TLS={'通' if ctrl[0] else '不通'}"
          + (f"  git(HTTPS)={'至少一次成功' if git_ok else '全部失败'}" if git_ok is not None else ""))
    if tcp_ok and not tls_ok and ssh_ok:
        print("\n· 判断：**到 GitHub 的 HTTPS(TLS) 被网络干扰**（TCP 能连、TLS 无响应），"
              "而 SSH 通道正常。")
        print("  → 改走 SSH（最省事，一次配好长期可用）：")
        print("      ssh-keygen -t ed25519 -C \"你的邮箱\"          # 已有密钥可跳过")
        print("      pbcopy < ~/.ssh/id_ed25519.pub                # Windows: "
              "type %USERPROFILE%\\.ssh\\id_ed25519.pub | clip")
        print("      # 打开 https://github.com/settings/ssh/new 粘贴 → Add SSH key")
        print("      ssh -T git@github.com                         # 期望：Hi <用户名>! You've successfully authenticated")
        print("      git remote set-url origin git@github.com:<用户名>/<仓库>.git")
        print("      git push -u origin main")
        print("  （若 22 端口哪天不通，把下面三行写进 ~/.ssh/config：")
        print("      Host github.com / HostName ssh.github.com / Port 443 / User git ）")
    elif tls_ok:
        print("\n· HTTPS 目前可用。如果 git 仍报 `Error in the HTTP2 framing layer`（时好时坏），执行：")
        print("      git config --global http.version HTTP/1.1")
        print("      git config --global http.postBuffer 524288000")
        print("  → 然后重试 push；仍然不稳就直接用 SSH（上面那组命令）。")
    else:
        print("\n· HTTPS 与 SSH 都不通 → 整体出网受限，三种办法：")
        print("  1) 先开代理/VPN（Clash 等）再 push：")
        print("       git config --global http.https://github.com.proxy http://127.0.0.1:7890")
        print("  2) 推 Gitee 中转，再用 GitHub 网页 New → Import a repository（服务端拉取）")
        print("  3) 直接把项目拷给队友（U 盘/内网共享），或让队友从 Gitee 克隆")
    if git_ok is not None and git_ok and not tls_ok:
        print("\n· 注意：本机出现了「TLS 测试不通、但 git 偶尔成功」——说明这条链路是"
              "**时通时断**（常见于运营商的间歇性干扰）。")
        print("  这类抖动下，靠重试也能推上去，但**稳定方案仍是 SSH**（或代理）。")
    print("\n提示：队友在同样网络下 `git clone https://github.com/...` 也会时通时断——让他们用 SSH")
    print("      （git clone git@github.com:<用户名>/<仓库>.git）或从 Gitee 克隆。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
