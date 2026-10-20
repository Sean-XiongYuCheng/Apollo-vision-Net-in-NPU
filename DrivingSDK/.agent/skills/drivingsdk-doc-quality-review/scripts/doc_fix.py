#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""格式化修复道（doc-quality-review skill 的「自动修复层」）。

定位：中英空格、半角↔全角标点等**可机械自动修复的排版**不进 issue，
由本脚本调用成熟格式化器批量 `--fix`，产出一个「格式化 diff/PR」。

引擎（都是 CPU、无需 GPU）：
  autocorrect（默认，保守）：只加中英/数字间空格、规范部分标点；不动括号配对，改动小、风险低。
  zhlint（--engine zhlint，彻底）：另做半角→全角、标点前空格清理、按内容统一括号宽度，但更激进。

默认 **dry-run**：只统计将改动的文件数/行数并打印 unified diff 预览，不写盘。
加 `--apply` 才真正修改文件（建议在干净 git 工作区上跑，改完用 `git diff` 复核再提 PR）。

用法：
  python3 doc_fix.py docs/zh README.md                 # 预览 autocorrect 将改什么
  python3 doc_fix.py docs/zh --engine zhlint            # 预览 zhlint
  python3 doc_fix.py docs/zh --apply                    # 真正修复（写盘）
"""

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tempfile

EXCLUDE_DIRS = {".git", "node_modules", ".venv", "venv", "site-packages"}


def collect_md(paths):
    out = []
    for p in paths:
        if os.path.isfile(p) and p.endswith((".md", ".markdown")):
            out.append(p)
        elif os.path.isdir(p):
            for dp, dn, fns in os.walk(p):
                dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
                for fn in fns:
                    if fn.endswith((".md", ".markdown")):
                        out.append(os.path.join(dp, fn))
    return sorted(set(out))


def fixed_text_autocorrect(path):
    """用 autocorrect 生成修复后的全文（不改原文件）。"""
    exe = shutil.which("autocorrect")
    if not exe:
        return None
    # autocorrect <file> 默认把修复结果打到 stdout；--fix 才写回。
    p = subprocess.run([exe, path], capture_output=True, text=True, timeout=120, check=False)
    return p.stdout if p.returncode == 0 and p.stdout else None


def fixed_text_zhlint(path):
    """用 zhlint --output 生成修复后的全文到临时文件再读回（zhlint 需相对路径）。"""
    exe = shutil.which("zhlint")
    if not exe:
        return None
    d = os.path.dirname(os.path.abspath(path)) or "."
    rel = os.path.basename(path)
    with tempfile.NamedTemporaryFile("r", suffix=".md", delete=False) as tf:
        outp = tf.name
    try:
        p = subprocess.run(
            [exe, rel, "--output", outp], cwd=d, capture_output=True, text=True, timeout=120, check=False
        )
        # zhlint 崩溃/超时被 kill 时 --output 可能是空或残缺文件，
        # 若此时读回并在 --apply 下写盘会覆盖原文导致数据丢失，故返回码非零一律放弃。
        if p.returncode != 0:
            return None
        with open(outp, encoding="utf-8") as f:
            text = f.read()
        return text if text else None
    finally:
        try:
            os.unlink(outp)
        except OSError:
            pass


ENGINES = {
    "autocorrect": dict(
        bin="autocorrect", fn=fixed_text_autocorrect, hint="下载 autocorrect 预编译二进制或 cargo install autocorrect"
    ),
    "zhlint": dict(bin="zhlint", fn=fixed_text_zhlint, hint="npm i -g zhlint"),
}


def main():
    ap = argparse.ArgumentParser(description="文档格式化修复（autocorrect/zhlint --fix）")
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--engine", choices=list(ENGINES), default="autocorrect")
    ap.add_argument("--apply", action="store_true", help="真正写盘（默认仅预览 diff）")
    ap.add_argument("--max-preview", type=int, default=8, help="预览前 N 个文件的 diff")
    args = ap.parse_args()

    spec = ENGINES[args.engine]
    if not shutil.which(spec["bin"]):
        sys.stderr.write("未找到格式化器 `%s`。安装：%s\n" % (spec["bin"], spec["hint"]))
        return 2

    files = collect_md(args.paths)
    sys.stderr.write(
        "[fix] 引擎=%s，候选 %d 个 markdown，模式=%s\n"
        % (args.engine, len(files), "APPLY" if args.apply else "dry-run")
    )

    changed, total_difflines, shown = [], 0, 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                old = f.read()
        except OSError:
            continue
        new = spec["fn"](path)
        if new is None or new == old:
            continue
        changed.append(path)
        diff = list(
            difflib.unified_diff(
                old.splitlines(), new.splitlines(), fromfile=path, tofile=path + " (fixed)", lineterm=""
            )
        )
        total_difflines += sum(1 for d in diff if d and d[0] in "+-" and d[:2] not in ("--", "++"))
        if shown < args.max_preview:
            shown += 1
            sys.stderr.write("\n----- %s -----\n" % path)
            for d in diff[:40]:
                sys.stderr.write(d + "\n")
        if args.apply:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)

    sys.stderr.write("\n[fix] 将改动 %d/%d 个文件，约 %d 行有变化。\n" % (len(changed), len(files), total_difflines))
    if args.apply:
        sys.stderr.write("[fix] 已写盘。请 `git diff` 复核后作为独立 docs PR 提交。\n")
    else:
        sys.stderr.write("[fix] 这是预览（未写盘）。确认无误后加 --apply 执行。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
