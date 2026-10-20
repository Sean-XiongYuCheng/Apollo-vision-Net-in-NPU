#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语法 / 错别字检测道（doc-quality-review skill）。封装开源 LanguageTool。

覆盖：英文语法+拼写（如 a/an、sentance→sentence）、中文部分错别字与「的地得」
（如 帐号→账号）。**不依赖 GPU**，但需要 Java + LanguageTool（较重，故为可选道）。

实测取舍（见 SKILL.md「已评估并否决的工具」）：LanguageTool 在中文技术文档上
低召回且带误报，因此本封装做了三层降噪：
  1) 先剥掉代码块/行内代码/链接 URL，只把正文喂给 LT（代码里全是误报源）；
  2) 只保留**带修改建议**的命中（LT 的 wa5 等风格规则常无建议、纯噪声，直接丢）；
  3) 维护一个**禁用规则名单**（如 zh 的 BU「步→不」这类高频误报），可在 --disable 扩充。
英文错别字另有仓库已用的 codespell（更轻），本道侧重语法 + 中文错别字。

依赖（任一即可，按此优先级探测）：
  - 环境变量 LT_JAR 指向 languagetool-commandline.jar，且 java 在 PATH（或 JAVA_HOME）；
  - 或 PATH 上有 `languagetool`（官方 CLI 包装）。
缺依赖时本道整体跳过，不影响其它道。

用法：
  LT_JAR=/path/languagetool-commandline.jar python3 grammar_check.py docs/zh --format json
  python3 grammar_check.py docs/zh/introduction.md            # 文本输出
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
RE_INLINE = re.compile(r"`[^`]*`")
RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # 保留显示文字，去掉 URL

# 默认禁用的高频误报规则（实测）。可用 --disable 追加。
DEFAULT_DISABLE = [
    "BU",  # zh：把「步」误判为「不」
    "wb4",  # zh：的/地/得，技术文档里大量误报（如「强大的并行算力」被改「地」）
    "s32",  # zh：选词风格（如「增加→扩大」），非错误、噪声
    "wb2",  # zh：高频误报的词法风格规则
    "WHITESPACE_RULE",
    "COMMA_PARENTHESIS_WHITESPACE",
]


def strip_markdown(text):
    """把代码块/行内代码/链接 URL 替换为等长空白，保留行号与偏移。"""
    out_lines = []
    fence = False
    for raw in text.splitlines():
        s = raw.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            out_lines.append(" " * len(raw))
            continue
        if fence:
            out_lines.append(" " * len(raw))
            continue
        line = RE_INLINE.sub(lambda m: " " * len(m.group(0)), raw)
        line = RE_LINK.sub(lambda m: m.group(1).ljust(len(m.group(0))), line)
        out_lines.append(line)
    return "\n".join(out_lines)


def detect_lang(text):
    return "zh-CN" if CJK_RE.search(text) else "en-US"


def offset_to_line(text, offset):
    return text.count("\n", 0, offset) + 1


def find_lt():
    """返回 ('jar', jar路径, java) 或 ('cli', languagetool 绝对路径, None) 或 None。"""
    cli = shutil.which("languagetool")
    if cli:
        return ("cli", cli, None)
    jar = os.environ.get("LT_JAR")
    java = shutil.which("java")
    if not java and os.environ.get("JAVA_HOME"):
        cand = os.path.join(os.environ["JAVA_HOME"], "bin", "java")
        java = cand if os.path.exists(cand) else None
    if jar and os.path.exists(jar) and java:
        return ("jar", jar, java)
    return None


def run_lt(text, lang, lt):
    """对已剥码文本跑 LanguageTool，返回 matches(list)。"""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        path = tf.name
    try:
        if lt[0] == "jar":
            cmd = [lt[2], "-jar", lt[1], "-l", lang, "--json", path]
        else:
            cmd = [lt[1], "-l", lang, "--json", path]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        return json.loads(p.stdout or "{}").get("matches", [])
    except Exception as e:
        sys.stderr.write("[grammar] LT 调用失败：%s\n" % e)
        return []
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def check_file(path, root, lt, disable):
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return []
    stripped = strip_markdown(raw)
    if not stripped.strip():
        return []
    lang = detect_lang(stripped)
    findings = []
    for m in run_lt(stripped, lang, lt):
        rid = m.get("rule", {}).get("id", "")
        reps = [r.get("value") for r in m.get("replacements", [])]
        # 降噪：丢无修改建议的命中 + 禁用名单
        if not reps or rid in disable:
            continue
        ctx = m.get("context", {})
        frag = ctx.get("text", "")[ctx.get("offset", 0) : ctx.get("offset", 0) + ctx.get("length", 0)]
        cat = m.get("rule", {}).get("category", {}).get("id", "")
        sev = "warning" if cat in ("TYPOS", "CAT1") else "info"
        findings.append(
            dict(
                file=os.path.relpath(path, root),
                line=offset_to_line(stripped, m.get("offset", 0)),
                rule="lt-%s" % rid,
                severity=sev,
                engine="languagetool",
                lane="grammar",
                snippet=frag[:80],
                message="%s（疑似%s）"
                % (m.get("message", ""), "拼写/错别字" if cat in ("TYPOS", "CAT1") else "语法/风格"),
                suggestion="建议改为：%s" % "、".join(reps[:3]),
            )
        )
    return findings


def iter_md(paths):
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            for dp, dn, fns in os.walk(p):
                dn[:] = [d for d in dn if d not in {".git", "node_modules", ".venv", "venv"}]
                out += [os.path.join(dp, fn) for fn in fns if fn.endswith((".md", ".markdown"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="语法/错别字检测（LanguageTool 封装）")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--root", default=".")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--disable", default="", help="额外禁用的 LT 规则 id，逗号分隔")
    args = ap.parse_args()

    lt = find_lt()
    if not lt:
        sys.stderr.write(
            "未找到 LanguageTool。安装其一：\n"
            "  1) 下载 LanguageTool 并设 LT_JAR=/path/languagetool-commandline.jar（需 java/JAVA_HOME）；\n"
            "  2) 或安装官方 `languagetool` CLI 到 PATH。\n"
            "（英文错别字也可用更轻的 codespell；本道侧重语法+中文错别字）\n"
        )
        return 2

    disable = set(DEFAULT_DISABLE) | {s for s in args.disable.split(",") if s}
    root = os.path.abspath(args.root)
    files = iter_md(args.paths)
    sys.stderr.write("[grammar] LanguageTool=%s，待检 %d 个文件\n" % (lt[0], len(files)))

    findings = []
    for i, f in enumerate(files, 1):
        findings.extend(check_file(f, root, lt, disable))
        if i % 20 == 0:
            sys.stderr.write("[grammar] 进度 %d/%d，累计命中 %d\n" % (i, len(files), len(findings)))

    if args.format == "json":
        print(json.dumps({"findings": findings}, ensure_ascii=False, indent=2))
    else:
        for f in findings:
            print(
                "[%s] %s:%s %s\n    %s -> %s"
                % (f["rule"], f["file"], f["line"], f["snippet"], f["message"], f["suggestion"])
            )
    sys.stderr.write("[grammar] 完成，命中 %d 条。\n" % len(findings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
