#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DrivingSDK 文档机械扫描器（doc-quality-review skill 的一部分）。

只用标准库，无第三方依赖。负责发现「机械可判定」的文档问题作为候选清单，
为 SKILL.md 里描述的语义/语法人工复核提供高召回的输入。

机械规则（高可判定，作为候选，不等于最终 issue）：
  - link-broken   相对链接 / 图片指向的文件不存在
  - fence-unbalanced  代码围栏 ``` 数量为奇数（疑似未闭合）
  - heading-no-space  标题 # 后缺空格
  - dup-punct     相邻重复 / 中英混排标点（。。 ，， 。. 等）
  - mixed-bracket 中英文括号不配对（（…) 或 (…））
  - repeat-word   叠词 / 重复单词（的的 了了 / the the）
  - placeholder   残留占位符（TODO/FIXME/待补充/TBD）
  - typo-cn       中文常见错别字（混淆集 assets/cn_confusion.txt）

prose 类规则（dup-punct / repeat-word / placeholder / mixed-bracket / typo-cn）会跳过
代码围栏与行内代码，降低误报。link/fence/heading 在原始行上判断。

用法：
  python3 scan_docs.py [ROOT] [--json out.json] [--format json|text]
                       [--include GLOB ...] [--exclude GLOB ...] [--rules r1,r2]

退出码：发现 error 级问题返回 1，否则 0（便于 CI / pre-commit 接入）。
"""

import argparse
import fnmatch
import json
import os
import re
import sys

# ---- 默认扫描 / 忽略范围 ----------------------------------------------------
DEFAULT_EXCLUDES = [
    "*/.git/*",
    "*/node_modules/*",
    "*/.venv/*",
    "*/venv/*",
    "*/site-packages/*",
    "*/.agent/skills/*",  # 跳过 skill 自身的示例文档
    "*/Third-Party*",
]

SEVERITY = {
    "link-broken": "error",
    "fence-unbalanced": "error",
    "heading-no-space": "warning",
    "dup-punct": "warning",
    "typo-cn": "warning",
    "mixed-bracket": "warning",
    "repeat-word": "warning",
    "placeholder": "info",
}

# ---- 正则 -------------------------------------------------------------------
CJK = r"一-鿿㐀-䶿"
RE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
RE_HEADING_BAD = re.compile(r"^(#{1,6})[^#\s]")
RE_DUP_PUNCT = re.compile(r"([。，、；：！？])\1|[。，；：！？][.,;:!?]|[.,;:!?][。，；：！？]")
# 注：中英空格、半角→全角标点等「可自动修复的排版」不在 builtin 内，交给格式化器
# （doc_fix.py 调 autocorrect/zhlint --fix）。builtin 只管「需人判断、不宜自动改」的问题。
# 中文叠词检测刻意不做：单字“的/了/与”相邻多为「目的/的」「参与/与」这类合法跨词，
# 误报率极高，交由人工语义复核。只保留高置信度的英文重复词。
RE_REPEAT_EN = re.compile(r"\b([A-Za-z]{2,})\s+\1\b", re.IGNORECASE)
# 占位符：只认英文 TODO 类标记和明确的中文待办词；不认“占位”——本仓“占位符(token)”是正经术语。
RE_PLACEHOLDER = re.compile(r"\b(TODO|FIXME|TBD)\b|待补充|待完善|此处待")
RE_INLINE_CODE = re.compile(r"`[^`]*`")


def load_confusion():
    """加载中文错别字混淆集 assets/cn_confusion.txt -> [(wrong, right), ...]。"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "cn_confusion.txt")
    pairs = []
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.split("#", 1)[0].strip()
                if not ln or "\t" not in ln:
                    continue
                w, r = [x.strip() for x in ln.split("\t")[:2]]
                if w and r:
                    pairs.append((w, r))
    except OSError:
        pass
    return pairs


def load_domain_terms():
    """加载领域保护词表 assets/domain_terms.txt -> set。

    typo-cn 命中若落在某个保护词内（错串是保护词的子串且该保护词出现在同一行），则跳过，防止误伤。
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "domain_terms.txt")
    terms = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.split("#", 1)[0].strip()
                if ln:
                    terms.add(ln)
    except OSError:
        pass
    return terms


CONFUSION = load_confusion()
DOMAIN_TERMS = load_domain_terms()


def _protected(wrong, prose):
    """错串 wrong 在 prose 中的出现是否被某个领域保护词覆盖（是则应跳过该 typo-cn 命中）。"""
    for t in DOMAIN_TERMS:
        if wrong in t and t in prose:
            return True
    return False


# 混用括号：只认“中文括号被英文括号闭合 / 英文括号被中文括号闭合”这类明确混用，
# 不认单独的「1）」中文序号或跨行未闭合（误报多）。
RE_MIXED_BRACKET = re.compile(r"（[^（）()]*\)|\([^（）()]*）")


def strip_inline_code(text):
    return RE_INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)


def is_external_or_anchor(target):
    t = target.strip()
    if not t:
        return True
    if t.startswith("#"):
        return True
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):  # http(s)://, ftp:// ...
        return True
    if t.startswith("mailto:") or t.startswith("tel:"):
        return True
    return False


def check_links(path, lineno, raw, findings, repo_root):
    for m in RE_LINK.finditer(raw):
        target = m.group(1).strip()
        # 去掉可能的标题部分： (path "title")
        target = re.split(r"\s+", target, 1)[0]
        if is_external_or_anchor(target):
            continue
        # 去掉锚点
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue
        # 去掉 query
        file_part = file_part.split("?", 1)[0]
        base = os.path.dirname(path)
        resolved = os.path.normpath(os.path.join(base, file_part))
        if not os.path.exists(resolved):
            findings.append(
                dict(
                    file=os.path.relpath(path, repo_root),
                    line=lineno,
                    rule="link-broken",
                    severity=SEVERITY["link-broken"],
                    snippet=m.group(0),
                    message="链接/图片目标不存在：%s" % file_part,
                    suggestion="修正相对路径或补齐被引用文件；确认大小写与实际文件名一致。",
                )
            )


def check_mixed_bracket(text):
    """只检测明确的中英文括号混用：（…) 或 (…）。返回匹配对象或 None。"""
    return RE_MIXED_BRACKET.search(text)


def scan_file(path, repo_root, rules):
    findings = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (UnicodeDecodeError, OSError):
        return findings

    fence = False
    fence_count = 0
    for i, raw in enumerate(lines, 1):
        line = raw.rstrip("\n")
        stripped = line.lstrip()

        # 代码围栏状态机
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence_count += 1
            fence = not fence
            continue

        # ---- 原始行规则（含代码块内也判断链接/标题意义不大，跳过代码块）----
        if not fence:
            if "heading-no-space" in rules and RE_HEADING_BAD.match(line):
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="heading-no-space",
                        severity=SEVERITY["heading-no-space"],
                        snippet=line[:60],
                        message="标题 # 与文字间缺少空格。",
                        suggestion="在 # 号与标题文字之间补一个空格，如 `## 标题`。",
                    )
                )
            if "link-broken" in rules:
                check_links(path, i, line, findings, repo_root)

        # ---- prose 规则：跳过代码块与行内代码 ----
        if fence:
            continue
        prose = strip_inline_code(line)

        if "dup-punct" in rules:
            m = RE_DUP_PUNCT.search(prose)
            if m:
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="dup-punct",
                        severity=SEVERITY["dup-punct"],
                        snippet=line.strip()[:80],
                        message="相邻重复或中英混排标点：%r" % m.group(0),
                        suggestion="统一为单个中文标点；正文中文语境下避免半角标点。",
                    )
                )

        if "typo-cn" in rules:
            for wrong, right in CONFUSION:
                if wrong not in prose:
                    continue
                if _protected(wrong, prose):  # 命中落在领域保护词内 → 跳过，防误伤
                    continue
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="typo-cn",
                        severity=SEVERITY["typo-cn"],
                        snippet=line.strip()[:80],
                        message="疑似错别字：%s" % wrong,
                        suggestion="应为「%s」。" % right,
                    )
                )

        if "repeat-word" in rules:
            m = RE_REPEAT_EN.search(prose)
            if m:
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="repeat-word",
                        severity=SEVERITY["repeat-word"],
                        snippet=line.strip()[:80],
                        message="疑似叠词/重复词：%r" % m.group(0),
                        suggestion="确认是否多打一个字/词，删除冗余。",
                    )
                )

        if "placeholder" in rules:
            m = RE_PLACEHOLDER.search(prose)
            if m:
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="placeholder",
                        severity=SEVERITY["placeholder"],
                        snippet=line.strip()[:80],
                        message="残留占位符/待办标记：%r" % m.group(0),
                        suggestion="补齐正式内容或在发布前移除占位标记。",
                    )
                )

        if "mixed-bracket" in rules:
            mb = check_mixed_bracket(prose)
            if mb:
                findings.append(
                    dict(
                        file=os.path.relpath(path, repo_root),
                        line=i,
                        rule="mixed-bracket",
                        severity=SEVERITY["mixed-bracket"],
                        snippet=line.strip()[:80],
                        message="中英文括号混用：%r" % mb.group(0),
                        suggestion="中文语境统一用（），英文/代码语境统一用 ()，避免一对括号中英混搭。",
                    )
                )

    if "fence-unbalanced" in rules and fence_count % 2 != 0:
        findings.append(
            dict(
                file=os.path.relpath(path, repo_root),
                line=0,
                rule="fence-unbalanced",
                severity=SEVERITY["fence-unbalanced"],
                snippet="```",
                message="代码围栏 ``` 数量为奇数(%d)，疑似未闭合。" % fence_count,
                suggestion="检查所有代码块是否成对闭合。",
            )
        )
    return findings


def iter_md_files(root, includes, excludes):
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if not (name.endswith(".md") or name.endswith(".markdown")):
                continue
            full = os.path.join(dirpath, name)
            rel = "/" + os.path.relpath(full, root)
            if any(fnmatch.fnmatch(full, p) or fnmatch.fnmatch(rel, p) for p in excludes):
                continue
            if includes and not any(fnmatch.fnmatch(full, p) or fnmatch.fnmatch(rel, p) for p in includes):
                continue
            yield full


def emit_text(findings):
    by_sev = {"error": [], "warning": [], "info": []}
    for f in findings:
        by_sev.setdefault(f["severity"], []).append(f)
    for sev in ("error", "warning", "info"):
        items = by_sev.get(sev, [])
        if not items:
            continue
        print("\n==== %s (%d) ====" % (sev.upper(), len(items)))
        for f in items:
            loc = "%s:%s" % (f["file"], f["line"])
            print(
                "[%s] %s\n    %s\n    snippet: %s\n    fix: %s"
                % (f["rule"], loc, f["message"], f["snippet"], f["suggestion"])
            )


def main():
    ap = argparse.ArgumentParser(description="DrivingSDK 文档结构扫描器（structure 道）")
    ap.add_argument("root", nargs="?", default=".", help="仓库根目录")
    ap.add_argument("--json", help="将结果写入 JSON 文件")
    ap.add_argument("--format", choices=["json", "text"], default="text")
    ap.add_argument("--include", action="append", default=[], help="只扫描匹配的 glob")
    ap.add_argument("--exclude", action="append", default=[], help="额外排除 glob")
    ap.add_argument("--rules", help="逗号分隔的规则白名单，默认全部")
    ap.add_argument("--no-default-excludes", action="store_true", help="不套用内置排除（如 .agent/skills/*）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    excludes = ([] if args.no_default_excludes else DEFAULT_EXCLUDES) + args.exclude
    all_rules = set(SEVERITY)
    rules = set(args.rules.split(",")) if args.rules else all_rules
    rules &= all_rules

    findings = []
    nfiles = 0
    for path in iter_md_files(root, args.include, excludes):
        nfiles += 1
        findings.extend(scan_file(path, root, rules))

    findings.sort(key=lambda f: (f["file"], f["line"]))
    sys.stderr.write("[scan] 扫描 %d 个 markdown，命中 %d 条候选\n" % (nfiles, len(findings)))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump({"root": root, "files_scanned": nfiles, "findings": findings}, out, ensure_ascii=False, indent=2)
        sys.stderr.write("[scan] JSON 已写入 %s\n" % args.json)

    if args.format == "json":
        print(json.dumps({"files_scanned": nfiles, "findings": findings}, ensure_ascii=False, indent=2))
    else:
        emit_text(findings)

    has_error = any(f["severity"] == "error" for f in findings)
    sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
