#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多道文档检查编排器（doc-quality-review skill）。CPU、无需 GPU。

按「问题怎么解决」分道，而不是把所有命中平铺成一张表：

  structure   内置正则 scan_docs.py（恒开）——坏链、未闭合代码块、标题缺空格、
              括号混用、重复标点、英文重复词、占位残留。需人判断、不宜自动改，进 issue。
  terminology Vale（装了且有 .vale.ini 才跑）——术语一致性/禁用词，进 issue。
  format      autocorrect --lint（装了才跑）——中英空格、半角↔全角等**可自动修复**排版。
              **不逐条进 issue**，只按文件聚合成一句汇总，提示去跑 doc_fix.py 批量修。

输出区分「issue 候选」（structure + terminology）与「格式化汇总」（format）。

用法：
  python3 run_review.py . --json /tmp/r.json --format md
  python3 run_review.py --list-engines
  python3 run_review.py . --vale-config <path/.vale.ini>     # 显式指定 Vale 配置
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# markdown 收集逻辑与默认排除集只在 scan_docs 里维护一份，这里复用，避免两处实现走偏。
from scan_docs import DEFAULT_EXCLUDES  # noqa: E402
from scan_docs import iter_md_files as scan_iter_md_files  # noqa: E402

SKILL_DIR = os.path.dirname(HERE)
DEFAULT_VALE_CFG = os.path.join(SKILL_DIR, "assets", "vale", ".vale.ini")
CJK_RE = re.compile(r"[㐀-鿿一-鿿]")
ISSUE_LANES = ("structure", "terminology", "grammar")


# ---------------- 通用 ----------------
def iter_md_files(root, includes, excludes, no_default_excludes=False):
    default_ex = [] if no_default_excludes else list(DEFAULT_EXCLUDES)
    return scan_iter_md_files(root, includes, default_ex + excludes)


def file_lang(path):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return "en"
    cjk = len(CJK_RE.findall(text))
    letters = sum(c.isascii() and c.isalpha() for c in text)
    if cjk == 0:
        return "en"
    return "zh" if cjk > letters * 0.3 else "mixed"


# ---------------- structure 道：内置正则 ----------------
def run_structure(root, files, args):
    cmd = [sys.executable, os.path.join(HERE, "scan_docs.py"), root, "--format", "json"]
    for inc in args.include:
        cmd += ["--include", inc]
    for exc in args.exclude:
        cmd += ["--exclude", exc]
    if args.no_default_excludes:
        cmd += ["--no-default-excludes"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        out = json.loads(p.stdout or "{}").get("findings", [])
        for f in out:
            f["engine"] = "builtin"
            f["lane"] = "structure"
        return out
    except Exception as e:
        sys.stderr.write("[structure] 失败：%s\n" % e)
        return []


# ---------------- format 道：autocorrect --lint（聚合） ----------------
def run_format(root, files, args):
    exe = shutil.which("autocorrect")
    if not exe:
        return []
    findings = []
    for path in files:
        try:
            p = subprocess.run(
                [exe, "--lint", "--format", "json", path], capture_output=True, text=True, timeout=120, check=False
            )
            data = json.loads(p.stdout or "{}")
        except Exception:
            continue
        n = sum(len(m.get("lines", [])) for m in data.get("messages", []))
        if n:
            findings.append(
                dict(
                    file=os.path.relpath(path, root),
                    line=0,
                    rule="format-autofix",
                    severity="info",
                    engine="autocorrect",
                    lane="format",
                    count=n,
                    snippet="",
                    message="%d 处可自动修复的中英排版/标点" % n,
                    suggestion="运行 `doc_fix.py %s` 预览，--apply 批量修复（不逐条进 issue）"
                    % os.path.relpath(path, root),
                )
            )
    return findings


# ---------------- terminology 道：Vale ----------------
def run_terminology(root, files, args):
    cfg = DEFAULT_VALE_CFG if os.path.exists(DEFAULT_VALE_CFG) else None
    exe = shutil.which("vale")
    if not exe or not cfg:
        return []
    findings = []
    # 分批调用，避免命令行过长
    for i in range(0, len(files), 200):
        batch = files[i : i + 200]
        try:
            p = subprocess.run(
                [exe, "--config", cfg, "--output", "JSON", *batch],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            data = json.loads(p.stdout or "{}")
        except Exception:
            continue
        for fpath, alerts in data.items():
            rel = os.path.relpath(fpath, root) if os.path.isabs(fpath) else fpath
            for a in alerts:
                sev = {"error": "error", "warning": "warning"}.get(a.get("Severity"), "info")
                findings.append(
                    dict(
                        file=rel,
                        line=a.get("Line", 0),
                        rule="vale-%s" % a.get("Check", "term"),
                        severity=sev,
                        engine="vale",
                        lane="terminology",
                        snippet=a.get("Match", "")[:80],
                        message=a.get("Message", ""),
                        suggestion="按术语/风格规则修正。",
                    )
                )
    return findings


def run_grammar(root, files, args):
    """grammar 道：调 grammar_check.py(LanguageTool) 查语法/错别字。"""
    if not files:
        return []
    cmd = [sys.executable, os.path.join(HERE, "grammar_check.py"), "--root", root, "--format", "json", *files]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        out = json.loads(p.stdout or "{}").get("findings", [])
        for f in out:
            f["lane"] = "grammar"
        return out
    except Exception as e:
        sys.stderr.write("[grammar] 失败：%s\n" % e)
        return []


def detect_autocorrect():
    return shutil.which("autocorrect") is not None


def detect_grammar():
    if shutil.which("languagetool"):
        return True
    jar = os.environ.get("LT_JAR")
    java = shutil.which("java") or (
        os.path.join(os.environ["JAVA_HOME"], "bin", "java") if os.environ.get("JAVA_HOME") else None
    )
    return bool(jar and os.path.exists(jar) and java and os.path.exists(java))


def detect_vale():
    cfg = os.path.exists(DEFAULT_VALE_CFG)
    return shutil.which("vale") is not None and cfg


# 引擎集合经实测裁剪（详见 SKILL.md「已评估并否决的工具」）。
# hint = 缺失时给用户的可操作安装提示。
ENGINES = {
    "structure": dict(run=run_structure, detect=lambda: True, lane="structure", hint="内置，无需安装"),
    "format": dict(
        run=run_format,
        detect=detect_autocorrect,
        lane="format",
        hint="装 autocorrect：从 https://github.com/huacnlee/autocorrect/releases 下预编译二进制放进 PATH",
    ),
    "terminology": dict(
        run=run_terminology,
        detect=detect_vale,
        lane="terminology",
        hint="装 vale：从 https://github.com/errata-ai/vale/releases 下二进制放进 PATH（配置已内置）",
    ),
    "grammar": dict(
        run=run_grammar,
        detect=detect_grammar,
        lane="grammar",
        hint="装 LanguageTool：解压 LanguageTool 后 export LT_JAR=.../languagetool-commandline.jar，并装 Java(JRE)",
    ),
}


def merge_dedup(findings):
    seen, out = set(), []
    for f in findings:
        # 含 message：同一行、同一规则可能命中多个不同问题（如一行两个错别字），
        # 它们 snippet（整行文本）相同但 message 不同，不应被当作重复丢弃。
        key = (f.get("file"), f.get("line"), f.get("rule"), f.get("snippet"), f.get("message"))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _agg_terms(terms):
    """术语命中按 message 聚合：message -> (次数, 文件数)。"""
    agg = {}
    for f in terms:
        d = agg.setdefault(f["message"], {"n": 0, "files": set()})
        d["n"] += 1
        d["files"].add(f.get("file"))
    return sorted(((m, d["n"], len(d["files"])) for m, d in agg.items()), key=lambda x: -x[1])


def emit(findings, fmt, lang_stats, used, skipped):
    structure = [f for f in findings if f.get("lane") == "structure"]
    terms = [f for f in findings if f.get("lane") == "terminology"]
    grammar = [f for f in findings if f.get("lane") == "grammar"]
    fmtl = [f for f in findings if f.get("lane") == "format"]
    order = {"error": 0, "warning": 1, "info": 2}
    structure.sort(key=lambda f: (order.get(f["severity"], 9), f.get("file", ""), f.get("line", 0)))
    grammar.sort(key=lambda f: (f.get("file", ""), f.get("line", 0)))
    term_agg = _agg_terms(terms)
    total_fmt = sum(f.get("count", 1) for f in fmtl)

    if fmt == "json":
        print(
            json.dumps(
                dict(
                    findings=findings,
                    issues=[f for f in findings if f.get("lane") in ISSUE_LANES],
                    format_summary=fmtl,
                    engines_used=used,
                    lang_stats=lang_stats,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if fmt == "md":
        print("# 文档质量检查报告\n")
        print("**启用道**：%s" % ", ".join(used))
        if skipped:
            print("\n**未启用（可选增强）**：")
            for n, h in skipped:
                print("- `%s` — %s" % (n, h))
        print(
            "\n**语言分布**：zh %d / en %d / mixed %d\n"
            % (lang_stats.get("zh", 0), lang_stats.get("en", 0), lang_stats.get("mixed", 0))
        )
        print("## 结构问题（逐条，共 %d 条）\n" % len(structure))
        if structure:
            print("| 文件:行 | 规则 | 级别 | 问题 | 建议 |\n|---|---|---|---|---|")
            for f in structure:
                loc = "%s:%s" % (f.get("file", ""), f.get("line", 0))
                msg = str(f.get("message", "")).replace("|", "\\|").replace("\n", " ")
                sug = str(f.get("suggestion", "")).replace("|", "\\|").replace("\n", " ")
                print("| %s | %s | %s | %s | %s |" % (loc, f.get("rule"), f["severity"], msg, sug))
        else:
            print("（无）")
        print("\n## 术语一致性（聚合，共 %d 处 / %d 文件）\n" % (len(terms), len({f.get("file") for f in terms})))
        if term_agg:
            print("| 建议 | 次数 | 文件数 |\n|---|---|---|")
            for m, n, nf in term_agg:
                print("| %s | %d | %d |" % (m.replace("|", "\\|"), n, nf))
        else:
            print("（无）")
        print("\n## 语法/错别字（LanguageTool，共 %d 条）\n" % len(grammar))
        if grammar:
            print("| 文件:行 | 规则 | 片段 | 问题 | 建议 |\n|---|---|---|---|---|")
            for f in grammar:
                loc = "%s:%s" % (f.get("file", ""), f.get("line", 0))
                msg = str(f.get("message", "")).replace("|", "\\|").replace("\n", " ")
                sug = str(f.get("suggestion", "")).replace("|", "\\|").replace("\n", " ")
                print(
                    "| %s | %s | %s | %s | %s |"
                    % (loc, f.get("rule"), str(f.get("snippet", "")).replace("|", "\\|"), msg, sug)
                )
        else:
            print("（无 / 未启用）")
        print("\n## 格式化汇总（自动修复，不进 issue）\n")
        print(
            "共 **%d** 处可自动修复排版，涉及 %d 个文件。"
            "运行 `doc_fix.py <paths> --apply` 批量修复并单独提格式化 PR。\n" % (total_fmt, len(fmtl))
        )
        for f in sorted(fmtl, key=lambda x: -x.get("count", 0))[:20]:
            print("- `%s`：%d 处" % (f["file"], f.get("count", 0)))
        return

    # text
    sys.stdout.write("启用道：%s\n" % ", ".join(used))
    if skipped:
        sys.stdout.write("未启用：%s\n" % ", ".join("%s(%s)" % (n, h) for n, h in skipped))
    sys.stdout.write("语言分布：%s\n\n== 结构问题 %d ==\n" % (lang_stats, len(structure)))
    for f in structure:
        sys.stdout.write(
            "[%s] %s:%s %s\n    %s -> %s\n"
            % (f.get("rule"), f.get("file"), f.get("line"), f["severity"], f.get("message"), f.get("suggestion"))
        )
    sys.stdout.write("\n== 术语一致性（聚合）%d 处 ==\n" % len(terms))
    for m, n, nf in term_agg:
        sys.stdout.write("  %s  ×%d，%d 文件\n" % (m, n, nf))
    sys.stdout.write("\n== 语法/错别字 %d 条 ==\n" % len(grammar))
    for f in grammar:
        sys.stdout.write(
            "  [%s] %s:%s %s -> %s\n"
            % (f.get("rule"), f.get("file"), f.get("line"), f.get("message"), f.get("suggestion"))
        )
    sys.stdout.write(
        "\n== 格式化汇总（不进 issue）== 共 %d 处，%d 文件；跑 doc_fix.py --apply 批量修\n" % (total_fmt, len(fmtl))
    )


def main():
    ap = argparse.ArgumentParser(description="多道文档检查编排器（CPU，无需 GPU）")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--format", choices=["text", "json", "md"], default="text")
    ap.add_argument("--json", help="同时写 JSON 文件（单对象，含全部 findings）")
    ap.add_argument("--jsonl", help="把每条问题逐行写入 JSONL（一行一问题，自动带 run_ts 时间戳）")
    ap.add_argument("--jsonl-append", action="store_true", help="--jsonl 改为追加模式，便于跨轮累积跟踪")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--engines", help="逗号分隔的道白名单，默认全部可用")
    ap.add_argument("--no-default-excludes", action="store_true", help="不套用内置排除（扫 skill 自带夹具时用）")
    ap.add_argument("--list-engines", action="store_true")
    args = ap.parse_args()

    if args.list_engines:
        print("道\t可用\t说明")
        for name, spec in ENGINES.items():
            print("%s\t%s\t%s" % (name, "yes" if spec["detect"]() else "no", spec["hint"]))
        return

    root = os.path.abspath(args.root)
    files = list(iter_md_files(root, args.include, args.exclude, args.no_default_excludes))
    lang_stats = {"zh": 0, "en": 0, "mixed": 0}
    for p in files:
        lang_stats[file_lang(p)] += 1

    want = set(args.engines.split(",")) if args.engines else set(ENGINES)
    used, skipped, all_f = [], [], []
    for name, spec in ENGINES.items():
        if name not in want:
            continue
        if not spec["detect"]():
            skipped.append((name, spec["hint"]))
            continue
        sys.stderr.write("[run] 道 %s ...\n" % name)
        all_f.extend(spec["run"](root, files, args) or [])
        used.append(name)

    findings = merge_dedup(all_f)
    issue_n = sum(1 for f in findings if f.get("lane") in ISSUE_LANES)
    sys.stderr.write("[run] %d 个 md，道=%s，issue 候选 %d 条\n" % (len(files), used, issue_n))

    # 缺失外部工具时给出可操作的安装提醒（agent 据此询问用户是否安装）
    if skipped:
        sys.stderr.write("\n[提醒] 以下增强道未启用（外部工具未安装），可按需安装后重跑：\n")
        for name, hint in skipped:
            sys.stderr.write("  - %s：%s\n" % (name, hint))
        sys.stderr.write("（structure 道无需任何依赖，已正常运行；是否安装上述工具由你决定）\n\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                dict(files_scanned=len(files), findings=findings, engines_used=used, lang_stats=lang_stats),
                f,
                ensure_ascii=False,
                indent=2,
            )
        sys.stderr.write("[run] JSON 写入 %s\n" % args.json)

    if args.jsonl:
        import datetime

        run_ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        mode = "a" if args.jsonl_append else "w"
        with open(args.jsonl, mode, encoding="utf-8") as f:
            for rec in findings:
                f.write(json.dumps(dict(rec, run_ts=run_ts), ensure_ascii=False) + "\n")
        sys.stderr.write(
            "[run] JSONL %s %d 条 -> %s\n" % ("追加" if args.jsonl_append else "写入", len(findings), args.jsonl)
        )

    emit(findings, args.format, lang_stats, used, skipped)
    has_err = any(f["severity"] == "error" and f.get("lane") in ISSUE_LANES for f in findings)
    sys.exit(1 if has_err else 0)


if __name__ == "__main__":
    main()
