#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把文档质量发现整理成 GitCode issue（doc-quality-review skill 的一部分）。

双模式（对应任务确认的「本地草稿 + 可选 API」）：

  draft   把 scan_docs.py 的 JSON 结果渲染成「可直接粘贴的 issue 草稿(md)」，
          按文件聚合成表格。零依赖、不联网、必定成功，是默认/兜底路径。

  create  用 GitCode OpenAPI 创建 issue。需要环境变量 GITCODE_TOKEN。
          端点默认 POST {api-base}/repos/{owner}/{repo}/issues（Bearer 认证）；
          GitCode/Gitee 另有 /repos/{owner}/issues(repo 放 body) 变体，可用
          --path-style owner 切换。务必先用 --dry-run 打印请求确认无误再真发。

安全：token 只从环境变量读取，绝不打印、不写盘、不进 issue 正文。

示例：
  # 1) 先扫描出 JSON
  python3 scan_docs.py . --json /tmp/scan.json
  # 2) 生成本地 issue 草稿
  python3 gitcode_issue.py draft /tmp/scan.json -o /tmp/issue_draft.md
  # 3)（可选）确认无误后用 API 创建
  export GITCODE_TOKEN=xxxx
  python3 gitcode_issue.py create --owner Ascend --repo DrivingSDK \
      --title "[docs] 文档质量扫描候选" --body-file /tmp/issue_draft.md --dry-run
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_API_BASE = "https://api.gitcode.com/api/v5"
RULE_LABEL = {
    "link-broken": "失效链接",
    "fence-unbalanced": "代码块未闭合",
    "heading-no-space": "标题格式",
    "dup-punct": "标点",
    "mixed-bracket": "括号混用",
    "repeat-word": "重复词",
    "placeholder": "占位残留",
    "trailing-space": "行尾空格",
}
SEV_ORDER = {"error": 0, "warning": 1, "info": 2}


def load_findings(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("findings", []), data.get("files_scanned")
    return data, None


ISSUE_LANES = ("structure", "terminology", "grammar")


def render_draft(findings, files_scanned, title):
    """渲染 issue 草稿。

    只把 issue 候选（structure/terminology/grammar）列入明细；
    可自动修复的 format 道仅折叠成一句汇总，不逐条占用 issue 篇幅。
    """
    # 兼容旧 JSON（无 lane 字段）：默认按 structure 处理。
    structure = [f for f in findings if f.get("lane", "structure") == "structure"]
    terms = [f for f in findings if f.get("lane") == "terminology"]
    grammar = [f for f in findings if f.get("lane") == "grammar"]
    issues = structure + terms + grammar
    fmtl = [f for f in findings if f.get("lane") == "format"]

    lines = []
    lines.append("## 背景\n")
    lines.append(
        "由 `doc-quality-review` skill 自动产出的文档质量 issue 候选，"
        "覆盖坏链、未闭合代码块、括号混用、标点、术语一致性等需人工判断的问题。\n"
    )
    n_err = sum(1 for f in issues if f["severity"] == "error")
    n_warn = sum(1 for f in issues if f["severity"] == "warning")
    n_info = sum(1 for f in issues if f["severity"] == "info")
    scope = "扫描 %s 个 markdown，" % files_scanned if files_scanned else ""
    lines.append("## 概览\n")
    lines.append(
        "%sissue 候选 **%d** 条：error %d / warning %d / info %d。\n" % (scope, len(issues), n_err, n_warn, n_info)
    )
    lines.append(
        "> 说明：以下为需人工判断的候选；中英空格/半角标点等可自动修复项不在此列"
        "（见末尾「格式化」一节，用 `doc_fix.py` 批量修）。语义/事实问题需复核后补充。\n"
    )

    # 结构类：逐条按文件列出（数量少、需逐个判断）
    by_file = {}
    for f in structure:
        by_file.setdefault(f["file"], []).append(f)
    lines.append("## 结构问题（逐条）\n")
    if not by_file:
        lines.append("（无）\n")
    for fpath in sorted(by_file):
        items = sorted(by_file[fpath], key=lambda x: (SEV_ORDER.get(x["severity"], 9), x["line"]))
        lines.append("### `%s`\n" % fpath)
        lines.append("| 位置 | 类型 | 级别 | 问题 | 修改建议 |")
        lines.append("|---|---|---|---|---|")
        for f in items:
            loc = "L%s" % f["line"] if f["line"] else "-"
            kind = RULE_LABEL.get(f["rule"], f["rule"])
            msg = f["message"].replace("|", "\\|").replace("\n", " ")
            sug = f["suggestion"].replace("|", "\\|").replace("\n", " ")
            lines.append("| %s | %s | %s | %s | %s |" % (loc, kind, f["severity"], msg, sug))
        lines.append("")

    # 术语类：按 message 聚合（多为同一替换，逐条列出无意义）
    if terms:
        agg = {}
        for f in terms:
            k = f["message"]
            d = agg.setdefault(k, {"n": 0, "files": set()})
            d["n"] += 1
            d["files"].add(f["file"])
        lines.append("## 术语一致性（聚合，可批量替换）\n")
        lines.append("| 建议 | 出现次数 | 涉及文件数 |")
        lines.append("|---|---|---|")
        for msg, d in sorted(agg.items(), key=lambda x: -x[1]["n"]):
            lines.append("| %s | %d | %d |" % (msg.replace("|", "\\|"), d["n"], len(d["files"])))
        lines.append("")

    # 语法/错别字（LanguageTool）
    if grammar:
        lines.append("## 语法 / 错别字\n")
        lines.append("| 文件:行 | 片段 | 问题 | 建议 |\n|---|---|---|---|")
        for f in sorted(grammar, key=lambda x: (x["file"], x["line"])):
            msg = f["message"].replace("|", "\\|").replace("\n", " ")
            sug = f["suggestion"].replace("|", "\\|").replace("\n", " ")
            lines.append(
                "| %s:%s | %s | %s | %s |"
                % (f["file"], f["line"], str(f.get("snippet", "")).replace("|", "\\|"), msg, sug)
            )
        lines.append("")

    if fmtl:
        total = sum(f.get("count", 1) for f in fmtl)
        lines.append("## 格式化（自动修复，不在本 issue 逐条列出）\n")
        lines.append(
            "另有 **%d** 处可自动修复的中英排版/标点，涉及 %d 个文件，"
            "建议单独运行 `doc_fix.py <paths> --apply` 批量修复并提格式化 PR。\n" % (total, len(fmtl))
        )

    lines.append("## 验收\n")
    lines.append(
        "- [ ] 逐条确认（区分真实问题 / 误报 / 不修）\n"
        "- [ ] 修复后 `run_review.py` 无 error 级 issue 候选残留\n"
        "- [ ] 关联 docs 类型 PR 并在合入后关闭本 issue\n"
    )
    body = "\n".join(lines)
    return title, body


def _multipart(fields):
    """dict -> (body bytes, content-type)。GitCode 写接口要 multipart/form-data，不是 JSON。"""
    boundary = "----drivingsdk" + uuid.uuid4().hex
    lines = []
    for k, v in fields.items():
        if v is None:
            continue
        lines.append("--%s" % boundary)
        lines.append('Content-Disposition: form-data; name="%s"' % k)
        lines.append("")
        lines.append(str(v))
    lines.append("--%s--" % boundary)
    lines.append("")
    body = "\r\n".join(lines).encode("utf-8")
    return body, "multipart/form-data; boundary=%s" % boundary


def http_post_multipart(url, fields, token, timeout=30):
    # GitCode 写接口要 multipart/form-data（不是 JSON），token 走 access_token 查询参数
    # （与 GitCode 官方 curl 示例一致；同时附 Bearer 头兜底）。
    sep = "&" if "?" in url else "?"
    full = "%s%saccess_token=%s" % (url, sep, urllib.parse.quote(token))
    data, ctype = _multipart(fields)
    req = urllib.request.Request(full, data=data, method="POST")
    req.add_header("Content-Type", ctype)
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", "Bearer %s" % token)
    req.add_header("User-Agent", "drivingsdk-doc-quality-review/1.0")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def cmd_draft(args):
    findings, files_scanned = load_findings(args.scan_json)
    if not findings:
        sys.stderr.write("[draft] scan JSON 中没有 findings，可能文档很干净 :)\n")
    title, body = render_draft(findings, files_scanned, args.title)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("<!-- 建议 issue 标题：%s -->\n\n" % title)
            f.write(body)
        sys.stderr.write("[draft] issue 草稿已写入 %s\n" % args.out)
    else:
        print("# 建议标题：%s\n" % title)
        print(body)
    return 0


def cmd_create(args):
    token = os.environ.get("GITCODE_TOKEN", "").strip()
    if args.body_file:
        with open(args.body_file, "r", encoding="utf-8") as f:
            body = f.read()
    elif args.body:
        body = args.body
    else:
        sys.stderr.write("[create] 需要 --body 或 --body-file 提供 issue 正文\n")
        return 2

    if args.path_style == "owner-repo":
        url = "%s/repos/%s/%s/issues" % (args.api_base.rstrip("/"), args.owner, args.repo)
        fields = {"title": args.title, "body": body}
    else:  # gitee 风格：repo 放在 body 里
        url = "%s/repos/%s/issues" % (args.api_base.rstrip("/"), args.owner)
        fields = {"repo": args.repo, "title": args.title, "body": body}
    if args.labels:
        fields["labels"] = args.labels  # GitCode 接受逗号分隔字符串

    if args.dry_run or not token:
        if not token and not args.dry_run:
            sys.stderr.write("[create] 未设置 GITCODE_TOKEN，自动转为 dry-run（仅打印请求）。\n")
        sys.stderr.write("[create] 将发送 POST %s（multipart/form-data）\n" % url)
        # 打印字段但绝不打印 token
        preview = {k: (v if k != "body" else v[:200] + "…") for k, v in fields.items()}
        sys.stderr.write("[create] fields=%s\n" % json.dumps(preview, ensure_ascii=False)[:2000])
        sys.stderr.write("[create] 认证：access_token 查询参数 + Bearer 头（均已隐藏）\n")
        return 0

    try:
        status, text = http_post_multipart(url, fields, token)
        sys.stderr.write("[create] HTTP %s\n" % status)
        try:
            obj = json.loads(text)
            num = obj.get("number") or obj.get("iid") or obj.get("id")
            html = obj.get("html_url") or obj.get("url") or ""
            sys.stderr.write("[create] 创建成功 issue=%s %s\n" % (num, html))
        except ValueError:
            sys.stderr.write("[create] 响应：%s\n" % text[:500])
        return 0
    except urllib.error.HTTPError as e:
        sys.stderr.write("[create] HTTP 错误 %s：%s\n" % (e.code, e.read().decode("utf-8", "replace")[:500]))
        sys.stderr.write("[create] 若 404/405，多半是端点风格不符，尝试切换 --path-style。\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write("[create] 网络错误：%s\n" % e.reason)
        return 1


def main():
    ap = argparse.ArgumentParser(description="GitCode 文档质量 issue 工具（双模式）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("draft", help="scan JSON -> 本地 issue 草稿(md)")
    d.add_argument("scan_json", help="scan_docs.py 产出的 JSON")
    d.add_argument("-o", "--out", help="输出 md 路径；缺省打印到 stdout")
    d.add_argument("--title", default="[docs] 文档质量扫描候选清单", help="建议 issue 标题")
    d.set_defaults(func=cmd_draft)

    c = sub.add_parser("create", help="用 GitCode API 创建 issue（需 GITCODE_TOKEN）")
    c.add_argument("--owner", default="Ascend")
    c.add_argument("--repo", default="DrivingSDK")
    c.add_argument("--title", required=True)
    c.add_argument("--body")
    c.add_argument("--body-file")
    c.add_argument("--labels", help="逗号分隔的标签")
    c.add_argument("--api-base", default=DEFAULT_API_BASE)
    c.add_argument(
        "--path-style",
        choices=["owner-repo", "owner"],
        default="owner-repo",
        help="owner-repo: /repos/{owner}/{repo}/issues；owner: /repos/{owner}/issues(repo 入 body)",
    )
    c.add_argument("--dry-run", action="store_true", help="只打印请求，不真正发送")
    c.set_defaults(func=cmd_create)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
