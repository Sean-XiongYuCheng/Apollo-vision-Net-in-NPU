#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM 语义复核层（doc-quality-review skill）——默认不启用、需用户显式提供接口。

定位：规则/小工具做高召回的机械候选；语义正确性、描述与代码是否一致、
术语前后矛盾、语病这类，交给 LLM 读上下文判断。算力在 API 侧，不吃本地 GPU。

★ 重要约束（按需求显式设计）★
  - 本脚本**不内置任何模型/厂商/密钥**，也不假设环境里有可用 LLM。
  - 使用前必须由**用户确认是否启用**，并由用户提供：
      --api-base   接口地址（OpenAI 兼容 /chat/completions 风格）
      --model      模型名
      --api-key-env 存放密钥的环境变量名（脚本只从该 env 读，绝不写盘/打印）
  - 三者缺一即拒绝运行并打印所需参数；默认 --dry-run 只打印将发送的请求、不联网。
  - 文档中英混排：prompt 要求按原文语言给出中文说明的修改建议。

用法：
  # 1) 先和用户确认是否启用 LLM 复核、用哪个接口
  # 2) dry-run 看将发送什么（不联网）
  python3 llm_review.py docs/zh/introduction.md \
      --api-base https://<user-endpoint>/v1 --model <model> --api-key-env MY_LLM_KEY
  # 3) 确认后去掉 --dry-run（需先 export MY_LLM_KEY=...）
  python3 llm_review.py docs/zh/introduction.md ... --no-dry-run --format json
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# 领域保护词表只在 scan_docs 里读一份，这里复用，避免两处解析走偏。
from scan_docs import load_domain_terms as _load_domain_terms  # noqa: E402

PROMPT_SYS_BASE = (
    "你是中文/英文技术文档校对助手。只做语义、语法、错别字、术语一致性、描述与事实是否自洽的判断，"
    "不要报纯排版/空格类问题（那些另有工具）。对每个问题给出：行号(line)、问题类型(category:"
    "语法/语义/术语/事实)、级别(severity: error/warning/info)、原文片段(snippet)、"
    "具体修改建议(suggestion)。无法确定真伪的标 severity=info 并在 suggestion 注明'待人工核实'。"
    "只输出 JSON 数组，不要多余文字。"
)


def load_domain_terms():
    """读取领域保护术语，注入提示词：这些是正确的品牌名/专业术语，不可当错别字。"""
    return sorted(_load_domain_terms())


def build_system_prompt():
    terms = load_domain_terms()
    if not terms:
        return PROMPT_SYS_BASE
    return (
        PROMPT_SYS_BASE
        + "\n\n【领域保护术语，均为正确写法，绝不可标为错别字/拼写/术语错误，并作为理解上下文的依据】：\n"
        + "、".join(terms)
    )


def build_user_prompt(path, text):
    numbered = "\n".join("%d: %s" % (i, ln) for i, ln in enumerate(text.splitlines(), 1))
    return "文件：%s\n以下为带行号的文档内容，请按上述要求复核并只返回 JSON 数组：\n\n%s" % (path, numbered)


def call_llm(api_base, model, key, messages, timeout=120):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer %s" % key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8", "replace"))
    # 内容被审核过滤、模型限流返回空结果等场景下 choices 可能为空，
    # 直接取 [0] 会抛 IndexError；此处显式转成 ValueError 由调用方优雅跳过。
    choices = obj.get("choices")
    if not choices:
        raise ValueError("LLM API 返回空 choices")
    return choices[0]["message"]["content"]


def parse_findings(content, path):
    txt = content.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt[txt.find("[") :]
    start, end = txt.find("["), txt.rfind("]")
    if start < 0 or end < 0 or end < start:
        sys.stderr.write(
            "[llm] %s：响应中未找到 JSON 数组，已跳过。原始响应前 200 字符：%s\n" % (path, txt[:200].replace("\n", " "))
        )
        return []
    try:
        arr = json.loads(txt[start : end + 1])
    except ValueError as e:
        # 解析失败时给出告警，避免与“确实没问题”混淆（否则用户只看到命中 0 条）。
        sys.stderr.write(
            "[llm] %s：JSON 解析失败（%s），已跳过。原始响应前 200 字符：%s\n" % (path, e, txt[:200].replace("\n", " "))
        )
        return []
    out = []
    for it in arr:
        out.append(
            dict(
                file=path,
                line=it.get("line", 0),
                rule="llm-%s" % it.get("category", "semantic"),
                severity=it.get("severity", "info"),
                engine="llm",
                snippet=str(it.get("snippet", ""))[:120],
                message=str(it.get("message", it.get("category", "语义问题"))),
                suggestion=str(it.get("suggestion", "")),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="LLM 文档语义复核（需用户提供接口，默认 dry-run）")
    ap.add_argument("paths", nargs="+", help="要复核的 markdown 文件")
    ap.add_argument("--api-base", help="OpenAI 兼容接口地址，如 https://host/v1")
    ap.add_argument("--model", help="模型名")
    ap.add_argument("--api-key-env", help="存放密钥的环境变量名（只从此 env 读取）")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--max-chars", type=int, default=12000, help="单文件最大送入字符数")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    g.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    args = ap.parse_args()

    missing = [
        n
        for n, v in [("--api-base", args.api_base), ("--model", args.model), ("--api-key-env", args.api_key_env)]
        if not v
    ]
    if missing:
        sys.stderr.write(
            "LLM 复核未配置接口，已拒绝运行。\n"
            "本层默认不启用：请先与用户确认是否启用，并由用户提供以下参数：%s\n"
            "示例：--api-base https://<endpoint>/v1 --model <model> --api-key-env <ENV_NAME>\n" % ", ".join(missing)
        )
        return 2

    key = os.environ.get(args.api_key_env, "").strip() if args.api_key_env else ""
    system_prompt = build_system_prompt()

    all_findings = []
    for path in args.paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()[: args.max_chars]
        except OSError as e:
            sys.stderr.write("[llm] 跳过 %s：%s\n" % (path, e))
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(path, text)},
        ]

        if args.dry_run or not key:
            if not key and not args.dry_run:
                sys.stderr.write("[llm] 环境变量 %s 未设置，转 dry-run。\n" % args.api_key_env)
            sys.stderr.write(
                "[llm][dry-run] 将 POST %s/chat/completions，model=%s，文件=%s\n"
                % (args.api_base.rstrip('/'), args.model, path)
            )
            sys.stderr.write("[llm][dry-run] 认证：Authorization: Bearer $%s（隐藏）\n" % args.api_key_env)
            continue
        try:
            content = call_llm(args.api_base, args.model, key, messages)
            all_findings.extend(parse_findings(content, path))
        except (urllib.error.URLError, KeyError, ValueError, IndexError) as e:
            sys.stderr.write("[llm] 调用失败 %s：%s\n" % (path, e))

    if args.dry_run:
        sys.stderr.write("[llm] dry-run 结束，未联网、未产生结果。\n")
        return 0

    if args.format == "json":
        print(json.dumps(all_findings, ensure_ascii=False, indent=2))
    else:
        for f in all_findings:
            print(
                "[llm/%s] %s:%s %s\n    %s -> %s"
                % (f["rule"], f["file"], f["line"], f["severity"], f["message"], f["suggestion"])
            )
    sys.stderr.write("[llm] 复核完成，命中 %d 条。\n" % len(all_findings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
