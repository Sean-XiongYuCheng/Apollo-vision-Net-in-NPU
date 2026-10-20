# DrivingSDK 文档质量体检与纠错

面向 DrivingSDK 仓库中英文文档（`docs/`、各 `model_examples/*/README.md`、根目录 md）的质量体检与纠错 skill。
按「问题怎么解决」分四道处理，**纯 CPU、无需 GPU**，外部工具装了才启用（未装自动跳过并提示安装方式）。

## 使用方法

向 AI 助手调用该 skill，说明要对哪些文档做体检即可触发，例如：

```text
用 drivingsdk-doc-quality-review 检查一下 docs/ 和 model_examples 的文档，给我一份问题清单
```

也可直接手动运行脚本（见下）。完整设计与流程见 [`SKILL.md`](./SKILL.md)，真实实测记录见 [`PRACTICE.md`](./PRACTICE.md)。

## 检测内容（四道）

| 道 | 引擎 | 覆盖问题 | 处置 |
| --- | --- | --- | --- |
| structure | 内置正则（恒开、无依赖） | 坏链/坏图、代码块未闭合、标题缺空格、中英括号混用、相邻重复标点、英文重复词、占位残留、中文常见错别字 | 进 issue 逐条修 |
| format | autocorrect（可选） | 中英/数字间空格、半角↔全角标点等可机械修复的排版 | 批量 `--fix`，单独格式化 PR |
| terminology | Vale + 仓内词表（可选） | 术语大小写一致性（`DrivingSDK`/`mx-driving` 等） | 聚合进 issue / 批量替换 |
| semantic | 代码助手自身 LLM（默认） | 任意错别字、语义歧义、**描述与代码是否一致**、过期信息 | LLM 复核进 issue |

## 手动运行

```bash
SK=.agent/skills/drivingsdk-doc-quality-review/scripts

# 1. 看哪些道可用（缺工具会提示安装方式）
python3 $SK/run_review.py --list-engines

# 2. 全量多道检查，产出 JSON + 逐行 JSONL
python3 $SK/run_review.py . --json /tmp/review.json --jsonl /tmp/findings.jsonl

# 3. 格式道批量修（默认 dry-run 预览，--apply 才写盘）
python3 $SK/doc_fix.py docs README.md            # 预览
python3 $SK/doc_fix.py docs README.md --apply     # 写盘

# 4. 生成本地 issue 草稿
python3 $SK/gitcode_issue.py draft /tmp/review.json -o /tmp/issue_draft.md
```

## 可选外部工具安装

- **autocorrect**（format 道）：从 <https://github.com/huacnlee/autocorrect/releases> 下预编译二进制放进 PATH。
- **Vale**（terminology 道）：从 <https://github.com/errata-ai/vale/releases> 下二进制放进 PATH，配置已内置于 `assets/vale/.vale.ini`。
- **LanguageTool**（grammar 道，opt-in）：解压后 `export LT_JAR=.../languagetool-commandline.jar`，需 Java。

## 设计要点

- **不把可自动修复的排版当 issue**：一次全量扫描就有数千处空格/标点项，交给 `doc_fix.py --fix` 批量修并单独提格式化 PR，避免淹没真问题。
- **精度优先**：只引入经实测低误报的工具；对昇腾/自动驾驶领域词用保护词表（`assets/domain_terms.txt`）剔除误报。
- **与仓内 pre-commit 互补**：本仓已启用 `typos`（英文拼写，带白名单），本 skill 聚焦 typos 管不到的结构/链接/术语/语义。
- **对外动作先确认**：创建 issue/PR 前一律先 dry-run；token 只从环境变量读，绝不写盘/打印。
