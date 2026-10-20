---
name: "drivingsdk-doc-quality-review"
description: "体检与修复 DrivingSDK 仓库中英文文档质量。按「问题怎么解决」分四道（纯 CPU、无需 GPU）：structure(内置正则)查坏链/未闭合代码块/括号混用/标点/中文错别字等需人判断的结构问题；format(autocorrect)批量自动修中英空格/半全角标点；terminology(Vale)查术语大小写一致性；semantic(LLM,默认由代码助手自身承担)查任意错别字/语义/描述与代码是否一致。结构与术语问题整理成 issue（本地草稿 / GitCode API），格式问题走 doc_fix.py 批量修。触发：「文档纠错」「检查文档」「doc review」「文档质量」「扫描 README/docs」「提文档 issue」「日常文档体检」。"
---

# DrivingSDK 文档质量体检与纠错

## 概览

对仓库的 markdown 文档（**中英文混排**）按**问题怎么解决**分四道处理，而不是把所有命中堆成一张表。
用 `scripts/run_review.py` 编排，各道独立、装了才跑：

| 道 | 工具 | 查什么 | 怎么处理 |
| --- | --- | --- | --- |
| **structure** | 内置 `scan_docs.py`（恒开、无依赖） | 坏链/坏图、代码块未闭合、标题缺空格、**括号中英混用**、相邻重复标点、英文重复词、占位残留、**中文常见错别字（混淆集）** | 需人判断 → **进 issue** |
| **format** | `autocorrect`（可选） | 中英/数字间空格、半角↔全角标点等**可机械自动修复**的排版 | 批量 `--fix` → **走 `doc_fix.py`，不逐条进 issue** |
| **terminology** | `Vale` + 仓内词表（可选） | 术语大小写一致性（如 `DrivingSDK`/`mx-driving` 的错误大小写） | 聚合 → **进 issue / 批量替换** |
| **grammar** | `LanguageTool`（可选） | **英文语法 + 拼写**；中文产出有限 | 进 issue |
| **semantic** | **代码助手自身**（默认）/ `llm_review.py`（headless 备选） | **任意错别字、语法、表述歧义、描述与代码/事实不符、逻辑矛盾、过期信息** | LLM 复核 → **进 issue** |

issue 候选（structure + terminology + grammar）用 `scripts/gitcode_issue.py` 整理：默认产出**本地草稿(md)**，可选调 **GitCode API** 创建。

> 设计取舍：可机械自动修复的排版（空格/标点宽度）交给成熟格式化器走 `--fix`，**不当 issue**（本仓一次扫描就有数千处，逐条进 issue 会淹没真问题）；
> 内置正则只做格式化器不管、又需人判断的结构问题；术语用 Vale 词表维护；语义交给 LLM。
> 引擎经实测裁剪，理由见下「已评估并否决的工具」一节。

## 与本仓 pre-commit 的关系（DrivingSDK 特有，务必先读）

DrivingSDK 的 `.pre-commit-config.yaml` **不排除 markdown**（仅排除 `LICENSES/` 与 `.html/.csv/.svg`），且**已启用 `typos`**（配置 `pre-commit/typos.toml`，内含领域词白名单）以及 `trailing-whitespace`、`end-of-file-fixer` 等基础钩子。因此：

- **英文拼写**已由仓内 `typos` 覆盖（带 curated 白名单，如 `CANN`、`thw` 等领域词已豁免）——本 skill **不重复做英文拼写**，structure 道也刻意不与 typos 争抢，转而聚焦 typos 管不到的**坏链、未闭合代码块、中英括号混用、标点、中文错别字**等。
- 本 skill 与 pre-commit **互补而非替代**：pre-commit 是提交时的行级门禁；本 skill 是整仓周期性体检 + issue 化 + 语义/一致性复核。
- 若要把结构检查也纳入 CI 门禁，可将 `run_review.py`（命中 error 级返回非零退出）接为一个 local hook（见末尾）。

## 能做什么 / 不能做什么（务必先读）

**能稳妥做到（纯 CPU，无需 GPU）：**

- ✅ **结构问题**：失效的相对链接/图片、代码块 ``` 未闭合、标题 `#` 缺空格、中英括号混用 `(x）`、相邻重复标点 `。。`、英文重复词 `the the`、残留 `TODO/FIXME`。高置信、低误报，进 issue。
- ✅ **中文常见错别字**：内置**混淆集**（`assets/cn_confusion.txt`，如 `因该→应该`、`建意→建议`、`按装→安装`）确定性匹配，**零误报、可持续扩充**；命中即真。
- ✅ **中英排版/标点（自动修复）**：中英文与数字间空格、半角↔全角标点规范化。由 `doc_fix.py`（autocorrect/zhlint）批量 `--fix`，出格式化 PR。
- ✅ **术语大小写一致性**：用 Vale 词表强制（如 `drivingsdk→DrivingSDK`、`Mx-Driving→mx-driving`），可扩充。
- ✅ **英文语法 + 英文拼写**：LanguageTool（a/an、主谓、拼写）为可选增强；英文拼写日常已由仓内 `typos` 兜底。
- ✅ **任意错别字 + 语义/事实/代码一致性**（**交互式运行时，由代码助手自身的 LLM 承担，无需外部 API**）：超出混淆集的错别字（如「将将」这类重复字）、表述歧义、**描述与代码不符**（助手会 grep/read 代码核对接口名/参数/路径/默认值）、逻辑矛盾、过期信息。
- ✅ **issue 产出**：本地草稿(md) + 可选 GitCode API 提交（先 dry-run）。

**做不到 / 有边界（已实测确认）：**

- ❌ **纯统计模型自动纠中文错别字**：pycorrector(KenLM) 在昇腾/中英混排语料上**误报灾难**（把 `昇腾→升腾` 等乱改），LanguageTool 中文几乎无产出。**故不走"模型猜错字"路线**，改用「混淆集（覆盖常见）+ 代码助手 LLM（覆盖任意）」组合。代价：混淆集召回受限于词表（需逐步扩充）。
- ❌ **强制统一有歧义的品牌写法**：本仓 `Driving SDK` 与 `DrivingSDK` 两种写法数量相当、均被官方使用，**不武断二选一自动改**（否则产出上百条方向可疑的"修复"）；terminology 道只规范无歧义的大小写笔误，写法方向作为发现项交维护者定夺。
- ❌ **跨文件/页内锚点（`#小节`）有效性校验**：成熟工具（markdownlint MD051、lychee fragments）在中文锚点上都误报，**刻意不做**。
- ❌ **自动 commit/push 修复**：只产出 issue 草稿或格式化 diff 预览，落盘需 `--apply`、提交/发 issue 都要人确认。

## 已评估并否决 / 采纳的工具（实测留痕，避免重复趟坑）

| 工具 | 结论（在中英混排、领域词多的语料上实测） |
| --- | --- |
| `typos` | **本仓已采纳**（pre-commit + curated `pre-commit/typos.toml` 白名单）；本 skill 不重复做英文拼写，structure 道与其互补。若无白名单则误报高（`CANN→CAN` 等），采纳前提是维护领域词白名单。 |
| `cspell` | 真实中文文档零命中（无增量）；英文拼写本仓已由 `typos` 覆盖 |
| `markdownlint-cli2` | 数千条风格噪声（MD033 内联 HTML、MD060 表格对齐、MD013 行长），本仓文档大量使用 HTML/表格，噪声淹没真问题 |
| `textlint`+中文规则 | `textlint-rule-preset-chinese` 不存在(404)，无可用中文规则集 |
| `lychee` | 本地链接与内置重复；**中文锚点片段误报**；外链对 soft-404 不可靠、需联网 |
| `pycorrector`(KenLM/CPU) | 干净例句有效，但真实技术文档**误报灾难**（`昇腾→升腾` 等，误纠率高）；依赖 torch + ~3GB 模型 |

> 取舍原则：**精度优先**——宁可少做，不引入会把正确内容改错的工具。

## 何时使用

- 用户要求「检查/修复文档错误」「做一轮文档体检」「给文档提 issue」。
- 文档 PR 合入前自查，或定期（如每周）文档质量巡检。
- CI/pre-commit 想加文档结构门禁（`run_review.py` 命中 error 级返回非零退出码）。

不适用：需要改代码逻辑、或要新写大段技术内容——那是开发/写作任务，不是纠错。

## 执行流程

开始前用任务清单记录阶段：

```text
阶段 1：run_review 多道检查（structure/format/terminology）
阶段 2：格式道 doc_fix 批量修（单独格式化 PR）
阶段 3：语义道复核（LLM 或人工）
阶段 4：LLM 汇总把关，整理 issue（结构+术语+语义）并产出
```

### 阶段 1：run_review 多道检查

```bash
SK=.agent/skills/drivingsdk-doc-quality-review/scripts
python3 $SK/run_review.py --list-engines                          # 先看哪些道可用
python3 $SK/run_review.py . --json /tmp/review.json --format md    # 全量跑
# 逐行 JSONL（一行一问题，自动带 run_ts，便于后续查看/处理/累积跟踪）：
python3 $SK/run_review.py . --jsonl /tmp/findings.jsonl
python3 $SK/run_review.py . --jsonl /tmp/issues_log.jsonl --jsonl-append   # 追加到历史日志
```

- `--jsonl` 每行一条问题（含 `file/line/lane/rule/severity/snippet/message/suggestion/run_ts`）；`--jsonl-append` 追加到同一文件，便于跨轮累积、grep/jq 处理与跟踪闭环。
- **外部工具未安装时**：`run_review` 会在 stderr 打印「未启用的道 + 安装命令」提醒；agent 应据此**告知用户缺哪些工具、如何装，并询问是否安装**后再决定是否重跑（`structure` 道无依赖，始终可用）。
- 输出区分「issue 候选」（structure 逐条 + terminology 聚合）与「格式化汇总」（format 只给每文件处数，不刷屏）。
- 语言分流：每个文件判 zh/en/mixed；中英文都覆盖，不要只当中文处理。
- 限定范围：`--include '*/docs/zh/*' --include '/README.md'`；指定道：`--engines structure,terminology`。
- 退出码：structure 命中 **error**（坏链/未闭合代码块）返回 1，可做 CI 门禁。
- 各道工具（都 CPU、无需 GPU，装了才启用）：
  - `structure`：内置，无依赖。
  - `format`：`autocorrect` 二进制（[安装](https://github.com/huacnlee/autocorrect)）。
  - `terminology`：`vale` 二进制（[安装](https://vale.sh)）+ 仓内 `assets/vale/.vale.ini`（已附 DrivingSDK 术语规则，可扩充）。
  - `grammar`：[LanguageTool](https://languagetool.org)（需 Java + `LT_JAR` 指向 `languagetool-commandline.jar`）。较慢、较重，opt-in。

### 阶段 2：格式道批量修（doc_fix.py）

中英空格、半角↔全角标点这类**不进 issue**，用格式化器批量修，单独提一个格式化 PR：

```bash
python3 $SK/doc_fix.py docs README.md            # 默认 dry-run，预览将改什么
python3 $SK/doc_fix.py docs README.md --apply     # 确认后写盘
python3 $SK/doc_fix.py docs --engine zhlint        # 可选更彻底的 zhlint（更激进）
```

- 默认 `autocorrect`（保守：只加空格、规范部分标点，不动括号配对）；`--engine zhlint` 更彻底（含半角→全角、括号宽度，但激进）。
- 默认 dry-run 不写盘；`--apply` 才改。建议在干净 git 工作区跑，改完 `git diff` 复核再提 `docs:` PR。
- 注意与仓内 pre-commit 的 `typos` 等钩子行为对齐，避免格式化 PR 与提交钩子来回打架。

### 阶段 3：语义/语法复核（LLM）

机械三道查不出的——**任意错别字、语法、表述歧义、描述与代码不符、矛盾、过期**——由 LLM 复核。
完整流程与产出格式见 `assets/semantic_review_prompt.md`。两种执行方式：

- **默认（in-agent）**：你（代码助手）就是 LLM，按该 prompt 通读 `docs/zh/`、`docs/en/`、各 `model_examples/*/README.md`，
  并对"描述与代码一致"类逐条 `grep`/`read` 仓库代码核对（如文档写的接口名/参数/默认值/脚本路径是否与 `mx_driving/`、`model_examples/` 一致）。
- **headless（外部 API）**：

  ```bash
  python3 $SK/llm_review.py docs/zh/introduction/Introduction.md \
      --api-base <接口>/v1 --model <模型> --api-key-env <ENV_NAME> --format json
  ```

  默认 dry-run、不联网；未配齐接口直接拒绝运行。**只看文本、无法核对代码**，故一致性判断弱于 in-agent。

判定原则：**拿不准不臆断**——去 `grep`/`read` 核对代码，查不到定论标「待确认(info)」；只提能给出具体改法+依据的问题。

### 阶段 4：LLM 汇总把关 + 产出 issue

**先让 LLM 对前面各道的候选过一遍再出 issue**（关键步骤）：

- 各机械道（structure/terminology/grammar）照常**全量检测、不前置压制**；汇总时把候选交给 LLM（in-agent 优先，按 `assets/semantic_review_prompt.md`）。
- LLM 复核时**带上领域保护术语表 `assets/domain_terms.txt`**（昇腾、算子、稀疏卷积、体素化、点云、CANN、mx_driving、BEVFormer…，均为正确写法）：
  - 据此**剔除误报**（若某候选把保护术语当错了，判误报删掉）；
  - 据此**补充机械道漏掉的语义/一致性问题**；
  - 这些术语同时是理解上下文的依据。
- `llm_review.py` 已自动把该术语表注入系统提示；in-agent 路径需主动读取该表作为提醒。

经把关后的最终清单：**structure（逐条）+ terminology（聚合）+ 语义发现**进 issue；format 折叠成一句汇总。

```bash
# A. 本地草稿（默认，推荐先走）
python3 $SK/gitcode_issue.py draft /tmp/review.json -o /tmp/issue_draft.md
# 把语义道人工发现手动补进草稿，再粘贴到 GitCode 新建 issue

# B. GitCode API 直接创建（可选，需确认）
export GITCODE_TOKEN=<个人令牌>            # 只从环境读，绝不写盘/打印
python3 $SK/gitcode_issue.py create --owner Ascend --repo DrivingSDK \
    --title "[docs] 文档质量问题清单" --body-file /tmp/issue_draft.md --labels docs --dry-run
# 确认无误后去掉 --dry-run
```

- 端点默认 `POST /api/v5/repos/{owner}/{repo}/issues`（Bearer）；404/405 时加 `--path-style owner` 切 Gitee 风格。
- 创建 issue / PR 是对外动作，**真正发送前必须先 `--dry-run` 给用户确认**。

## 与仓库协作约定

- 修复 PR 用 `docs:` 前缀（见 `../../CONTRIBUTING.md`），关联 issue，合入后关闭。
- 走 fork + 开发分支 + PR 流程，PR 描述按仓库模板填写；提交前跑 `pre-commit run`。
- 格式化（format 道 `--apply`）单独成一个 PR，与结构/术语/语义修复分开，便于 review。
- 不在 issue/PR/日志泄露任何 token。

## 日常运行与 CI

```bash
SK=.agent/skills/drivingsdk-doc-quality-review/scripts
python3 $SK/run_review.py . --json /tmp/review.json --format md > /tmp/review.md
# error 级返回非零，可挂 CI / pre-commit / 定时任务
```

本仓 pre-commit 已对 markdown 生效（`typos` + 基础钩子），本 skill 补齐 typos 管不到的结构/链接/术语/语义层面。接入建议：加一个 local hook 跑 `run_review.py --engines structure`（error 级阻断），把 `autocorrect --fix` 作为格式化 hook。

## 验证期望

- `run_review.py --list-engines` 正确反映各道可得性；缺外部工具自动跳过、不报错、structure 仍可用。
- structure 道对干净文档零误报、对注入问题能命中（混淆集只匹配已知错串，命中即真）。
- 中英文文档都覆盖（语言分流）。
- `llm_review.py` 未配齐接口拒绝运行；默认 dry-run、不联网；密钥只从用户指定 env 读、绝不外泄。
- issue 候选每条可定位到 `文件:行` 并有可执行建议；format 大量命中不淹没 issue。
- GitCode 创建前有 dry-run，token 全程不外泄。
