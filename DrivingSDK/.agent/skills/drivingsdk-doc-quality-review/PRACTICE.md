# doc-quality-review 实践文档

本文记录 `drivingsdk-doc-quality-review` 在 DrivingSDK 仓库上的设计落地与**真实实测**。

- 对象：DrivingSDK（`gitcode.com/Ascend/DrivingSDK`，master），全量 **155 个 markdown**（语言分流：zh 97 / en 2 / mixed 56；已排除 skill 自身文档）。
- 工具：实测在 PATH 提供 `autocorrect 2.16.3` 与 `vale 3.9.1`，四道里的 structure/format/terminology 全部跑通；grammar(LanguageTool) 需 Java，本轮 opt-out。
- 架构：四道按「问题怎么解决」分流，每道选用经实测有效的工具（否决/采纳理由见 `SKILL.md`）。

## 一、四道架构

| 道 | 工具 | 处理 |
| --- | --- | --- |
| structure | 内置 `scan_docs.py`（恒开） | 坏链/未闭合代码块/括号混用/标点/中文错别字等需人判断 → 进 issue |
| format | `autocorrect`（可选，`zhlint` 备选） | 中英空格/半全角标点 → 批量 `--fix`，不进 issue |
| terminology | `Vale` + 词表（可选） | 术语大小写一致性 → 聚合进 issue / 批量替换 |
| semantic | in-agent LLM（默认）/ `llm_review.py`（headless） | 语义/事实/描述与代码一致 → 进 issue |

为什么这么分：**可机械自动修复的排版**（空格、标点宽度）量极大（本仓 2472 处），不该逐条进 issue 把真问题淹没——
它们交给格式化器 `--fix`；内置正则只管格式化器不碰、又需人判断的结构问题；术语用 Vale 词表维护；语义交给 LLM。

## 二、如何运行

```bash
SK=.agent/skills/drivingsdk-doc-quality-review/scripts
python3 $SK/run_review.py --list-engines                          # 看哪些道可用
python3 $SK/run_review.py . --json /tmp/review.json --format md    # 多道检查
python3 $SK/doc_fix.py docs README.md                             # 格式道预览（--apply 写盘）
python3 $SK/gitcode_issue.py draft /tmp/review.json -o /tmp/issue.md   # issue 草稿
```

## 三、真实仓库实测结果

启用 structure + format + terminology 三道（autocorrect + vale 在 PATH），扫 155 个 markdown：

### structure（2 条，逐条进 issue）

| 文件:行 | 规则 | 级别 | 问题 |
| --- | --- | --- | --- |
| `docs/zh/api/context/npu_voxel_pooling_train.md` 末尾 | fence-unbalanced | error | 第 43 行 ` ```python ` 代码块到文件结束（L76）未闭合 |
| `docs/zh/migration_tuning/model_optimization.md`:173 | dup-punct | warning | `aten::item，.npu()，.cpu()` 中方法名未用行内代码标注，`，.` 触发重复标点告警；建议写成 `` `.npu()`、`.cpu()` `` |

> DrivingSDK 文档结构整体较干净：全仓仅 1 个 error（未闭合代码块）+ 1 个 warning（低置信、属排版精修）。

### terminology（Vale，本轮 0 条自动命中）

- 采用**保守规则**：只规范无歧义的大小写笔误（`drivingsdk→DrivingSDK`、`Driving Sdk→Driving SDK`、`Mx-Driving→mx-driving`），本仓未出现这些笔误，故 0 命中。
- 规则本身经验证有效：对注入的 `drivingsdk`/`Driving Sdk`/`Mx-Driving` 均能命中，对真实文档零误报。
- **刻意不做的事**：本仓 `Driving SDK`（221 处）与 `DrivingSDK`（186 处）两种写法数量相当、均被官方使用（GitCode 仓名 `DrivingSDK`、文档标题 `Driving SDK`），属需维护者定夺的规范问题，**不武断自动统一**（否则会产出上百条方向可疑的"修复"）。作为发现项列入 issue 交维护者决策。

### format（聚合汇总，不进 issue）

- **2472 处**可自动修复的中英排版/标点，涉及 **147 个文件**。Top：`model_examples/CenterPoint/README.md` 82、`model_examples/Cosmos-Transfer1/README.md` 64、`docs/zh/faq/faq.md` 64、`docs/zh/operator_adaption/from-scratch.md` 63、`model_examples/MapTR/README.md` 61……
- 处理方式：`doc_fix.py <paths> --apply` 批量修，单独提一个格式化 PR，不进 issue。

### semantic（in-agent LLM，通读 + 核对代码）

机械三道查不出、需通读或核对代码的：

- `docs/zh/get_started/quick_start_guide.md`:124 —「并**将将**权重文件放置在」重复「将」字，应为「并将权重文件」（warning，中文叠词机械道刻意不查，避免误报）。
- `docs/zh/get_started/quick_start_guide.md`:122 —「获取**权**模型预训练权重」疑「权」字冗余/错位，应为「获取模型预训练权重」（warning）。
- 全仓产品名 `Driving SDK`/`DrivingSDK` 不统一（见 terminology），建议维护者确定规范写法后统一（info）。

**代码一致性抽查（in-agent grep/read 核对，本轮通过）**：

- `quick_start_guide.md` 示例 `from mx_driving import scatter_max`，签名 `out, argmax = scatter_max(updates, indices, out)` 与 `mx_driving/__init__.py` 导出及 `_C` 声明一致。
- `Introduction.md` 提及的 UniAD / SparseDrive / BEVFormer / MapTR / OpenVLA / Cosmos 在 `model_examples/` 下均真实存在。
- `quick_start_guide.md` 引用的 `model_examples/BEVFusion/install_BEVFusion.sh`、`test/train_full_8p_base_fp32.sh` 均存在。

## 四、issue 产出对比（设计改进的实证）

把同一次扫描结果灌进 `gitcode_issue draft`：

- **若旧式平铺**（autocorrect 命中逐条进 issue）：草稿约 **2504 行**，其中 2472 行是空格/标点 nit，真问题被淹没。
- **本设计**（format 折叠、terminology 聚合）：草稿 **35 行**——2 条结构问题逐条 + 术语聚合说明 + 1 句格式化汇总；语义发现另行人工补入。

## 五、日常化与落地

- 本仓 `.pre-commit-config.yaml` **不排除 .md** 且**已启用 `typos`**（带 `pre-commit/typos.toml` 领域词白名单）+ 基础钩子。本 skill 与其**互补**：typos 管英文拼写，本 skill 补坏链/未闭合代码块/括号/标点/中文错别字/语义与一致性。
- structure 道适合 CI / pre-commit（error 级非零退出）：可加 local hook 跑 `run_review.py --engines structure`。
- format 道走 `--fix` 出格式化 PR；structure/terminology/semantic 出 issue → `docs:` PR 闭环。
- LLM 语义道默认由代码助手 in-agent 承担（无需外部接口）；headless/CI 场景才用 `llm_review.py` 配外部接口。

> 复现说明：本文结果基于当前实跑（2026-07 master）；修复合入后条目会变化，以当次扫描为准。
