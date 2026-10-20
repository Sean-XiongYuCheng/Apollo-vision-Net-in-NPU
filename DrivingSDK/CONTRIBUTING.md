# 开发Driving SDK

## 贡献声明

贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)。
通常，Driving SDK 一年会有 4 个正式 release 版本。
如果您遇到 bug，请[提交 issue](https://gitcode.com/Ascend/DrivingSDK/issues)。
如果您计划贡献 bug-fixes，请提交 Pull Requests，参见[具体要求](#pull-request)。
如果您计划贡献新特性、功能（如新增模型、新增算子等），请先创建 issue 与我们讨论。写明模型或算子名称，需求背景/目的，如何设计，对现有仓的影响。未经讨论提交 PR 可能会导致请求被拒绝，因为项目演进方向可能与您的想法存在偏差。

新增算子和模型的具体开发指南可分别参考：

* [算子开发指南](./docs/zh/operator_adaption/from-scratch.md)
* [模型迁移指南](./docs/zh/migration_tuning/model_migration.md)
* [模型性能调优指南](./docs/zh/migration_tuning/model_optimization.md)

## 目录

- [Pull Request](#pull-request)
  - [PR 创建流程](#pr-创建流程)
  - [PR 最佳实践](#pr-最佳实践)
  - [代码提交前检查](#代码提交前检查)
  - [PR 评审与合入规则](#pr-评审与合入规则)
- [Special Interest Group](#special-interest-group)

## Pull Request

### PR 创建流程

1. **Fork 本项目的仓库**
2. **Clone 到本地**
3. **创建开发分支**
4. **进行开发**
   * 编写代码
   * 添加测试
   * 更新文档
   * 使用 pre-commit 工具对代码和文档进行格式检查与修复（见下方[代码提交前检查](#代码提交前检查)）
   * 确保代码通过本地测试
5. **提交代码**
6. **推送到 Fork 仓库**
7. **创建 Pull Request**
   * 访问 GitCode 仓库页面
   * 点击“Pull Request”或“合并请求”
   * 填写 PR 描述（见 PR 创建页面模板）

### PR 最佳实践

1. **保持 PR 小规模**
   * 一次 PR 只解决一个问题
   * 便于评审和理解
   * 提高合并效率
   * 建议：单个 PR 不超过 1000 行（含测试）代码变更
2. **及时更新**
   * 定期同步上游主分支
   * 及时响应评审意见
   * 保持 PR 活跃
3. **清晰描述**
   * 详细描述变更原因和方式
   * 提供测试方法
   * 添加截图或示例（如适用）

### 代码提交前检查

为了确保提交的代码和文档符合项目规范，推荐使用 [pre-commit](https://pre-commit.com/) 工具在提交前自动进行格式检查和修复。pre-commit 是一款基于 Git 钩子的开源代码质量管控工具，在执行 `git commit` 前会自动完成代码校验、格式规范化、基础问题排查等工作，校验不通过则阻断提交流程，从源头保障代码规范统一、质量合规。

#### 安装与配置

```shell
# 配置国内pip镜像源，提升安装下载速度，解决安装卡顿、超时等问题。
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# 安装 pre-commit
pip install pre-commit

# 注册Git钩子，后续执行 git commit 命令时，自动触发代码检查流程，无需手动调用。
pre-commit install
```

#### 使用方式

```shell

#执行代码检查，扫描Git暂存区内的改动文件，自动完成代码格式化与合规性检查，并完成格式化问题代码自动修复，其它未能自动修复的错误请参考提示人工修复。
git add .
pre-commit run

# 钩子安装成功后，提交代码时可自动修复代码格式类问题，若没有其他不能自动修复的代码问题可提交成功；否则会提交失败，请人工解决其他不能自动修复的问题后再行commit。
git add .
git commit -S -m "test"
```

如需了解更详细的配置说明、使用指导，或在使用过程中遇到问题，请参考 [Ascend 社区 pre-commit 使用指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

### PR 评审与合入规则

#### 评审要求

1. **评审人员要求**
   * 评审人员必须熟悉相关代码领域
   * 评审人员不能是 PR 作者本人
2. **评审检查项**
   * ✅ 代码质量和风格
   * ✅ 功能正确性
   * ✅ 测试覆盖率（分支 60%，行 80%）
   * ✅ 文档完整性
   * ✅ 性能影响
   * ✅ 安全性
   * ✅ 向后兼容性
3. **CI 检查要求**
   * ✅ 所有 CI 检查必须通过
4. **无 Block 评论**
   * PR 不能有任何未解决问题

#### 合入规则

1. **Squash and Merge**
   * 将 PR 的所有提交合并为一个提交
   * 保持主分支历史清晰
   * 提交消息使用 PR 标题
2. **必须满足的条件**
   * ✅ 至少 1 位 Maintainer 或 Committer 的 `/lgtm`，和 1 个 `/approve`
   * `/lgtm`（Looks Good To Me）：表示代码评审通过
   * `/approve`：表示正式批准合入
3. **禁止的操作**
   * ❌ 禁止 Force Push 到主分支
   * ❌ 禁止合并自己的 PR（必须有他人评审）

#### 合并权限

* **Maintainer**：可以合并任何 PR
* **Committer**：可以合并任何 PR
* **Contributor**：无合并权限，需要等待 Maintainer 或 Committer 合并

## Special Interest Group

### 例会

* 周期：每 1 个月举行一次例会，可通过 [Ascend 开源社区](https://meeting.ascend.osinfra.cn/)搜索、查看 sig-DrivingSDK 的会议链接。
* 申报议题：通过 [sig-DrivingSDK Etherpad 链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)进入共享文档，编辑申报议题。
* 参会人员：maintainer、committer、contributor 等核心成员，其他对本 SIG 感兴趣的人员。
* 会议内容：讨论遗留问题和进展；当期申报的议题；需求评审、任务和优先级；需求规划和进展（roadmap）；新晋 maintainer、committer 准入评审。
* 会议归档：会议纪要位于 [sig-DrivingSDK Etherpad 链接](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK)。

### 成员列表

[SIG 成员列表](https://gitcode.com/Ascend/community/blob/master/MindSDK/sigs/DrivingSDK/sig-info.yaml)。
