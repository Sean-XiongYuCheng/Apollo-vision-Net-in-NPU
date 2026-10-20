# Driving SDK Development

## Contribution Statement

Before contributing, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930).
Generally, Driving SDK has four official releases per year.
If you encounter a bug, [submit an issue](https://gitcode.com/Ascend/DrivingSDK/issues).
If you plan to contribute bug fixes, submit pull requests. See the [specific requirements](#pull-request).
If you plan to contribute new features (such as new models or operators), create an issue first to discuss them with us. In the issue, provide the model or operator name, the requirement background or purpose, the design approach, and the impact on the existing repository. Submitting a PR without prior discussion may cause it to be rejected, because the project evolution direction may differ from your idea.

For detailed development guides for new operators and models, see the following documents:

* [Operator development guide](./docs/en/operator_adaption/from-scratch.md)
* [Model migration guide](./docs/en/migration_tuning/model_migration.md)
* [Model performance tuning guide](./docs/en/migration_tuning/model_optimization.md)

## Table of Contents

- [Pull Request](#pull-request)
  - [PR Creation Process](#pr-creation-process)
  - [PR Best Practices](#pr-best-practices)
  - [Pre-Commit Checks](#pre-commit-checks)
  - [PR Review and Merge Rules](#pr-review-and-merge-rules)
<!-- - [Special Interest Group](#special-interest-group) -->

## Pull Request

### PR Creation Process

1. **Fork this repository.**
2. **Clone it locally.**
3. **Create a development branch.**
4. **Do the development work.**
   * Write code.
   * Add tests.
   * Update documentation.
   * Use the pre-commit tool to check and fix formatting for code and documentation. See [Pre-Commit Checks](#pre-commit-checks).
   * Ensure that the code passes local tests.
5. **Submit the code.**
6. **Push it to your forked repository.**
7. **Create a pull request.**
   * Go to the GitCode repository page.
   * Click "Pull Request" or "Merge Request".
   * Fill in the PR description. See the PR creation page template.

### PR Best Practices

1. **Keep PRs small.**
   * Each PR should solve only one problem.
   * Smaller PRs are easier to review and understand.
   * They improve merge efficiency.
   * Recommendation: Keep a single PR to no more than 1,000 lines of code changes, including tests.
2. **Update promptly.**
   * Synchronize with the upstream main branch regularly.
   * Respond to review comments promptly.
   * Keep the PR active.
3. **Provide clear descriptions.**
   * Describe the reason for and approach to the changes in detail.
   * Provide test instructions.
   * Add screenshots or examples, if applicable.

### Pre-Commit Checks

To ensure that the submitted code and documentation comply with project standards, you are advised to use the [pre-commit](https://pre-commit.com/) tool to automatically check and fix formatting before submission. `pre-commit` is an open-source code quality control tool based on Git hooks. Before you run the `git commit` command, it automatically performs code checks, normalizes formatting, and detects basic issues. If a check fails, it blocks submission and helps ensure that code standards and quality remain consistent from the outset.

#### Installation and Configuration

```shell
# Configure a geographically closer PyPI mirror to speed up downloads and avoid stalls and timeouts
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple

# Install pre-commit
pip install pre-commit

# Register Git hooks so that the code check process is triggered automatically when you run the git commit command, without manual invocation
pre-commit install
```

#### Usage

```shell

# Run code checks to scan the changed files in the Git staging area, automatically format the code and run compliance checks, and automatically fix formatting issues. For errors that cannot be fixed automatically, follow the prompts to fix them manually
git add .
pre-commit run

# After the hooks are installed successfully, submitting code automatically fixes formatting issues. If there are no other code issues that cannot be fixed automatically, the submission succeeds; otherwise, it fails. Resolve the remaining issues that cannot be fixed automatically before committing
git add .
git commit -S -m "test"
```

<!-- For more detailed configuration instructions and usage guidance, or if you encounter problems during use, see the [Ascend Community pre-commit User Guide](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md). -->

### PR Review and Merge Rules

#### Review Requirements

1. **Reviewer requirements**
   * Reviewers must be familiar with the relevant code area.
   * Reviewers cannot be any of the PR authors.
2. **Review checklist**
   * ✅ Code quality and style.
   * ✅ Functional correctness.
   * ✅ Test coverage. Branch coverage must reach 60 percent and line coverage must reach 80 percent.
   * ✅ Documentation completeness.
   * ✅ Performance impact.
   * ✅ Security.
   * ✅ Backward compatibility.
3. **CI requirements**
   * ✅ All CI checks must pass.
4. **No blocking comments**
   * The PR must have no unresolved issues.

#### Merge Rules

1. **Squash and merge**
   * Merge all commits in the PR into a single commit.
   * Keep the history of the main branch clean.
   * Use the PR title as the commit message.
2. **Required conditions**
   * ✅ At least one `/lgtm` from a maintainer or committer and one `/approve`.
   * `/lgtm` (Looks Good To Me): indicates that the code review has passed.
   * `/approve`: indicates official approval for merging.
3. **Prohibited actions**
   * ❌ Do not force-push to the main branch.
   * ❌ Do not merge your own PR. It must be reviewed by someone else.

#### Merge Permissions

* **Maintainer**: can merge any PR.
* **Committer**: can merge any PR.
* **Contributor**: has no merge permission and must wait for a maintainer or committer to merge the PR.

<!-- ## Special Interest Group

### Meetings

* Frequency: A meeting is held once every month. You can search for the sig-DrivingSDK meeting link on the [Ascend Open Source Community](https://meeting.ascend.osinfra.cn/) website.
* Topic submission: Open the shared document through the [sig-DrivingSDK Etherpad link](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK) and edit the proposed topics.
* Attendees: Core members such as maintainers, committers, contributors, and other people interested in this SIG.
* Meeting content: Discuss unresolved issues and progress, the topics submitted for the current period, requirements review, tasks and priorities, requirements planning and progress (roadmap), and admission reviews for new maintainers and committers.
* Archive: Meeting minutes are available at the [sig-DrivingSDK Etherpad link](https://etherpad.ascend.osinfra.cn/p/sig-DrivingSDK).

### Member List

[SIG member list](https://gitcode.com/Ascend/community/blob/master/MindSDK/sigs/DrivingSDK/sig-info.yaml) -->
