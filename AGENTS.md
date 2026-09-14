# 项目协作规范

## 仓库信息

- 远程仓库：`git@github.com:youthWorld/rag-center.git`

## 工作流程

- 每完成一个阶段性任务并通过用户要求的验收测试后，通知用户进行最终 review。
- 阶段性任务包括项目骨架搭建、功能开发、Bug 修复和文档变更等(参考下面提交信息中的“**类型**”)。

## 提交信息

提交信息使用以下格式：

```text
<type>(<scope>): <description>
```

常用示例：

```text
feat(auth): add email verification flow
fix(cart): correct total price calculation
refactor(api): extract common response handler
```

### 类型

| 类型 | 用途 |
| --- | --- |
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `refactor` | 不改变功能的重构 |
| `test` | 添加或修改测试 |
| `docs` | 文档变更 |
| `style` | 不影响逻辑的格式调整 |
| `chore` | 工程维护、依赖或配置变更 |
| `perf` | 性能优化 |

日常优先使用 `feat`、`fix`、`refactor` 和 `chore`。

### 编写要求

- 使用动词开头和现在时，例如 `add`、`fix`、`update`。
- 首字母小写，不超过 50 个字符，结尾不加句号。
- `scope` 可选，应与项目模块对应。
- 复杂改动可增加 body 和 footer；在 body 中说明重要架构决策的原因。
- 不兼容变更需补充 `BREAKING CHANGE` 及影响范围。
- 关联 Issue 可使用 `Closes #234` 等 footer。

## 分支策略

- `main` 或 `master`：仅保存稳定、可部署代码，不直接进行功能开发。
- `feature/*`：新功能开发。
- `fix/*`：问题修复。
- `refactor/*`：不改变功能的重构。
- `chore/*`：依赖、配置和工程维护。

命名示例：

```text
feature/user-login
fix/cart-calculation
refactor/auth-middleware
chore/upgrade-deps
```
