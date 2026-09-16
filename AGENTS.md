# 项目协作规范

## 仓库信息

- 远程仓库：`git@github.com:youthWorld/rag-center.git`

## 工作流程

- 每完成一个阶段性任务并通过用户要求的验收测试后，通知用户进行最终 review。
- 阶段性任务包括项目骨架搭建、功能开发、Bug 修复和文档变更等(参考下面提交信息中的“**类型**”)。

## 歧义处理

- 当任务目标、范围、验收标准或实现约束存在歧义时，应暂停实施，先与用户充分讨论并确认解决方案；在达成一致前，不基于未经确认的假设进行可能影响结果的修改。

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

| 类型       | 用途                     |
| ---------- | ------------------------ |
| `feat`     | 新功能                   |
| `fix`      | Bug 修复                 |
| `refactor` | 不改变功能的重构         |
| `test`     | 添加或修改测试           |
| `docs`     | 文档变更                 |
| `style`    | 不影响逻辑的格式调整     |
| `chore`    | 工程维护、依赖或配置变更 |
| `perf`     | 性能优化                 |

日常优先使用 `feat`、`fix`、`refactor` 和 `chore`。

### 编写要求

- 使用动词开头和现在时，例如 `add`、`fix`、`update`。
- 首字母小写，不超过 50 个字符，结尾不加句号。
- `scope` 可选，应与项目模块对应。
- 复杂改动可增加 body 和 footer；在 body 中说明重要架构决策的原因。
- 不兼容变更需补充 `BREAKING CHANGE` 及影响范围。
- 关联 Issue 可使用 `Closes #234` 等 footer。

## 分支命名策略

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

## Git 开发与合并流程

需要独立分支开发时，必须在实现功能前基于最新主分支创建功能分支，具体分支命名类型见分支命名策略。

```bash
git switch master
git pull --ff-only origin master
git switch -c feature/<task-name>
```

本地开发完成并通过必要的测试后，通知用户进行测试和review，按以下顺序完成提交和合并：

1. 在功能分支提交本地改动。
2. 先将本地功能分支推送到远程同名功能分支，作为云端备份：

   ```bash
   git push -u origin feature/<task-name>
   ```

3. 切回本地主分支并拉取远程最新代码，确保合并基准与远程完全同步：

   ```bash
   git switch master
   git pull --ff-only origin master
   ```

4. 将本地功能分支合并到本地主分支：

   ```bash
   git merge feature/<task-name>
   ```

   > 若合并出现冲突，则暂停并和用户协商，手动修改冲突文件标记的内容后，执行 `git add <冲突文件>` + `git commit` 完成合并；建议优先在功能分支内解决完所有冲突，再合入主分支。

5. 确认合并结果可正常编译 / 启动、核心功能验证通过后，将本地主分支推送到远程主分支：

   ```bash
   git push origin master
   ```

6. 本地和远程功能分支均不删除，保留为只读归档记录。已合并的功能分支不再继续开发；后续任务应从最新的 `master` 重新创建新的功能分支。
