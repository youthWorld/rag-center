# 租户接入与套餐使用

RAG Center 是面向业务方的检索中台。它负责知识库、文档索引和 RAG 检索，`retrieve` 接口返回结构化 chunk、引用来源、分数和运行 metadata；最终答案由业务方调用自己的模型或编排服务生成。

## 开通租户

在项目根目录执行以下命令。新建租户默认使用 `free` 套餐。

```powershell
uv run python scripts/create_tenant.py --id tenant_demo --name "演示租户"
uv run python scripts/create_api_key.py --tenant-id tenant_demo --name "本地开发"
```

`create_api_key.py` 只在创建时打印一次明文 API Key，后续请求使用该 Key：

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/auth/me" `
  -H "Authorization: Bearer <api-key>"
```

`/auth/me` 会返回当前租户的套餐、可用 profile、能力开关、配额上限和当天用量。前端调试台也依赖这个接口决定哪些控件可用。

## 套餐对照

| 套餐 | 可用 profile | hybrid | rerank | query 改写 | QPS | 每日检索 | 知识库 | 每库文档 | 同时索引 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `free` | `speed` | 否 | 否 | 否 | 3 | 500 | 1 | 30 | 1 |
| `standard` | `speed`、`balanced`、`custom` | 是 | 否 | 否 | 10 | 5,000 | 5 | 200 | 2 |
| `pro` | `speed`、`balanced`、`quality`、`custom` | 是 | 是 | 是 | 50 | 100,000 | 50 | 5,000 | 10 |

套餐由服务端校验。调试台只根据 `/auth/me` 提前置灰控件，不能绕过接口限制。

升档使用运维脚本：

```powershell
uv run python scripts/update_tenant_plan.py --tenant-id tenant_demo --plan standard
uv run python scripts/update_tenant_plan.py --tenant-id tenant_demo --plan pro
```

## 检索 profile

`profile` 描述这一次检索采用的策略。每次请求建议显式传入它：

- `speed`：向量召回，默认返回 3 个结果，延迟最低。
- `balanced`：hybrid 融合召回，默认返回 5 个结果。
- `quality`：hybrid 融合召回，并启用 rerank 和 query 改写，默认返回 8 个结果。
- `custom`：由请求体中的高级参数决定，适合调试和对比实验。

### speed

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/rag/retrieve" `
  -H "Authorization: Bearer <api-key>" `
  -H "Content-Type: application/json" `
  -d '{"kb_id":"<knowledge-base-id>","user_id":"user_demo","query":"退款需要几天内申请？","profile":"speed"}'
```

### balanced

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/rag/retrieve" `
  -H "Authorization: Bearer <api-key>" `
  -H "Content-Type: application/json" `
  -d '{"kb_id":"<knowledge-base-id>","user_id":"user_demo","query":"退款需要几天内申请？","profile":"balanced"}'
```

### quality

`quality` 仅对 `pro` 套餐开放：

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/rag/retrieve" `
  -H "Authorization: Bearer <api-key>" `
  -H "Content-Type: application/json" `
  -d '{"kb_id":"<knowledge-base-id>","user_id":"user_demo","query":"退款需要几天内申请？","profile":"quality"}'
```

### custom

`custom` 需要同时传入高级检索参数。`rerank_options` 和 `query_options` 只有在对应套餐能力可用且本次确实要启用时才传入：

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/rag/retrieve" `
  -H "Authorization: Bearer <api-key>" `
  -H "Content-Type: application/json" `
  -d '{"kb_id":"<knowledge-base-id>","user_id":"user_demo","query":"退款需要几天内申请？","profile":"custom","top_k":8,"retrieval_options":{"mode":"hybrid","vector_top_k":20,"bm25_top_k":20,"rrf_k":60},"rerank_options":{"enabled":true,"top_n":5},"query_options":{"enabled":true,"strategy":"rewrite"}}'
```

套餐决定 profile 和能力上限，profile 决定本次请求如何运行。服务端在未传 `profile` 时默认使用 `balanced`；对 `free` 租户应显式传 `speed`，避免请求被套餐校验拒绝。服务端响应的 `metadata.tenant_policy` 会记录实际生效的套餐、profile、检索模式、rerank 和 query 改写状态。

## 词表配置

所有套餐都可以为知识库配置同义词词表。准备一个 JSON 文件，例如 `kb-settings.json`：

```json
{
  "synonyms": [
    {"terms": ["退款", "退货退款"], "expansions": ["退款申请"]}
  ]
}
```

执行整体替换：

```powershell
uv run python scripts/update_kb_settings.py `
  --kb_id <knowledge-base-id> `
  --settings-file .\kb-settings.json
```

词表在 query 处理阶段使用，不受套餐的 `query_rewrite_allowed` 开关影响。`query_rewrite_allowed` 控制的是 LLM query 改写。

## 检索反馈

开启 Langfuse 后，`retrieve` 响应会返回 `metadata.log_id` 和 `metadata.trace_id`。业务方应保存本次检索的 `trace_id`，需要反馈时调用反馈接口提交 1～5 分和可选备注：

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/rag/feedback" `
  -H "Authorization: Bearer <api-key>" `
  -H "Content-Type: application/json" `
  -d '{"trace_id":"<trace-id>","log_id":"<log-id>","score":2,"comment":"排第一的 chunk 不是目标文档"}'
```

反馈会以 `user_feedback` score 写入对应的 Langfuse trace。同一条 trace 只接受一次反馈，重复提交会返回 HTTP 409、错误码 `20023`。运营可以在 Langfuse 按 `user_feedback` 筛选低分 trace，复盘 query 改写、召回和 rerank 的效果。

## 离线检索测评

可以使用仓库中的 [`eval/`](../eval/) 目录搭建检索调优闭环：先从 Langfuse 导出低分反馈，
再人工补齐 `ground_truth`，最后调用 `scripts/run_retrieval_eval.py` 重新检索并计算
`context_precision`、`context_recall`。完整命令和数据集格式见根目录 README 的「离线检索测评」小节。

评测请求固定使用 `user_id=eval_runner`，因此可以在 Langfuse 中和线上检索流量区分；评测报告
写入 `eval/reports/`，不会进入版本库。

## 常见错误码

| code | 含义 | 处理方式 |
| ---: | --- | --- |
| `20010` | 未授权 | 检查 `Authorization: Bearer <api-key>`、Key 状态和过期时间。 |
| `20013` | 功能超出套餐 | 根据 `/auth/me` 的 features 更换 profile 或升级套餐。 |
| `20014` | 配额超限 | 检查 `limits` 和 `usage`，减少知识库、文档或当天检索量，或升级套餐。页面会展示响应中的 `msg`。 |
| `20005` | QPS 限流 | 降低并发或增加请求间隔。 |
| `20020` | 反馈不可用 | 检查 Langfuse 是否启动、配置是否完整。 |
| `20021` | 检索日志与 trace 不匹配 | 确认 `log_id`、`trace_id` 来自同一次检索且属于当前租户。 |
| `20022` | 反馈分数无效 | `score` 必须是 1～5 的整数。 |
| `20023` | 反馈已提交 | 同一条检索只能提交一次反馈。 |

错误响应统一包含 `code`、`msg` 和 `data`，例如：

```json
{
  "code": 20014,
  "msg": "free plan allows only 1 knowledge base",
  "data": {"limit": 1}
}
```

## 旧租户

执行 `0005_tenant_plan` 迁移时，已有租户默认迁移到 `standard`，`tenant_demo` 迁移到 `pro`。迁移完成后，可以使用 `update_tenant_plan.py` 调整套餐；之后新建租户默认仍为 `free`。
