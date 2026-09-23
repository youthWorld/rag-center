# RAG Center

RAG Center 是一个面向业务方提供统一 RAG 能力的后端中台骨架。当前实现知识库与文档生命周期管理、Celery 异步索引，以及 pgvector 向量检索、Elasticsearch BM25 检索和 RRF 混合召回，保留各类 Provider 的扩展边界。

## 技术栈

- Python 3.11
- FastAPI
- PostgreSQL 16 + pgvector
- Elasticsearch 8.17 + analysis-ik
- Redis 7 + Celery
- SQLAlchemy 2.x + Alembic
- Pydantic v2
- OpenAI-compatible Embedding Provider
- OpenAI-compatible Chat LLM Provider（可选，用于重排序）
- uv

## 目录结构

```text
app/
  api/v1/routes/       HTTP 路由
  core/                配置、日志和异常
  db/                  SQLAlchemy 基础类和会话
  models/              知识库、文档、chunk、检索日志
  schemas/             Pydantic 请求和响应模型
  services/            业务编排和索引流程
  tasks/               Celery 异步任务
  repositories/        数据库读写
  providers/           Embedding、DocumentParser、VectorStore 和 KeywordSearch 抽象及实现
  utils/               UUID 和文本切片工具
migrations/            Alembic 迁移
eval/                  离线检索评测数据集和报告目录
tests/                 自动化测试
```

## 离线检索测评

离线测评从 Langfuse 的低分反馈生成候选 case，人工补充 `ground_truth` 后重新调用
`retrieve`，再用 RAGAS 计算 `context_precision` 和 `context_recall`。评测脚本独立运行，
不会创建数据库表，也不会嵌入 FastAPI。

先安装评测依赖：

```powershell
uv sync --extra eval
```

### 1. 导出 Langfuse 低分 case

脚本读取 `.env` 中的 `LANGFUSE_HOST`、`LANGFUSE_PUBLIC_KEY` 和 `LANGFUSE_SECRET_KEY`，
默认导出 `user_feedback` 分值低于 3 的记录。访问 Langfuse 时会关闭系统代理，适配本地
`localhost` 服务：

```powershell
uv run python scripts/export_eval_cases_from_langfuse.py `
  --max-score 3 `
  --days 30 `
  --kb-id <knowledge-base-id> `
  --output eval/datasets/imported_from_feedback.json
```

导出的 `ground_truth` 为空，需要人工根据知识库原文补齐。审核后可以合并到主集；合并按
`question` 去重，不会覆盖已有的 `ground_truth`：

```powershell
uv run python scripts/export_eval_cases_from_langfuse.py `
  --max-score 3 `
  --merge eval/datasets/ecommerce_retrieval.json `
  --output eval/datasets/ecommerce_retrieval.json
```

### 2. 运行 RAGAS 检索评测

`run_retrieval_eval.py` 对每个有效 case 重新调用 `POST /api/v1/rag/retrieve`，并固定使用
`user_id=eval_runner`。RAGAS 评测阶段复用 `.env` 中的 `MODEL_BASE_URL`、`MODEL_API_KEY`；
必要时可同时配置现有的 `LLM_MODEL` 作为评测模型名。

```powershell
uv run python scripts/run_retrieval_eval.py `
  --dataset eval/datasets/ecommerce_retrieval.json `
  --api-key rk_live_你的key `
  --profile balanced `
  --output eval/reports/balanced.json
```

没有 `ground_truth` 的 case 会自动跳过并在终端提示。可以对同一数据集分别运行多个 profile，
或者在调整词表、重排和 query 改写配置前后分别输出报告：

```powershell
uv run python scripts/run_retrieval_eval.py --dataset <dataset> --api-key <api-key> `
  --profile balanced --output eval/reports/before_synonyms.json
uv run python scripts/run_retrieval_eval.py --dataset <dataset> --api-key <api-key> `
  --profile balanced --output eval/reports/after_synonyms.json
```

`eval/datasets/_seed_ecommerce.json` 和 `eval/datasets/ecommerce_retrieval.json` 使用占位
`kb_id`，接入真实知识库前请替换为实际 ID。`eval/reports/` 下的评测报告已加入 `.gitignore`。

## 三个核心接口

所有业务接口使用统一响应格式：

```json
{
  "code": 0,
  "msg": "success",
  "data": {}
}
```

### 创建知识库

`POST /api/v1/knowledge-bases/create`

```json
{
  "name": "退款政策知识库",
  "description": "用于客服退款问题问答"
}
```

### 上传文档并异步索引

`POST /api/v1/documents/upload`

```json
{
  "kb_id": "<knowledge-base-id>",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。"
}
```

接口只负责校验知识库归属、保存原文并投递 Celery 任务，立即返回 `status=3`（PROCESSING）和 `chunk_count=0`。Worker 在后台完成文本切片、Embedding 生成，以及 pgvector 和 Elasticsearch 双写；状态值为 `1` 表示成功，`2` 表示失败。可以通过 `GET /api/v1/documents/{document_id}` 轮询状态。

上传接口同时支持 `multipart/form-data`。`.md` 和 `.txt` 继续使用 JSON 文本上传；`.docx` 和文字版 `.pdf` 使用文件上传，原文件会保存到 `DOCUMENT_STORAGE_PATH/{tenant_id}/{document_id}/`，由 Worker 在索引前解析：

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -H "Authorization: Bearer <api-key>" `
  -F "kb_id=<knowledge-base-id>" `
  -F "file=@sample.docx"
```

DOCX/PDF 解析依赖属于后端核心依赖，执行普通同步即可安装：

```powershell
uv sync
```

支持的文件扩展名为 `.md`、`.txt`、`.docx` 和 `.pdf`，默认大小限制为 `DOCUMENT_MAX_SIZE_MB=20`。扫描件 PDF 会进入 `FAILED`，错误信息会提示本期不支持 OCR。

### 生命周期运维

知识库支持详情、settings 整体替换和删除：

- `GET /api/v1/knowledge-bases/{kb_id}`
- `PATCH /api/v1/knowledge-bases/{kb_id}`
- `DELETE /api/v1/knowledge-bases/{kb_id}`

文档支持状态查询、删除和失败重试：

- `GET /api/v1/documents/{document_id}`
- `DELETE /api/v1/documents/{document_id}`
- `POST /api/v1/documents/{document_id}/reindex`

删除文档和知识库时，会清理原始上传文件以及该文档在 pgvector 与 Elasticsearch 中的旧检索数据。`reindex` 支持 SUCCESS 和 FAILED 文档：它保留原文档记录和内容，清理旧 chunk，将状态改为 PROCESSING，再投递新的 Celery 索引任务。PROCESSING 文档不能删除或 reindex；包含 PROCESSING 文档的知识库不能删除。

带有原始文件的文档可以通过 `POST /api/v1/documents/{document_id}/reindex?reparse=true` 从文件重新解析；纯 JSON 文本文档会忽略该参数。批量脚本也支持 `--reparse`。

失败文档可以通过接口重新索引：

```powershell
curl.exe -s -X POST "http://127.0.0.1:8000/api/v1/documents/<document_id>/reindex" `
  -H "Authorization: Bearer <api-key>"
```

### 切块策略升级后批量 reindex

Markdown 结构化切块只会影响新执行的索引任务。升级切块代码后，已有文档的旧 chunk 仍然存在于 pgvector 和 Elasticsearch 中，需要对目标知识库执行一次批量 reindex：

```powershell
uv run python scripts/reindex_knowledge_base.py --kb-id <knowledge-base-id>
```

脚本默认选择目标知识库下状态为 `SUCCESS(1)` 的文档；`--tenant-id` 可显式校验租户，不传时从知识库记录反查。也可以重复传入 `--document-id`，只重建指定文档：

```powershell
uv run python scripts/reindex_knowledge_base.py `
  --kb-id <knowledge-base-id> `
  --tenant-id <tenant-id> `
  --document-id <document-id-1> `
  --document-id <document-id-2>
```

每篇文档会先清理 pgvector 和 Elasticsearch 中的旧 chunk，再提交 `PROCESSING` 状态并投递 `index_document_task`。脚本会等待 Worker 将任务处理到终态：`success` 表示文档已经恢复为 `SUCCESS(1)`，`failed` 表示入队失败、Worker 索引失败或等待超时，`skipped` 表示文档已经处于 `PROCESSING`。单篇失败不会阻断其余文档，并会输出对应的 `document_id`。建议在低峰期执行。

执行批量 reindex 前先重启 Celery Worker，使 Worker 加载最新的 Markdown 切块代码和环境配置。

### Markdown 切块与 reindex 验收

API、PostgreSQL、Redis、Elasticsearch、Celery Worker 和后端启动后，可以运行真实端到端验收：

```powershell
uv run python scripts/acceptance/run_markdown_reindex_acceptance.py
```

多格式解析验收需要一个已有知识库 ID：

```powershell
uv run python scripts/verify_parser_12.py `
  --kb-id <knowledge-base-id> `
  --api-key rk_live_你的key
```

该脚本位于 `scripts/acceptance/`，仅用于真实环境验收，不作为日常业务脚本使用。脚本会创建带唯一标记的临时租户、API Key、知识库和文档，验证标题切块、表格切块、检索 metadata、批量 reindex、Worker 失败继续执行、PROCESSING 跳过、SUCCESS 强制 reindex 和纯文本文档。脚本无论成功或失败都会在 `finally` 中删除临时租户、API Key、知识库、文档、PostgreSQL chunks、检索日志和 Elasticsearch 文档，并进行残留检查。

### RAG 检索增强

`POST /api/v1/rag/retrieve`

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "profile": "balanced"
}
```

接口只返回结构化召回 chunk、引用来源、分数和检索元数据，不生成最终答案。业务方可以使用这些上下文调用自己的大模型或编排服务。

`profile` 未传时默认使用 `balanced`。`speed` 使用向量召回，`balanced` 使用混合召回，`quality` 在混合召回基础上启用重排和 query 改写，`custom` 保留请求中的 `retrieval_options`、`rerank_options` 和 `query_options`。可用 profile 和能力上限由租户套餐决定。

检索配置按字段确定优先级：先检查套餐是否允许所选 profile 和最终生效的功能，再由命名 profile 的预设覆盖请求中同名字段；预设未定义的字段仍可由请求提供。`custom` 不展开预设，显式请求值优先，未传的字段依次使用实际环境配置和代码默认值。前端与 curl 的请求优先级相同。即使某字段会被预设覆盖，请求仍须满足 API Schema 的类型和范围校验。环境中的 `RERANK_ENABLED` 与 `QUERY_REWRITE_ENABLED` 是缺省值，不是能否决 profile 或显式请求的全局禁用开关；套餐限制始终有效。

`custom` 下 hybrid 检索的 `top_k` 控制融合后送往后续阶段的候选数量（启用重排时最终返回数量由 `rerank_options.top_n` 控制）；若未传 `retrieval_options.vector_top_k` 或 `bm25_top_k`，分别回退到 `HYBRID_VECTOR_TOP_K` 和 `HYBRID_BM25_TOP_K`，不再跟随请求的 `top_k`。改写的 `query_options.enabled` 若显式传入，优先于 `strategy`：`true` 表示改写，`false` 表示不改写；未传 `enabled` 才由 `strategy` 决定或回退到配置。调试台自定义模式取消勾选重排或改写时会显式发送 `enabled:false`；curl 省略该字段则使用服务端缺省值。

使用 `custom` profile 时，可以通过请求体覆盖检索参数：

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "profile": "custom",
  "top_k": 20,
  "retrieval_options": {
    "mode": "hybrid",
    "vector_top_k": 20,
    "bm25_top_k": 20,
    "rrf_k": 60
  }
}
```

hybrid 模式先合并两路召回结果，再按 `1 / (rrf_k + rank)` 计算 RRF 分数。响应中的 `score` 是融合分数，同时保留 `vector_score`、`bm25_score`、两路排名和 `retrieval_source`。

通过 `rerank_options` 可以在向量召回后启用大模型重排序：

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "top_k": 20,
  "rerank_options": {
    "enabled": true,
    "top_n": 5
  }
}
```

启用后，接口会保留原始向量分数 `score`，并在 `retrieved_chunks` 中返回 `rerank_score`。重排序失败时会记录降级日志并返回原始向量排序结果。

重排序由 `LLMRerankProvider` 通过通用 `LLMProvider` 调用 OpenAI-compatible Chat Completions，不绑定具体模型厂商。相关配置包括 `LLM_PROVIDER`、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`、`RERANK_ENABLED`、`RERANK_TOP_N`、`RERANK_MAX_CANDIDATES` 和 `RERANK_CHUNK_MAX_CHARS`，完整示例见 `.env.example`。

### 提问语义优化

检索前可以按请求启用 LLM query 改写：

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "背调要问啥",
  "query_options": {
    "enabled": true,
    "strategy": "rewrite"
  }
}
```

改写使用已有的 `LLM_PROVIDER`、`LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL`，全局默认由 `QUERY_REWRITE_ENABLED=false` 关闭，超时由 `QUERY_REWRITE_TIMEOUT_MS` 控制。LLM 失败时会回退到用户原话，不影响检索；改写耗时单独记录在 `metadata.query_processing.rewrite_latency_ms`，不计入检索 `latency_ms`。

知识库的 `settings` 字段可配置词表扩展。词表始终在改写之后执行，即使没有启用 LLM 改写也会生效：

```json
{
  "synonyms": [
    {"terms": ["背调"], "expand": ["背景调查", "标准问题清单"]}
  ],
  "rewrite_hint": "补充给 LLM 改写 Prompt 的领域说明"
}
```

使用运维脚本会**整文件覆盖**指定知识库的 `settings` 字段，并校验知识库存在：

```powershell
uv run python scripts/update_kb_settings.py `
  --kb_id <kb_id> `
  --settings-file examples/kb_settings.example.json
```

脚本示例文件见 `examples/kb_settings.example.json`。

### Celery 任务运维

以下三种操作都用于异步索引运维，但处理层级不同：

| 操作                                      | 适用场景                         | 是否清理该文档旧检索数据 | 任务消息   |
| ----------------------------------------- | -------------------------------- | ------------------------ | ---------- |
| `POST /documents/{document_id}/reindex` | 文档状态为 SUCCESS 或 FAILED    | 是                       | 新建任务   |
| `restore_celery_task.py restore`        | 原消息仍在 Redis 未确认列表      | 否                       | 恢复原消息 |
| `index_document_task.delay(...)`        | 文档为 PROCESSING 且原消息已丢失 | 否                       | 新建任务   |

#### 通过 reindex 接口重试失败文档

文档状态为 `FAILED(2)` 时，优先使用业务接口重新索引。该接口会清理该文档在 PostgreSQL 和 Elasticsearch 中的旧检索数据，将状态改为 `PROCESSING(3)`，然后新建 Celery 索引任务：

```powershell
curl.exe -sS -X POST "http://127.0.0.1:8000/api/v1/documents/<document_id>/reindex" -H "Authorization: Bearer <api-key>"
```

接口返回后由 Worker 异步执行，完成后可通过文档状态接口确认结果。不要对 FAILED 文档直接调用 `.delay(...)`，也不要重新上传同一文档。

#### 恢复未确认消息

Celery Redis Broker 中仍处于未确认状态的任务可以先查看，再按 `delivery_tag` 精确恢复：

```powershell
# 查看当前所有未确认任务
uv run python scripts/restore_celery_task.py list

# 预览指定任务
uv run python scripts/restore_celery_task.py inspect `
  --delivery-tag <delivery_tag>

# 校验任务类型和文档 ID 后恢复原消息
uv run python scripts/restore_celery_task.py restore `
  --delivery-tag <delivery_tag> `
  --expected-task app.tasks.indexing.index_document_task `
  --expected-document-id <document_id> `
  --yes
```

恢复前应先停止正在运行的 Worker，避免原 Worker 与恢复后的消息并发处理同一文档。该脚本只操作 Broker 中的原始消息，不创建新任务、不修改数据库，也不清理 Result Backend。

#### 直接新建任务

如果任务已经被 Worker 确认、从 Redis 的未确认列表中消失，或者已经无法获得原来的 `delivery_tag`，则不能再使用上面的恢复脚本。确认原任务不会继续执行后，可以根据已有的 `document_id` 新建一条索引任务：

```powershell
# 先启动已包含最新代码的 Worker；Windows 本机建议使用 solo
uv run celery -A app.celery_app worker --loglevel=info --pool=solo

# 在另一个 PowerShell 窗口中按已有 document_id 重新投递
uv run python -c "from app.tasks.indexing import index_document_task; print(index_document_task.delay('<document_id>').id)"
```

该命令只会向 Celery Broker 新建一条 `index_document_task` 消息，不会重新上传文档，也不会新建 `documents` 记录。任务会读取已有文档，完成切块、Embedding、pgvector 和 Elasticsearch 写入。重新投递前应确认文档仍为 `PROCESSING`，并确保原任务没有正在运行或等待可见性超时；否则可能出现同一文档被并发或重复索引。若文档已经是 `FAILED`，应优先调用上面的 `reindex` 接口，由业务流程清理旧检索数据并重新设置状态。

#### 使用建议

- 文档状态为 `FAILED(2)`：使用 `POST /documents/{document_id}/reindex`。它会清理该文档旧的 PostgreSQL chunk 和 Elasticsearch 文档，然后将状态改为 `PROCESSING(3)` 并创建新任务。
- 文档状态为 `PROCESSING(3)`，且 `restore_celery_task.py list` 能看到对应未确认消息：使用 `inspect` 校验后执行 `restore`，不要再次调用 `.delay(...)`。
- 文档状态为 `PROCESSING(3)`，原消息已经确认或丢失，且确认没有旧 Worker 正在执行：才使用 `.delay(...)` 新建任务。
- 不要对 `SUCCESS(1)` 文档直接投递任务；任务入口会因状态不是 PROCESSING 而跳过。
- 三种操作都不要重新上传文档；它们使用已有的 `document_id` 和 `documents.content`。

## 多用户鉴权

默认开启多用户鉴权。客户端先创建租户和 API Key，再在请求头中携带 `Authorization: Bearer <api-key>` 调用业务接口：

```powershell
uv run python scripts/create_tenant.py --id tenant_demo --name "演示租户"
uv run python scripts/create_api_key.py --tenant-id tenant_demo --name "本地开发"
```

新建租户默认使用 `free` 套餐。可以通过运维脚本切换套餐：

```powershell
uv run python scripts/update_tenant_plan.py --tenant-id tenant_demo --plan pro
```

API Key 只在创建命令中明文输出一次，数据库只保存 SHA-256 hash。`GET /api/v1/auth/me` 可用于验证当前 Key 和租户信息。设置 `AUTH_ENABLED=false` 时跳过 Key 校验并固定使用 `tenant_demo`，便于本地调试。

新租户的完整开通、套餐限制、profile 用法和常见错误码见 [租户接入与套餐使用](docs/tenant-onboarding.md)。

## 本地启动

1. 安装 Python 3.11 和 uv。
2. 创建环境文件：

   ```powershell
   Copy-Item .env.example .env
   ```
3. 在 `.env` 中填写 `MODEL_API_KEY`，并确认 `EMBEDDING_DIMENSIONS` 与模型输出维度一致。当前示例使用 DashScope 的 `qwen3.7-text-embedding`，显式请求 `1536` 维输出。
4. 安装依赖：

   ```powershell
   uv sync
   ```
5. 启动 PostgreSQL、pgvector 和 Elasticsearch。当前 Docker Compose 只负责基础设施，不构建或运行 Python/API 容器：

   ```powershell
   docker compose up -d
   ```

   Compose 默认将宿主机端口映射为 PostgreSQL `15433`、Redis `16379`、Elasticsearch `19200`、Elasticsearch transport `19300`、Langfuse PostgreSQL `15432` 和 Langfuse `13000`；容器内部端口保持不变。
6. 执行数据库迁移：

   ```powershell
   uv run alembic upgrade head
   ```
7. 启动 API：

   ```powershell
   uv run uvicorn app.main:app --reload --loop app.core.event_loop:selector_event_loop_factory
   ```
8. 启动 Celery Worker：

   ```powershell
   uv run celery -A app.celery_app worker --loglevel=info
   ```

当前开发模式下，Python、uv 和 FastAPI 都由本机环境管理；Docker 运行 PostgreSQL + pgvector、Elasticsearch 和 Redis。Elasticsearch 索引 mapping 使用 `analysis-ik` 提供的 `ik_max_word` 与 `ik_smart` 分词器。

### Windows 下 psycopg 事件循环兼容

Windows 使用 psycopg 异步连接时，Uvicorn 默认的单进程启动方式会创建 `ProactorEventLoop`，访问需要 PostgreSQL 的接口时可能出现：

```text
Psycopg cannot use the 'ProactorEventLoop' to run in async mode
```

开发环境推荐使用上面的 `--reload --loop app.core.event_loop:selector_event_loop_factory` 启动命令。若不使用 `--reload`，则必须显式指定：

```powershell
uv run uvicorn app.main:app --loop app.core.event_loop:selector_event_loop_factory
```

提交代码后如果没有重启后端，已运行的进程不会加载新的代码或事件循环配置；`--reload` 模式则会由重载子进程使用兼容的 Selector loop。

## 测试和代码检查

```powershell
uv run pytest
uv run ruff check .
```

测试通过依赖替身 Provider，不需要调用真实 Embedding 服务；真实接口调用仍需要配置 `MODEL_API_KEY` 和兼容的模型地址。

### 真实依赖生命周期验收

API、Docker 基础设施和 Celery Worker 均启动后，可以执行真实 API + PostgreSQL + Redis/Celery + pgvector + Elasticsearch 验收：

```powershell
uv run python scripts/run_lifecycle_integration.py
```

脚本会创建带唯一标记的临时租户、API Key、知识库和文档，验证文档/知识库生命周期、异步索引、检索清理和跨租户隔离，结束时通过 API 及数据库/Elasticsearch 复查并删除所有测试数据。已有 Worker 会被复用；没有 Worker 时脚本会临时启动 `--pool=solo` Worker，并在结束时停止它。

## 日志和错误码

应用启动时会初始化全局日志管线：

- 控制台输出带颜色，文件输出使用统一格式：时间、级别、模块/函数/行号、请求 ID 和消息。
- 普通日志写入 `logs/app.log`，`ERROR` 及以上日志单独写入 `logs/error.log`。
- 两个文件都支持大小和时间双重轮转，默认单文件 `10 MB`、保留 `5` 个备份。
- `X-Request-ID` 会沿请求链路透传；未提供时由服务自动生成。
- 日志队列由后台监听线程写入文件，业务协程不会同步执行文件 I/O。

可通过环境变量调整日志配置，例如：

```dotenv
LOG_LEVEL=INFO
LOG_DIR=logs
LOG_MAX_BYTES=10485760
LOG_BACKUP_COUNT=5
LOG_ROTATION_WHEN=midnight
LOG_ROTATION_INTERVAL=1
LOG_CONSOLE_COLOR=true
LOG_PAYLOAD_MAX_LENGTH=2000
```

错误响应统一为：

```json
{
  "code": 0,
  "msg": "success",
  "data": {}
}
```

错误码按范围划分：`10000` 段为通用客户端错误，`20000` 段为接口/HTTP，`30000` 段为数据库/存储，`40000` 段为 LLM，`50000` 段为系统服务异常。业务代码可以使用统一异常和快速抛错工具：

```python
from app.core.error_codes import ErrorCode
from app.core.exceptions import raise_app_error

raise_app_error(
    ErrorCode.PARAM_ERROR,
    data={"field": "query"},
    context={"operation": "retrieve"},
)
```

接口和模型调用日志也可以复用工具：

```python
from app.core.logging import log_api_call, log_llm_call


@log_api_call
async def handle_request(request):
    return {"ok": True}


result = await log_llm_call(
    lambda: client.embeddings.create(model="embedding-model", input="hello"),
    model="embedding-model",
    prompt="hello",
)
```

## 数据库迁移

```powershell
uv run alembic upgrade head
uv run alembic downgrade -1
```

当前数据库向量列使用 1536 维，与项目当前 `.env` 中的 Embedding 模型输出保持一致。`EMBEDDING_DIMENSIONS` 必须与实际模型输出维度一致；若更换 Embedding 模型或维度，应同步调整配置并新增迁移，不要直接修改已经应用的历史迁移。

如果出现 `expected 1536 dimensions, not 1024`，说明 Embedding 请求没有按配置返回 1536 维，或数据库仍未升级到当前结构。确认 `.env` 中为 `EMBEDDING_DIMENSIONS=1536`，重启 API，并执行：

```powershell
uv run alembic upgrade head
```

## 后续扩展方向

- 接入 Elasticsearch / OpenSearch，实现 BM25 混合检索。
- 增加 Milvus / Qdrant `VectorStore` 实现。
- 增加权限 ACL 过滤。
- 增加评测集和反馈闭环。
- 增加管理后台。

## 当前边界

当前仍不包含 OCR、OpenSearch、复杂权限系统、评测系统、A/B 测试和多轮会话记忆。
