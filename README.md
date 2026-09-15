# RAG Center

RAG Center 是一个面向业务方提供统一 RAG 能力的后端中台骨架。当前实现知识库创建、文档同步索引，以及 pgvector 向量检索、Elasticsearch BM25 检索和 RRF 混合召回，保留各类 Provider 的扩展边界。

## 技术栈

- Python 3.11
- FastAPI
- PostgreSQL 16 + pgvector
- Elasticsearch 8.17 + analysis-ik
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
  services/            业务编排和同步索引流程
  repositories/        数据库读写
  providers/           Embedding、DocumentParser、VectorStore 和 KeywordSearch 抽象及实现
  utils/               UUID 和文本切片工具
migrations/            Alembic 迁移
tests/                 自动化测试
```

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

### 上传文档并同步索引

`POST /api/v1/documents/upload`

```json
{
  "kb_id": "<knowledge-base-id>",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。"
}
```

接口会同步完成文本切片、Embedding 生成，以及 pgvector 和 Elasticsearch 双写。返回状态值为 `1` 表示成功，`2` 表示失败，`3` 预留给后续异步索引。

### RAG 检索增强

`POST /api/v1/rag/retrieve`

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？"
}
```

接口只返回结构化召回 chunk、引用来源、分数和检索元数据，不生成最终答案。业务方可以使用这些上下文调用自己的大模型或编排服务。

默认使用 `RETRIEVAL_MODE=vector` 保持纯向量检索。也可以通过请求体覆盖为混合召回：

```json
{
  "kb_id": "<knowledge-base-id>",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
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

## 多用户鉴权

默认开启多用户鉴权。客户端先创建租户和 API Key，再在请求头中携带 `Authorization: Bearer <api-key>` 调用业务接口：

```powershell
uv run python scripts/create_tenant.py --id tenant_demo --name "演示租户"
uv run python scripts/create_api_key.py --tenant-id tenant_demo --name "本地开发"
```

API Key 只在创建命令中明文输出一次，数据库只保存 SHA-256 hash。`GET /api/v1/auth/me` 可用于验证当前 Key 和租户信息。设置 `AUTH_ENABLED=false` 时跳过 Key 校验并固定使用 `tenant_demo`，便于本地调试。

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
6. 执行数据库迁移：

   ```powershell
   uv run alembic upgrade head
   ```
7. 启动 API：

   ```powershell
   uv run uvicorn app.main:app --reload
   ```

当前开发模式下，Python、uv 和 FastAPI 都由本机环境管理；Docker 运行 PostgreSQL + pgvector 和 Elasticsearch。Elasticsearch 索引 mapping 使用 `analysis-ik` 提供的 `ik_max_word` 与 `ik_smart` 分词器。

## 测试和代码检查

```powershell
uv run pytest
uv run ruff check .
```

测试通过依赖替身 Provider，不需要调用真实 Embedding 服务；真实接口调用仍需要配置 `MODEL_API_KEY` 和兼容的模型地址。

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
- 将 `IndexingService.index_document()` 迁移到 BackgroundTasks、Celery 或消息队列。
- 增加 Milvus / Qdrant `VectorStore` 实现。
- 增加权限 ACL 过滤。
- 增加评测集和反馈闭环。
- 增加管理后台。

## 当前边界

当前仍不包含多种文件格式解析、Elasticsearch / OpenSearch、复杂权限系统、评测系统、A/B 测试、多轮会话记忆以及 Redis / Celery 异步任务。
