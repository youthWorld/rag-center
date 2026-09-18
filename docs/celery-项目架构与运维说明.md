# Celery 项目架构与运维说明

本文档说明当前 RAG Center 项目中的 Celery 组成、文档索引任务流程、配置加载方式、失败重试机制，以及开发时需要注意的 Worker 重启问题。

## 1. Celery 的语法和使用

### 1.1 创建 Celery 应用

最基本的 Celery 应用由一个 `Celery` 实例组成：

```python
from celery import Celery

celery_app = Celery("demo")
```

在实际项目中，还需要配置 Broker。Broker 是任务消息的中转站，常见实现包括 Redis 和 RabbitMQ：

```python
celery_app = Celery(
    "demo",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
```

当前项目将这些配置放在 `app/celery_app.py` 和 `.env` 中，而不是直接写死在构造函数里。

### 1.2 声明任务

使用 `@celery_app.task` 装饰器可以把普通 Python 函数注册成 Celery 任务：

```python
@celery_app.task
def add(left: int, right: int) -> int:
    return left + right
```

任务必须能够被 Worker 导入。当前项目通过 `imports=("app.tasks.indexing",)` 导入任务模块，并注册出：

```text
app.tasks.indexing.index_document_task
```

任务名称很重要，因为 Redis 中的消息通常只保存任务名称和参数，Worker 会根据任务名称找到对应函数。

### 1.3 投递任务：`delay()`

最常用的投递方式是 `.delay()`：

```python
result = add.delay(1, 2)
```

它等价于：

```python
result = add.apply_async(args=[1, 2])
```

`.delay()` 会立即把任务消息发送到 Broker，并返回一个 `AsyncResult` 对象。它不会等待任务真正执行完成。

当前项目的上传代码就是：

```python
index_document_task.delay(document.id)
```

这里的 `document.id` 会被序列化成任务参数发送到 Redis。

### 1.4 投递任务：`apply_async()`

需要指定更多投递选项时使用 `apply_async()`：

```python
add.apply_async(
    args=[1, 2],
    countdown=10,
)
```

常用参数包括：

| 参数 | 作用 |
| --- | --- |
| `args` | 位置参数列表 |
| `kwargs` | 关键字参数字典 |
| `countdown` | 延迟多少秒后执行 |
| `eta` | 指定执行时间 |
| `expires` | 任务过期时间 |
| `queue` | 指定任务队列 |
| `priority` | 指定消息优先级，具体支持情况取决于 Broker |

例如：

```python
index_document_task.apply_async(
    args=[document.id],
    countdown=5,
)
```

当前项目没有自定义队列，索引任务使用 Celery 默认队列。

### 1.5 `AsyncResult`：查看任务结果

任务投递后，Celery 会返回一个 `AsyncResult`：

```python
result = add.delay(1, 2)

task_id = result.id
status = result.status
finished = result.ready()
```

如果配置了 Result Backend，可以读取结果：

```python
value = result.get(timeout=30)
```

常见状态包括：

```text
PENDING    尚未开始或找不到结果
STARTED    已开始执行
SUCCESS    执行成功
FAILURE    执行失败
RETRY      等待重试
```

不要在另一个 Celery 任务内部随意调用 `result.get()`，也不要在 FastAPI 上传接口中同步等待它，否则会失去异步处理的意义。当前项目以 PostgreSQL 中的文档状态为业务事实来源，前端通过文档查询接口轮询状态。

### 1.6 绑定任务和重试

如果任务需要访问任务实例本身，需要使用 `bind=True`：

```python
@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
def process_document(self, document_id: str) -> dict:
    try:
        return do_work(document_id)
    except TimeoutError as exc:
        raise self.retry(exc=exc)
```

绑定后可以访问：

```python
self.request.id       # 当前任务 ID
self.request.retries  # 已重试次数
self.retry(...)       # 重新投递当前任务
```

当前项目的 `index_document_task` 使用了这种方式，只对网络超时、连接错误、限流等临时错误进行重试。

### 1.7 启动 Worker

通用启动语法是：

```powershell
celery -A <python模块>:<Celery实例> worker --loglevel=info
```

当前项目的启动命令是：

```powershell
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

其中：

- `-A app.celery_app`：告诉 Celery 从 `app.celery_app` 查找应用对象。
- `worker`：启动任务消费者。
- `--loglevel=info`：显示普通运行日志。
- `--pool=solo`：使用单进程执行，适合当前 Windows 开发环境。

### 1.8 查看 Worker

常用运维命令：

```powershell
# 检查 Worker 是否在线
uv run celery -A app.celery_app inspect ping

# 查看 Worker 启动时间、进程和任务统计
uv run celery -A app.celery_app inspect stats

# 查看正在执行的任务
uv run celery -A app.celery_app inspect active

# 查看 Worker 已注册的任务
uv run celery -A app.celery_app inspect registered
```

### 1.9 当前项目的最小使用示例

当前项目投递索引任务的完整调用可以简化为：

```python
from app.tasks.indexing import index_document_task

async_result = index_document_task.delay("document-id")
print(async_result.id)
```

执行这段代码前，必须确保：

1. Redis Broker 正常运行。
2. Worker 已经启动。
3. Worker 注册了 `app.tasks.indexing.index_document_task`。
4. PostgreSQL 中存在对应的文档记录。
5. 文档状态符合任务入口的要求，当前项目要求为 `PROCESSING(3)`。

## 2. Celery 在项目中的作用

Celery 用于把文档索引从 HTTP 请求中拆分出来，交给后台 Worker 异步执行。

```text
前端
  |
  v
FastAPI API
  |
  +-- PostgreSQL：保存文档原文和文档状态
  |
  +-- Redis DB 0：保存待执行的 Celery 任务
                              |
                              v
                       Celery Worker
                              |
              +---------------+----------------+
              |               |                |
              v               v                v
          文档切块        Embedding API     pgvector / Elasticsearch
                              |
                              v
                       PostgreSQL 更新状态
```

当前开发环境中：

- PostgreSQL、Redis、Elasticsearch 由 Docker Compose 提供。
- FastAPI 和 Celery Worker 由本机的 `uv` 启动。
- `docker-compose.yml` 当前没有定义 API 或 Celery Worker 容器。

## 3. Celery 的三个核心角色

### 3.1 FastAPI：任务生产者

文档上传接口位于：

```text
app/api/v1/routes/documents.py
```

实际业务编排位于：

```text
app/services/document_service.py
```

上传时，`DocumentService.upload()` 依次执行：

1. 校验知识库和租户权限。
2. 创建 `documents` 数据库记录。
3. 将文档状态设置为 `PROCESSING(3)`。
4. 提交数据库事务，确保 Worker 能读取到文档。
5. 调用 `index_document_task.delay(document.id)` 投递任务。
6. 立即返回 `status=3` 和 `chunk_count=0`。

核心代码位于：

```python
document = await self.indexing_service.create_document_record(
    tenant_id,
    request,
)
index_document_task.delay(document.id)
```

Celery 任务消息中主要包含任务名称和 `document_id`，不会包含 Python 源代码，也不会包含上传时的代码版本。

### 3.2 Redis：消息代理 Broker

当前配置为：

```env
CELERY_BROKER_URL=redis://localhost:6379/0
```

Redis 0 号数据库用于保存待执行的任务消息。Redis 只负责保存和分发消息，不负责执行索引逻辑。

### 3.3 Celery Worker：任务消费者

Worker 持续监听 Redis，取到任务后调用对应的 Python 函数。

Windows 本地开发建议使用：

```powershell
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

`--pool=solo` 表示使用单进程执行任务，适合当前 Windows 本地环境和项目中的异步数据库调用。

## 4. Celery 应用初始化

Celery 应用位于：

```text
app/celery_app.py
```

初始化代码的主要作用如下：

```python
from app.core.config import settings

celery_app = Celery("rag_center")
celery_app.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    imports=("app.tasks.indexing",),
)
```

配置说明：

| 配置 | 当前作用 |
| --- | --- |
| `broker_url` | 指定 Redis 任务队列，当前为 Redis DB 0 |
| `result_backend` | 保存 Celery 任务结果，当前为 Redis DB 1 |
| `task_serializer` | 使用 JSON 序列化任务参数 |
| `accept_content` | 只接受 JSON 格式的任务消息 |
| `timezone` | Celery 使用 UTC 时间 |
| `task_track_started` | 允许记录任务开始状态 |
| `imports` | 启动时导入并注册索引任务模块 |

## 5. Worker 启动时会加载什么

执行下面的命令时：

```powershell
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

Worker 会启动一个长期运行的 Python 进程，并大致执行以下过程：

1. 导入 `app.celery_app`。
2. 导入 `app.core.config.settings`。
3. 读取 `.env` 并创建全局 `Settings` 对象。
4. 根据 `settings` 配置 Redis Broker 和 Result Backend。
5. 导入 `app.tasks.indexing`。
6. 注册 `app.tasks.indexing.index_document_task`。
7. 开始监听 Redis 队列。

Python 模块导入后会保留在当前进程内存中。修改磁盘上的 `.py` 文件，不会自动替换已经加载的函数和类。

`.env` 也不会在每个任务开始时重新读取。当前项目中的：

```python
settings = get_settings()
```

会在模块首次导入时创建全局配置对象，`get_settings()` 还使用了 `lru_cache`。

因此，修改以下内容后都应该重启 Worker：

- `app/tasks/*.py`
- `app/services/indexing_service.py`
- `app/utils/markdown_splitter.py`
- `app/core/config.py`
- `.env`
- Embedding、数据库、Elasticsearch 等索引相关配置

## 6. 一次文档上传的完整流程

### 6.1 API 侧

```text
前端上传文件
    |
    v
FastAPI 接收 title 和 content
    |
    v
创建 documents 记录，状态为 PROCESSING
    |
    v
提交数据库事务
    |
    v
向 Redis 投递 document_id
    |
    v
API 立即返回 status=PROCESSING
```

API 不会等待文档切块、Embedding 或索引写入完成。

### 6.2 Worker 侧

Worker 执行 `app/tasks/indexing.py` 中的：

```python
index_document_task(document_id)
```

任务内部执行：

1. 创建异步事件循环。
2. 创建 Worker 自己的数据库 Session。
3. 使用当前进程中的 `settings` 创建 `IndexingService`。
4. 从 PostgreSQL 查询 `document_id` 对应的文档。
5. 只有文档状态为 `PROCESSING` 时才继续处理。
6. 解析文档内容。
7. 选择普通文本切块器或 Markdown 结构化切块器。
8. 调用 Embedding Provider。
9. 写入 PostgreSQL 的 `chunks` 表。
10. 写入 Elasticsearch 的 `rag_chunks` 索引。
11. 将文档状态更新为 `SUCCESS`。
12. 关闭 Elasticsearch 客户端。

索引主流程位于：

```text
app/tasks/indexing.py
app/services/indexing_service.py
```

## 7. 当前 Markdown 索引流程

当前 `IndexingService.index_document()` 先调用文档解析器：

```python
parsed_content = self.document_parser.parse(
    document.content,
    source_type=document.source_type or "text",
)
```

`PlainTextDocumentParser` 返回字符串是正常行为。Markdown 原文应该以字符串形式交给 Markdown 切块器，标题和表格是在后续切块阶段识别的。

当前 Markdown 类型判定逻辑位于：

```python
def _is_markdown_document(document: Document) -> bool:
    source_type = (document.source_type or "").lower()
    if source_type in {"markdown", "md"}:
        return True
    title = (document.title or "").lower()
    return title.endswith((".md", ".markdown"))
```

因此，当前即使数据库中的 `source_type` 仍然是 `text`，只要标题以 `.md` 或 `.markdown` 结尾，也会进入 `MarkdownStructuredSplitter`。

Markdown 结构化切块会写入以下 metadata：

```json
{
  "chunk_index": 0,
  "source_type": "text",
  "heading_path": "一级标题/二级标题",
  "chunk_type": "section",
  "table_part": null
}
```

表格 chunk 的 `chunk_type` 为 `table`，大表拆分时会写入 `table_part`。

## 8. 为什么代码修改后必须重启 Worker

假设 Worker 在 09:00 启动：

```text
09:00 Worker 导入旧版 indexing_service.py
09:35 修改磁盘上的 indexing_service.py
09:50 创建新的上传任务
09:50 旧 Worker 执行新的任务
```

任务虽然是在 09:35 之后创建的，但 Worker 内存中的函数仍然是旧版本。

任务消息中只有：

```json
{
  "task": "app.tasks.indexing.index_document_task",
  "args": ["document-id"]
}
```

消息不会携带：

- 任务创建时的 Python 源代码；
- 代码分支或 Git commit；
- 最新 `.env` 内容；
- 最新的切块器对象。

另外，`uvicorn --reload` 只会重载 FastAPI 进程，不会重启 Celery Worker。因此可能出现：

```text
FastAPI：新代码
Celery Worker：旧代码
```

本次 Markdown 问题正是这种情况：第一批任务创建于 09:50:26，而当时实际运行的旧 Worker 使用了旧版 `TextSplitter`。之后 Worker 重启，新上传的文档才产生了 `heading_path`。

## 9. 任务重试机制

任务定义为：

```python
@celery_app.task(bind=True, max_retries=2, default_retry_delay=10)
```

含义是：

- 初次执行失败后，最多再尝试 2 次；
- 总执行次数最多为 3 次；
- 两次重试之间默认等待 10 秒。

项目只对部分临时错误重试，包括：

- 网络连接错误；
- 超时；
- `API_TIMEOUT`；
- `API_RATE_LIMIT`；
- `LLM_TIMEOUT`；
- `LLM_RATE_LIMIT`；
- 类名中包含 `timeout`、`connection`、`transport` 或 `temporarily` 的异常。

非临时错误通常不会重试，例如文档内容为空、参数错误或解析逻辑错误。

重试前，项目会把文档状态重新设置为 `PROCESSING`。最终失败后，文档状态为 `FAILED`，错误信息写入 `error_message`。

## 10. 文档状态和查询方式

状态定义在 `app/models/document.py`：

```python
SUCCESS = 1
FAILED = 2
PROCESSING = 3
```

前端不依赖 Celery Result Backend 判断最终业务状态，而是调用：

```text
GET /api/v1/documents/{document_id}
```

该接口从 PostgreSQL 查询：

- 文档状态；
- 错误信息；
- chunk 数量。

因此，PostgreSQL 是当前文档索引状态的业务事实来源。

## 11. Result Backend 的作用

当前配置为：

```env
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

Redis 1 号数据库用于保存 Celery 任务结果，例如任务返回的：

```python
{
    "document_id": document_id,
    "status": int(DocumentStatus.SUCCESS),
    "chunk_count": result,
}
```

当前项目没有把这个结果作为前端主要状态来源。它主要用于 Celery 自身的任务状态追踪和运维查看。

Redis 数据库用途如下：

| Redis 数据库 | 用途 |
| --- | --- |
| DB 0 | Celery Broker，保存待执行任务 |
| DB 1 | Celery Result Backend，保存任务结果 |

Redis 不保存文档原文，也不保存最终 chunk 数据。

## 12. Reindex 和失败恢复

### 12.1 业务层 reindex

接口：

```text
POST /api/v1/documents/{document_id}/reindex
```

只允许 `FAILED` 文档执行。它会：

1. 删除 PostgreSQL 中的旧 chunk。
2. 删除 Elasticsearch 中的旧 chunk。
3. 将文档状态改为 `PROCESSING`。
4. 创建新的 Celery 任务。

### 12.2 直接重新投递任务

如果文档仍然是 `PROCESSING`，但原任务已经丢失，可以手动创建任务：

```powershell
uv run python -c "from app.tasks.indexing import index_document_task; print(index_document_task.delay('<document_id>').id)"
```

执行前必须确认原任务没有仍在运行，否则可能产生重复索引。

### 12.3 查看未确认消息

项目提供了 Redis Celery 任务运维脚本：

```powershell
uv run python scripts/restore_celery_task.py list
```

脚本还支持查看和恢复指定的未确认消息，具体参数见 `README.md` 的 Celery 任务运维章节。

## 13. Worker 运维命令

启动 Worker：

```powershell
uv run celery -A app.celery_app worker --loglevel=info --pool=solo
```

检查 Worker 是否在线：

```powershell
uv run celery -A app.celery_app inspect ping
```

查看 Worker 启动信息、进程和已执行任务：

```powershell
uv run celery -A app.celery_app inspect stats
```

查看当前正在执行的任务：

```powershell
uv run celery -A app.celery_app inspect active
```

查看 Worker 已注册的任务：

```powershell
uv run celery -A app.celery_app inspect registered
```

修改索引代码或相关配置后，先停止旧 Worker，再重新执行启动命令：

```text
Ctrl+C
重新执行 celery worker 启动命令
```

如果机器上启动了多个 Worker，必须确认所有 Worker 都使用同一份代码和配置，否则同一队列可能被不同版本的 Worker 消费。

## 14. 日常排查顺序

遇到文档一直处于 `PROCESSING`、索引数量不正确或 metadata 不符合预期时，建议按以下顺序检查：

1. 检查 Redis、PostgreSQL 和 Elasticsearch 是否正常运行。
2. 执行 `inspect ping` 确认 Worker 在线。
3. 执行 `inspect stats` 查看 Worker 的启动时间和已执行任务数量。
4. 确认 Worker 启动时间晚于最近一次代码和 `.env` 修改时间。
5. 执行 `inspect registered` 确认任务已经注册。
6. 检查 PostgreSQL 中的 `documents.status` 和 `error_message`。
7. 检查 PostgreSQL `chunks.metadata`。
8. 检查 Elasticsearch 中相同 `document_id` 的数据。
9. 确认没有多个不同目录启动的 Worker。

最重要的判断原则是：

> 任务创建时间不能代表 Worker 使用的代码版本。真正决定执行逻辑的是消费该任务的 Worker 进程启动时加载的代码和配置。
