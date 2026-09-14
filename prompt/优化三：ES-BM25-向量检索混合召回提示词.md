# ES BM25 + 向量检索混合召回提示词

## 使用场景

这份提示词用于在当前 RAG 中台工程基础上，增加 Elasticsearch BM25 关键词检索，并与现有 pgvector 向量检索做混合召回。

目标是让检索能力同时覆盖：

- 向量检索擅长的语义相似问题
- BM25 擅长的关键词、专有名词、错误码、条款名、型号等精确匹配问题

采用 RRF 做结果融合，不直接把 BM25 分数和向量分数加权相加。

## 一句话目标

请为当前 RAG 中台增加"pgvector 向量检索 + Elasticsearch BM25 检索 + RRF 融合排序"的混合检索能力，在不生成最终答案的前提下，返回结构化 chunks、来源分数、排名和融合分数，并保留后续接入 rerank 的扩展空间。

## 当前链路

当前 RAG 检索链路是：

```Plain
用户 query
  -> query embedding
  -> pgvector 向量召回 top_k chunks
  -> 可选 LLM rerank
  -> 返回 retrieved_chunks
```

需要升级为：

```Plain
用户 query
  -> query embedding
  -> pgvector 向量召回 vector_top_k
  -> Elasticsearch BM25 关键词召回 bm25_top_k
  -> 按 chunk_id 合并两路结果
  -> 使用 RRF 计算 fused_score
  -> 按 fused_score 排序得到候选 chunks
  -> 可选 LLM rerank
  -> 返回最终 chunks
```

## 为什么使用 RRF

不要第一版直接使用：

```Plain
final_score = vector_score * 0.6 + bm25_score * 0.4
```

原因：

- 向量分数和 BM25 分数不是同一量纲
- BM25 分数受词频、字段长度、语料分布影响
- 向量相似度分数和 BM25 分数直接相加容易不稳定
- 需要额外做归一化，第一版复杂度较高

使用 RRF：

```Plain
rrf_score = 1 / (rrf_k + rank)
```

如果一个 chunk 同时被向量检索和 BM25 检索召回：

```Plain
fused_score =
  1 / (rrf_k + vector_rank)
  + 1 / (rrf_k + bm25_rank)
```

`rrf_k` 默认建议为 `60`。

RRF 的优点：

- 不依赖不同检索器的原始分数尺度
- 实现简单
- 工程上稳定
- 适合作为第一版混合检索默认融合策略

## 新增技术栈

在现有技术栈基础上新增Elasticsearch

用途：

- 保存 chunk 的文本字段
- 基于 `title`、`content`、`tenant_id`、`kb_id` 做 BM25 检索
- 与 pgvector 召回结果做混合融合

## 新增代码结构

请在现有工程中新增：

```Plain
app/
  providers/
    keyword_search/
      base.py
      elasticsearch.py
  services/
    hybrid_search_service.py
  schemas/
    hybrid_search.py
```

说明：

- `keyword_search/base.py` 定义统一关键词检索接口
- `keyword_search/elasticsearch.py` 实现 Elasticsearch BM25 检索
- `hybrid_search_service.py` 负责合并向量召回和 BM25 召回，并执行 RRF 融合
- `schemas/hybrid_search.py` 定义混合检索内部结构

不要让 `RagService` 直接写 Elasticsearch 查询细节。

## KeywordSearchProvider 抽象

新增统一关键词检索抽象：

```Python
from abc import ABC, abstractmethod


class KeywordSearchProvider(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: list[dict]) -> None:
        pass

    @abstractmethod
    async def keyword_search(
        self,
        *,
        query: str,
        tenant_id: str,
        kb_id: str,
        top_k: int = 20,
    ) -> list[dict]:
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> None:
        pass
```

第一版实现：

```Plain
ElasticsearchKeywordSearchProvider
```

后续可以扩展：

```Plain
OpenSearchKeywordSearchProvider
```

## ES 写入数据

文档上传索引时，当前已经写入：

```Plain
pgvector chunks 表
```

新增后，需要同时写入：

```Plain
Elasticsearch chunks index
```

每个 chunk 写入 ES 的字段建议：

```JSON
{
  "tenant_id": "tenant_demo",
  "kb_id": "kb_xxx",
  "document_id": "doc_xxx",
  "chunk_id": "chunk_xxx",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。",
  "metadata": {
    "chunk_index": 0,
    "source_type": "text"
  },
  "created_at": "2026-06-10T10:00:00Z"
}
```

要求：

- ES 中的 `chunk_id` 必须和数据库中的 chunk ID 保持一致
- 必须写入 `tenant_id` 和 `kb_id`，用于检索过滤
- 不要把 embedding 向量写入 ES
- ES 写入逻辑封装在 `KeywordSearchProvider.add_chunks()`

## BM25 查询要求

BM25 查询必须带过滤条件：

```Plain
tenant_id = request.tenant_id
kb_id = request.kb_id
```

建议检索字段：

```Plain
title
content
```

第一版可以使用 `multi_match`：

```JSON
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "tenant_id": "tenant_demo" } },
        { "term": { "kb_id": "kb_xxx" } }
      ],
      "must": {
        "multi_match": {
          "query": "退款需要几天内申请？",
          "fields": ["title^2", "content"]
        }
      }
    }
  },
  "size": 20
}
```

## RRF 融合逻辑

新增 `HybridSearchService`，负责融合两路结果。

输入：

```Python
vector_chunks: list[dict]
bm25_chunks: list[dict]
rrf_k: int
top_n: int
```

向量召回结果示例：

```JSON
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。",
  "score": 0.86
}
```

BM25 召回结果示例：

```JSON
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。",
  "bm25_score": 12.4
}
```

融合后输出：

```JSON
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "title": "退款政策",
  "content": "用户可在订单完成后 7 天内申请退款。",
  "score": 0.0321,
  "vector_score": 0.86,
  "bm25_score": 12.4,
  "vector_rank": 1,
  "bm25_rank": 2,
  "retrieval_source": "hybrid"
}
```

字段说明：

| 字段                 | 说明                                       |
| -------------------- | ------------------------------------------ |
| `score`            | 最终融合分数，即 RRF fused_score           |
| `vector_score`     | 原始向量召回分数，没有则为`null`         |
| `bm25_score`       | 原始 BM25 分数，没有则为`null`           |
| `vector_rank`      | 在向量召回结果中的排名，没有则为`null`   |
| `bm25_rank`        | 在 BM25 召回结果中的排名，没有则为`null` |
| `retrieval_source` | `vector`、`bm25` 或 `hybrid`         |

排序规则：

```Plain
按 score 从高到低排序
```

其中：

```Plain
score = fused_score
```

## 和 rerank 的关系

正确顺序是：

```Plain
向量召回 + BM25 召回
  -> RRF 融合
  -> 得到候选 chunks
  -> 可选 LLM rerank
  -> 返回 top_n chunks
```

说明：

- 向量检索和 BM25 都属于召回阶段
- RRF 属于召回结果融合
- LLM rerank 属于精排阶段
- 如果开启 rerank，rerank 处理的是 RRF 融合后的候选 chunks

## 接口请求变化

保留现有接口：

```HTTP
POST /api/v1/rag/retrieve
```

请求体增加可选 `retrieval_options`：

```JSON
{
  "tenant_id": "tenant_demo",
  "kb_id": "kb_xxx",
  "user_id": "user_demo",
  "query": "退款需要几天内申请？",
  "top_k": 20,
  "retrieval_options": {
    "mode": "hybrid",
    "vector_top_k": 20,
    "bm25_top_k": 20,
    "rrf_k": 60
  },
  "rerank_options": {
    "enabled": true,
    "top_n": 5
  }
}
```

字段说明：

| 字段                               | 类型    | 是否必须 | 说明                                           |
| ---------------------------------- | ------- | -------- | ---------------------------------------------- |
| `retrieval_options.mode`         | string  | 否       | `vector`、`bm25`、`hybrid`，默认读取配置 |
| `retrieval_options.vector_top_k` | integer | 否       | 向量召回数量                                   |
| `retrieval_options.bm25_top_k`   | integer | 否       | BM25 召回数量                                  |
| `retrieval_options.rrf_k`        | integer | 否       | RRF 参数，默认 60                              |

兼容要求：

- 如果不传 `retrieval_options`，按 `.env` 配置决定检索模式
- 如果配置未开启 hybrid，则保持当前向量检索行为
- 不要新增额外 HTTP 接口

## 接口响应变化

响应仍保持统一结构：

```JSON
{
  "code": 0,
  "msg": "success",
  "data": {}
}
```

`data.retrieved_chunks` 增加混合检索相关字段：

```JSON
{
  "code": 0,
  "msg": "success",
  "data": {
    "query": "退款需要几天内申请？",
    "kb_id": "kb_xxx",
    "retrieved_chunks": [
      {
        "document_id": "doc_001",
        "chunk_id": "chunk_001",
        "title": "退款政策",
        "content": "用户可在订单完成后 7 天内申请退款。",
        "score": 0.0321,
        "vector_score": 0.86,
        "bm25_score": 12.4,
        "vector_rank": 1,
        "bm25_rank": 2,
        "retrieval_source": "hybrid",
        "rerank_score": 0.95
      }
    ],
    "metadata": {
      "top_k": 20,
      "retrieval": {
        "mode": "hybrid",
        "fusion": "rrf",
        "rrf_k": 60,
        "vector_store": "pgvector",
        "keyword_search": "elasticsearch",
        "vector_top_k": 20,
        "bm25_top_k": 20,
        "vector_count": 20,
        "bm25_count": 18,
        "fused_count": 31
      },
      "rerank": {
        "enabled": true,
        "provider": "llm",
        "top_n": 5
      }
    }
  }
}
```

注意：

- 不返回 `context_text`
- 不生成最终答案
- `score` 在 hybrid 模式下表示 RRF 融合分数
- `vector_score` 和 `bm25_score` 保留原始检索分数，方便调试和评估

## 配置项

在 `.env.example`和.env中新增：

```Plain
KEYWORD_SEARCH_PROVIDER=elasticsearch
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=rag_chunks

RETRIEVAL_MODE=vector
HYBRID_FUSION=rrf
HYBRID_RRF_K=60
HYBRID_VECTOR_TOP_K=20
HYBRID_BM25_TOP_K=20
HYBRID_TOP_N=20
```

说明：

| 配置项                      | 说明                                           |
| --------------------------- | ---------------------------------------------- |
| `KEYWORD_SEARCH_PROVIDER` | 关键词检索 provider，第一版为`elasticsearch` |
| `ELASTICSEARCH_URL`       | ES 地址                                        |
| `ELASTICSEARCH_INDEX`     | chunk 索引名                                   |
| `RETRIEVAL_MODE`          | 默认检索模式，`vector`、`bm25`、`hybrid` |
| `HYBRID_FUSION`           | 融合策略，第一版为`rrf`                      |
| `HYBRID_RRF_K`            | RRF 参数，默认 60                              |
| `HYBRID_VECTOR_TOP_K`     | hybrid 模式下向量召回数量                      |
| `HYBRID_BM25_TOP_K`       | hybrid 模式下 BM25 召回数量                    |
| `HYBRID_TOP_N`            | 融合后默认返回候选数量                         |

## Docker Compose

在 `docker-compose.yml` 中新增 Elasticsearch 服务。

本地测试可以使用单节点模式：

```YAML
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.17.10
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false
    - ES_JAVA_OPTS=-Xms512m -Xmx512m
  ports:
    - "9200:9200"
  volumes:
    - es_data:/usr/share/elasticsearch/data
```

同时增加 volume：

```YAML
volumes:
  es_data:
```

## 失败降级策略

要求：

- 如果 `RETRIEVAL_MODE=vector`，完全不依赖 ES
- 如果 `RETRIEVAL_MODE=hybrid`，pgvector 检索失败则接口失败
- 如果 `RETRIEVAL_MODE=hybrid`，ES BM25 检索失败，可以降级为纯向量结果，但一定要记录此失败日志
- 降级时需要在 metadata 中标记

降级响应示例：

```JSON
{
  "metadata": {
    "retrieval": {
      "mode": "hybrid",
      "fusion": "rrf",
      "degraded": true,
      "degraded_reason": "bm25 search failed",
      "vector_store": "pgvector",
      "keyword_search": "elasticsearch"
    }
  }
}
```

## 日志要求

需要增加关键日志：

```Plain
HYBRID_SEARCH_START
VECTOR_SEARCH_SUCCESS
BM25_SEARCH_SUCCESS
BM25_SEARCH_FAILED
RRF_FUSION_SUCCESS
HYBRID_SEARCH_DEGRADED
```

日志中至少包含：

- `tenant_id`
- `kb_id`
- `query`
- `vector_top_k`
- `bm25_top_k`
- `vector_count`
- `bm25_count`
- `fused_count`
- `cost_ms`

## 实现边界

第一版不要实现：

- ES 管理后台
- learning-to-rank
- 动态权重训练
- 多字段复杂 boost 策略
- 多索引路由

第一版只要求：

- 写入 ES chunk index
- 查询 ES BM25
- title/content 使用 IK 中文分词
- 与 pgvector 结果按 chunk_id 合并
- 使用 RRF 融合排序
- 返回结构化检索结果

## 验收标准

完成后需要满足：

- 代码中存在 `KeywordSearchProvider` 抽象
- 代码中存在 `ElasticsearchKeywordSearchProvider` 实现
- ES index mapping 中 `title` 和 `content` 使用 `ik_max_word` / `ik_smart`
- 文档上传成功后，chunk 同时写入 pgvector 和 ES
- `/api/v1/rag/retrieve` 支持 `retrieval_options.mode = hybrid`
- hybrid 模式下会同时执行向量召回和 BM25 召回
- 两路结果按 `chunk_id` 合并
- 使用 RRF 计算 `score`
- 响应中包含 `vector_score`、`bm25_score`、`vector_rank`、`bm25_rank`、`retrieval_source`
- ES 失败时可以降级为纯向量结果
- 不生成最终答案
- 不返回 `context_text`
