# 优化二十：实现与工程验收记录

日期：2026-09-27。分支：feature/opt20-retrieval-performance。以专题05（含更新后的Q22）为准。

## 实现范围

- 公开档位固定5/5/5；默认/null为balanced；命名档位高级覆盖、未知字段、公开rrf_k报错，Evidence独立，custom数量和模式边界明确。
- 单库/多库共用全局召回，一次Embedding、一次全局SQL、一次全局BM25；成对授权范围、稳定RRF、去重、无每库保底。
- quality基础20 + 额外图最多10 → 精排最多30 → 最终5；保留Evidence快照窗口、Context及Research固定Top10策略。
- 两路并行；向量SQL使用独立只读事务并回滚释放；失败/取消、ES部分结果和逻辑调用统计已覆盖。
- FastAPI统一持有模型HTTP池及独立ES池，脚本/Worker管理自有资源；同步消费者、前端、README、.env及.env.example。

## 真实环境与预期

- 使用已配置PostgreSQL、Redis、Elasticsearch、模型服务和Langfuse；租户tenant_demo为Pro。调用已获用户无限费用预算授权；没有修改既有租户配额、Research预算和超时。
- 未替换原8000/15173服务。启动本任务隔离后端18020、前端15174；前端从frontend工作目录启动并代理至18020。Windows后端显式使用Selector事件循环。
- 使用既有电商平台运营规则库及客户服务与售后运营知识库；显式v1/v2版本。没有上传、重建索引、清理历史数据或Langfuse记录。
- 测前只读核对原文：营业执照/法人/账户、品牌授权、保证金1000—50000元、开店费通常0元、技术服务费0.6%—5%；普通类目审核24小时、敏感类目72小时；入驻资料另有1—3/1—5工作日等阶段说明。
- 问题：新店第一次上商品前要准备哪些资质、费用，通常多久能完成审核？验收检查返回真实原文与身份，不评估总体检索质量或收益。

## 真实API矩阵

| 范围 | 版本 | 档位 | 返回 | Embedding/SQL/BM25 | 模型逻辑调用 | 融合/基础/精排输入 | 结果 |
|---|---|---|---:|---|---:|---|---|
| 1kb | v1 | speed | 5 | 1/1/0 | 1 | 10/10/0 | PASS |
| 1kb | v1 | balanced | 5 | 1/1/1 | 1 | 16/16/0 | PASS |
| 1kb | v1 | quality | 5 | 1/1/1 | 3 | 27/20/20 | PASS |
| 1kb | v2 | speed | 5 | 1/1/0 | 1 | 10/10/0 | PASS |
| 1kb | v2 | balanced | 5 | 1/1/1 | 1 | 15/15/0 | PASS |
| 1kb | v2 | quality | 5 | 1/1/1 | 3 | 24/20/21 | PASS |
| 2kb | v1 | speed | 5 | 1/1/0 | 1 | 10/10/0 | PASS |
| 2kb | v1 | balanced | 5 | 1/1/1 | 1 | 16/16/0 | PASS |
| 2kb | v1 | quality | 5 | 1/1/1 | 3 | 30/20/20 | PASS |
| 2kb | v2 | speed | 5 | 1/1/0 | 1 | 10/10/0 | PASS |
| 2kb | v2 | balanced | 5 | 1/1/1 | 1 | 15/15/0 | PASS |
| 2kb | v2 | quality | 5 | 1/1/1 | 3 | 25/20/20 | PASS |

12/12组合通过。所有结果均在请求库/版本范围内；本题双库结果自然全部来自电商库（5+0），不是保底分配。三档返回原文中均包含资质、保证金及24/72小时说明。quality实际改写、精排均执行且未降级。此分布仅说明本次功能行为，不代表质量改善。

## 边界、业务与故障验收

- 13项API用例均符合预期状态：custom三模式、默认/null profile、Evidence、命名档位显式null拒绝、rrf_k拒绝、模式不匹配拒绝、跨租户404、不可查询版本400、完整BM25空结果、Research。
- 纯BM25的Embedding及模型总调用为0；空结果200且不伪造降级。Evidence真实执行并回查成功，结果complete。
- Research真实返回可引用答案；2个子检索、2次Embedding、2次精排、3次角色调用，总计7次；llm_call_count仍为3，终止evidence_complete。
- 真实向量维度错误触发数据库DataError后，请求会话SELECT 1及后续正确维度检索均成功。
- 本地故障注入：ES端点指向本机未监听端口，保留真实Embedding+SQL结果；exists即失败，BM25_search计数0。
- 本地故障注入：独立向量只读事务执行SELECT 1/0，保留真实ES结果；Embedding/向量/搜索计数1/1/1，fusion=none，请求会话仍可用。故障注入不是正常服务通过，也不是Mock联调。
- 取消、ES部分分片/超时、重试逻辑计数、池所有权和并行启动由确定性单元测试验证；不向共享生产服务人为制造分片故障。

## 前后端真实联调

- 浏览器通过15174 → 18020 → 真实数据库/ES/模型完成balanced、quality + Evidence、custom BM25、Research。
- 抓取请求体：quality仅发送范围/query/user/profile及独立evidence_options，没有隐藏高级覆盖；custom BM25仅提交bm25_top_k，不含vector_top_k或rrf_k；Research只提交范围/query/user。
- custom同义词默认关闭，开启后切换档位再切回可恢复草稿；真实同义词扩展反映在search_query。
- quality页面真实展示Vector20/BM2520、基础20、精排输入21、最终5、Embedding1；启用Evidence总调用4。页面Evidence原文可回查。
- 最终代码重启后再次完成quality + Evidence，截图见 logs/opt20-quality-browser.png。

## 自动化验收

- 完整后端：393 passed，35.58秒；见 logs/opt20-pytest-final.txt。
- 前端：11 passed；npm run build通过（保留既有>500KB包体警告）。
- Ruff：app与tests检查通过；git diff --check通过。
- 单元测试覆盖快照窗口、过滤不足不补查、Research失败/超时仍计模型调用、SDK重试不重计、共享池身份/鉴权/关闭、真实业务契约等。

## 证据与限制

- logs/opt20-real-matrix.json：12项真实API响应；logs/opt20-real-edges.json：13项边界/Evidence/Research响应。
- logs/opt20-real-isolation.json：真实数据库异常恢复和两类故障注入计数。
- logs/opt20-frontend-tests.txt、logs/opt20-build.txt及最终pytest日志：自动化输出。logs目录保持本地，不把既有历史日志混入提交。
- 早期隔离服务直接用uvicorn CLI在Windows下产生Proactor/psycopg不兼容，切换正确事件循环后重测通过。一次回归遇到原有多进程日志轮转WinError32，最终通过LOG_DIR隔离重跑；未更改日志策略或删除日志。
- 不执行正式实验、baseline/candidate、A/B、RAGAS；本轮不适用，不标PASS，不宣称质量、P95或连接池容量收益。
- 用户已确认不再自行进行前端测试，并授权执行AGENTS规定的提交、功能分支备份、主分支同步合并和推送流程；具体结果以Git记录为准。
