# LangGraph 的持久化机制是什么？如何保存执行状态？

LangGraph 的持久化让一次图运行可以跨越失败、中断和进程重启。它不是把对话历史随便塞进列表，而是在超级步边界写入可恢复的快照。官方把持久化分成两套互补系统：Checkpointer 保存一条线程里的图状态，Store 保存跨线程的长期数据。

## Checkpointer：线程内的短期记忆

Checkpointer 在每个超级步结束后保存检查点。一次调用对应一条线程，用 `thread_id` 标识。有了它，图可以：

- 在人机审批处暂停，稍后从同一点继续
- 进程崩溃后按线程恢复
- 查看历史检查点，做时间旅行式重放

开发环境常用内存实现：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke({"messages": [{"role": "user", "content": "我叫 Bob"}]}, config)
```

`InMemorySaver` / `MemorySaver` 把检查点放在内存里，进程退出就消失。本地演示足够，生产环境要换成 `SqliteSaver` 或 `PostgresSaver` 这类落盘实现。`PostgresSaver` 还要注意 `thread_id` 长度限制，过长会写入失败，官方建议控制在 255 个字符以内。

## Store：跨线程的长期记忆

Store 不保存图的完整运行快照，而是保存你自己定义的键值数据，例如用户偏好、长期事实、跨会话共享知识。Checkpointer 管“这次任务跑到哪了”，Store 管“这个用户以前说过什么”。大多数应用可以两套一起用：编译时同时传入 checkpointer 和 store。

不要用 Checkpointer 冒充长期记忆。把用户画像塞进某一条线程的状态里，换一个 `thread_id` 就看不见了。跨会话数据应写入 Store。

## 如何保存和恢复执行状态

保存是自动的：只要编译时带了 checkpointer，执行过程中会按超级步写入。恢复时使用同一个图、同一个 checkpointer、同一个 `thread_id` 再次 `invoke` 或 `stream`。运行时先读出该线程最新检查点，再继续。

人机交互恢复时，下一次调用传入 `Command(resume=...)`，而不是重新提交原始输入。检查点里已经有暂停前的状态，resume 值会成为 `interrupt()` 的返回值。

检查点不会无限增长而不产生代价。很长的会话会积累大量快照，占用存储并可能拖慢读取。生产上需要定期清理过期检查点，或配置保留策略。这是运维问题，不是把 checkpointer 关掉就能解决的。

## 应用场景

保险理赔助手可能在“等待客户补充发票”这一步停三天。进程不可能一直挂着等。Checkpointer 把当前案件状态写入 Postgres，三天后客户上传发票，服务用原来的 `thread_id` 恢复图，从等待节点继续，而不是从头识别案件。同一客户的常用银行账户则放进 Store，新案件也能读到。

## 容易混淆的边界

LangGraph 持久化不是 Chroma 那种向量索引，也不是把 Markdown 文档存进数据库。本 RAG 项目的 `chroma_db/` 保存的是文档片段向量，和 LangGraph 的检查点是两回事。

它也不是 LangChain 早期那种 ConversationBufferMemory。后者通常只是把消息列表追加到提示词；LangGraph checkpointer 保存的是整张图的状态快照，包含自定义字段、工具结果和中断位置。

父图和子图各有检查点命名空间。子图更新的状态，父图不一定立刻按你想象的方式看见。需要跨图共享的数据，优先放 Store，或明确配置写入父检查点。不要假设“子图 return 了，父图状态就自动合并了一切”。
