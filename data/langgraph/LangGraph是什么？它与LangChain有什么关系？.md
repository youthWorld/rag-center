# LangGraph 是什么？它与 LangChain 有什么关系？

LangGraph 是 LangChain 团队提供的底层编排运行时，用来构建长时间运行、带状态的智能体和工作流。它把一次任务建模成图：节点负责做事，边负责决定下一步，共享状态在图里流动。和许多“把提示词包一层就叫 Agent”的框架不同，LangGraph 把控制权交给开发者，允许在同一张图里混合确定性代码和模型驱动步骤。

## 它解决什么问题

普通的链式调用适合单向流水线，例如“检索 -> 生成 -> 输出”。真实业务里经常出现循环、分支、失败重试、中途等人审批、过几天再继续。这些需求要求程序能停下来、把状态存住、之后从同一点恢复。LangGraph 的核心能力正好对应这些场景：持久化执行、流式输出、人机交互和状态管理。

可以把它理解成智能体的操作系统内核。模型、工具、提示词仍然由你选择；LangGraph 负责调度它们何时运行、如何把结果写回状态、出错后怎么续跑。

## 和 LangChain 的关系

LangChain 与 LangGraph 是互补产品，不是互相替代的两个版本。

LangChain 更偏应用层：模型封装、工具定义、消息类型，以及常见的智能体循环。如果你只是要一个标准的“模型调工具再回答”的助手，LangChain 的高层 Agent 接口通常更快。

LangGraph 更偏运行时：图结构、检查点、中断恢复、长时间工作流。官方也强调，使用 LangGraph 并不强制依赖 LangChain。你可以只用普通 Python 函数当节点；需要调用聊天模型或工具时，再引入 LangChain 组件。

同一生态里还有 LangSmith，用于追踪、评估和发布，它既不是 LangChain 也不是 LangGraph。Deep Agents 这类更高层的 Agent Harness 会把规划、子智能体和文件系统能力架在 LangGraph 之上。检索问答时不要把这些名字混成同一个东西。

## 应用场景

客服退款助手需要先读工单，再判断是否满足退款政策，金额超过阈值时必须等人审批，审批通过后才调用支付接口。链路里既有规则判断，也有大模型起草说明，还有可能在审批节点停一整晚。用纯 LangChain 链式调用很难自然表达“停住、存状态、第二天继续”；用 LangGraph 可以把读工单、策略检查、人工审批、执行退款画成一张可恢复的图。

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class RefundState(TypedDict):
    ticket: str
    amount: float
    approved: bool

def policy_check(state: RefundState) -> RefundState:
    return {"approved": state["amount"] < 200}

builder = StateGraph(RefundState)
builder.add_node("policy_check", policy_check)
builder.add_edge(START, "policy_check")
builder.add_edge("policy_check", END)
graph = builder.compile()
```

## 容易混淆的边界

LangGraph 不是向量数据库，也不负责把文档切块检索。本项目的 RAG 流程用 Chroma 做检索，用千问做回答，这些都不需要 LangGraph。

LangGraph 也不是必须上生产才能用的重型平台。本地用 `InMemorySaver` 就能练习图的编译和执行；只有需要跨进程恢复时，才换成 SQLite 或 Postgres 检查点。

和 LangChain 相比：LangChain 提供积木，LangGraph 提供编排这些积木的图运行时。需要循环、持久化和人机审批时优先考虑 LangGraph；只是单次模型调用或线性链时，不必强行上图。
