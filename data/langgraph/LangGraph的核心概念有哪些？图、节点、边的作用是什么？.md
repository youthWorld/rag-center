# LangGraph 的核心概念有哪些？图、节点、边的作用是什么？

LangGraph 用一张有向图描述智能体。开发时先定义共享状态，再往图里加节点和边，最后编译。真正干活的是节点，决定下一步去哪的是边，所有节点读写的那份数据叫做状态。官方有一句很准确的概括：节点做工作，边告诉下一步做什么。

## 图

图的常用实现是 `StateGraph`。它由状态模式参数化，内部按超级步推进：同一超级步里可以并行运行多个节点，下一超级步再运行它们的下游。执行开始时节点都是未激活的；节点收到沿边传来的状态后被激活，运行自己的函数，再把更新发出去。当没有节点处于激活状态、也没有在途消息时，图结束。

图不是装饰性的流程图。它决定哪些步骤可以并行、哪些必须串行、哪些允许循环。没有循环的线性链也能用图表达，但 LangGraph 的价值在于循环和分支是一等公民。

## 节点

节点是普通 Python 函数，或任何可调用对象。它接收当前状态，做计算或副作用，然后返回状态更新。更新通常是一个部分字典，不必返回完整状态。节点里可以调用大模型、执行工具、查数据库，也可以只做规则判断。

节点不要承担路由职责，除非你显式使用 `Command` 这类在节点内部指定下一跳的写法。默认设计里，节点只负责“这一步做什么”，下一步由边决定。这样测试节点时只需喂入一段状态，断言返回的更新即可。

## 边

边连接节点，表示控制流。固定边永远从 A 走到 B，例如从 `START` 进入第一个业务节点，或从最后一个节点走到 `END`。条件边会先运行一个路由函数，根据当前状态返回下一个节点名。

`START` 和 `END` 是图的入口和出口，不是业务节点。漏掉从 `START` 出发的边，或留下没有任何出边、又不到达 `END` 的孤立节点，编译阶段通常会检查出来。

## 应用场景

工单分流系统可以把流程画成：入口节点读取邮件，分类节点判断是咨询、投诉还是故障，然后由边分别接到知识库回答、安抚话术或创建技术工单。分类节点只输出类别，不自己调用三个下游；三个下游节点各自实现，图结构把它们连起来。以后要插入“敏感词检查”节点，只需加节点和边，不必改原有分类函数。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class TicketState(TypedDict):
    email: str
    category: str
    reply: str

def classify(state: TicketState) -> dict:
    text = state["email"]
    category = "fault" if "宕机" in text else "consult"
    return {"category": category}

def answer_consult(state: TicketState) -> dict:
    return {"reply": "这是咨询答复"}

builder = StateGraph(TicketState)
builder.add_node("classify", classify)
builder.add_node("answer_consult", answer_consult)
builder.add_edge(START, "classify")
builder.add_edge("answer_consult", END)
```

## 容易混淆的边界

状态不是节点。状态是图上流动的数据结构，常用 `TypedDict` 或 Pydantic 模型描述。节点读写状态，但状态本身不执行。

边也不是节点。有人把“判断是否继续”写成一个大节点，在函数末尾用 if/else 调用其他函数。那是把图又写回了普通过程式代码。判断下一跳应该优先用条件边，或在节点里返回 `Command(goto=...)`，而不是在节点中直接调用另一个节点函数。

图、节点、边这三个词在本文档中特指 LangGraph 的编排概念。它们与神经网络里的图、AST 的边不是一回事。条件边的具体写法见单独文档，这里只说明边负责路由。
