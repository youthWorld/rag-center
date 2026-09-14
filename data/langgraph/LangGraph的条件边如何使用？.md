# LangGraph 的条件边如何使用？

条件边是图里的动态路由。普通边表示“做完 A 一定去 B”；条件边表示“做完 A 之后，根据当前状态选择下一个节点”。循环、重试、工具调用后是否结束，通常都靠条件边实现。

## 普通边和条件边的区别

普通边用 `add_edge(source, target)` 注册，运行时不再看状态，路径是固定的。条件边用 `add_conditional_edges` 注册，需要提供一个路由函数。路由函数读取状态，返回下一个节点的名字，也可以返回 `END` 表示结束。

这个区别很实际：固定流水线里的“清洗完就写入”适合普通边；“模型如果还要调工具就去工具节点，否则结束”必须用条件边。把后者写成普通边，图就无法根据模型输出变化。

## 基本写法

路由函数的返回值必须能映射到已经注册的节点，或 `END`。为了避免拼写错误，常用一个字典作为路径映射：返回值是映射的键，值才是真正的节点名。

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    need_tool: bool
    answer: str

def call_model(state: AgentState) -> dict:
    need_tool = "查询库存" in state["question"]
    return {"need_tool": need_tool}

def route_after_model(state: AgentState) -> Literal["use_tool", "finish"]:
    return "use_tool" if state["need_tool"] else "finish"

def use_tool(state: AgentState) -> dict:
    return {"answer": "库存还剩 12 件"}

def finish(state: AgentState) -> dict:
    return {"answer": state.get("answer") or "直接给出结论"}

builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("use_tool", use_tool)
builder.add_node("finish", finish)
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    route_after_model,
    {"use_tool": "use_tool", "finish": "finish"},
)
builder.add_edge("use_tool", "finish")
builder.add_edge("finish", END)
```

上面的 `route_after_model` 只做路由，不修改状态。这是条件边最干净的用法：判断函数保持无副作用，方便单测。

## 应用场景

仓库客服机器人会先让模型看用户问题。如果问题只是问好，条件边直接走到回复节点；如果问题涉及库存或物流单号，条件边走到工具节点，工具结果写回状态后再回到模型节点。这会形成循环：模型 -> 条件边 -> 工具 -> 模型。循环次数可以用状态里的计数器限制，路由函数在超过上限时返回结束，避免死循环。

## 常见细节

路由函数可以返回一个节点名，也可以返回节点名列表，表示同时激活多个下游。教学里先掌握返回单个名字即可。

路径映射不是必须的，但建议写上。没有映射时，返回值会被当成节点名；一旦返回了未注册的名字，运行到这一步才会失败。有映射时，非法返回值更容易在开发阶段暴露。

条件边看的是“当前状态快照”，不是节点函数的局部变量。节点如果计算出了路由依据，必须把依据写进状态，路由函数才能看见。只在节点局部变量里判断，条件边拿不到。

## 容易混淆的边界

条件边不是异常处理。节点抛错不会自动走另一条条件边，除非你把错误写进状态并自己路由。错误重试、降级和人机打断见错误处理文档。

条件边也不是人机交互。它只是在图结构上选择下一节点，不会暂停等待人类。需要审批时用 `interrupt()`，不要指望条件边自己停住。

还有一种在节点内部返回 `Command(goto=...)` 的写法，路由发生在节点返回值里，而不是 `add_conditional_edges`。两种都可以实现分支。条件边把路由留在图结构上，更直观；`Command` 适合节点在运行时才知道要跳到哪里。本项目文档讨论的“条件边”特指 `add_conditional_edges` 这种图结构路由。
