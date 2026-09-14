# LangGraph 中如何处理错误和异常？

图里某个节点失败时，默认会让整次运行失败。生产里要把失败分成几类，分别处理：瞬时网络问题适合重试；工具报错可以交给模型换策略；缺用户信息应暂停等人补充；重试耗尽后才宣告失败或走降级节点。LangGraph 提供重试策略、状态内错误回环、中断和错误处理回调，而不是一个万能的 try/except 包住整张图。

## 瞬时错误：RetryPolicy

对超时、限流、偶发 HTTP 错误，在 `add_node` 时挂上 `RetryPolicy`。可以设置最大次数、间隔，以及哪些异常值得重试。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

def search_docs(state: dict) -> dict:
    raise TimeoutError("检索服务超时")

builder = StateGraph(dict)
builder.add_node(
    "search_docs",
    search_docs,
    retry_policy=RetryPolicy(max_attempts=3),
)
builder.add_edge(START, "search_docs")
builder.add_edge("search_docs", END)
```

只对真正瞬时的错误重试。业务上的“订单不存在”重试没有意义，只会放大负载。可以用 `retry_on` 过滤异常类型，例如网络错误重试，校验错误不重试。

重试会从节点函数开头再执行一遍，和中断恢复一样。节点里的副作用必须能安全重复，或先检查“是否已经做过”。

## 模型可恢复错误：把错误写入状态

工具调用失败时，把错误信息写进状态并走回模型节点，让模型换参数或换工具。这不是静默吞掉异常，而是把失败变成图里可见的数据。

```python
def call_tool(state: dict) -> dict:
    try:
        result = lookup_order(state["order_id"])
        return {"tool_result": result, "error": ""}
    except KeyError as exc:
        return {"error": f"工具失败: {exc}"}

def should_continue(state: dict) -> str:
    if state.get("error"):
        return "model"
    return "final"
```

条件边看到 `error` 字段后回到模型。模型根据错误改写下一步。要给这种回环加次数上限，防止工具一直失败时原地打转。

## 需要人来补齐的错误：中断

缺发票、地址含糊、权限不够，靠重试解决不了。这时应 `interrupt()`，把缺什么告诉调用方，等人类补数据后再 resume。这类失败叫暂停，不叫崩溃。

## 重试之后仍然失败

可以让图失败并把异常抛给调用方，由外层记录日志或告警；也可以配置节点级错误处理，进入降级节点，例如返回“当前无法查询，请稍后重试”并结束。选择取决于业务能不能接受降级答案。教学上先保证异常不会被空的 `except: pass` 吃掉。

## 应用场景

机票查询节点调用外部 API。遇到 429 或超时，用 RetryPolicy 重试三次。如果 API 返回“航班号格式错误”，不要重试，把错误写进状态让模型要求用户核对航班号。如果用户一直不提供航班号，用中断等待，而不是让节点空转。三次重试后接口仍然 500，则进入降级节点，告知用户系统繁忙，并保留 thread_id 以便稍后恢复。

## 容易混淆的边界

错误处理不是条件边的替代品。条件边处理的是正常业务分支，例如“要不要调用工具”。异常是节点没能完成它承诺的工作。

错误处理也不是持久化。Checkpointer 能让崩溃后恢复，但不会自动把一次工具失败变成成功。恢复后节点仍可能再次失败，还需要重试或降级策略。

不要在节点里捕获所有异常后返回空结果。那样图会认为本步成功，后续节点拿到空数据却不知道为什么。至少把错误写入状态，或让不该恢复的异常继续向上抛。静默吞掉未知异常会让排查变得非常困难。
