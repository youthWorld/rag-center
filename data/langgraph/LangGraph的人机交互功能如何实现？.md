# LangGraph 的人机交互功能如何实现？

人机交互（HITL）指图在运行中途停下来，把控制权交给人类，收到输入后再继续。LangGraph 把这种暂停做成一等能力：调用 `interrupt()` 时，运行时用 Checkpointer 保存状态，然后无限期等待，直到你用 `Command(resume=...)` 恢复。

## 为什么必须有 Checkpointer

暂停不是 `input()` 那样阻塞当前线程。图可以把状态写到存储里，进程退出也没关系。恢复时用同一个 `thread_id` 找到检查点。没有 checkpointer，中断没有地方可写，人机交互无法工作。生产环境应使用可落盘的 checkpointer，避免服务重启后待审批任务全部丢失。

`thread_id` 相当于游标。复用它就是恢复同一条任务；换新 ID 会开始一条没有历史的新线程。

## 使用 interrupt()

在任何节点里调用 `interrupt(payload)`。payload 必须 JSON 可序列化，会返回给调用方，用来展示审批信息。恢复时传入的值会成为这行 `interrupt()` 的返回值，节点再根据返回值继续。

```python
from langgraph.types import interrupt, Command

def approval_node(state: dict) -> dict:
    approved = interrupt({
        "question": "是否批准这次退款？",
        "amount": state["amount"],
    })
    return {"approved": approved}
```

第一次 `invoke` 或 `stream_events` 会在这里停住。使用 `invoke` 时，中断信息出现在结果的 `__interrupt__` 里；使用 `stream_events(..., version="v3")` 时，可通过 `stream.interrupted` 和 `stream.interrupts` 读取。

```python
config = {"configurable": {"thread_id": "refund-1"}}
stream = graph.stream_events({"amount": 880}, config=config, version="v3")
final = stream.output
if stream.interrupted:
    print(stream.interrupts)
    resumed = graph.stream_events(Command(resume=True), config=config, version="v3")
```

恢复必须带原来的 `thread_id`。`Command(resume=...)` 是用来继续中断的输入方式；不要把 `Command(update=...)` 当成下一轮用户消息传给 `invoke`。

## 静态断点

除了代码里的 `interrupt()`，还可以在编译时设置 `interrupt_before` 或 `interrupt_after`，在进入或离开某个节点时暂停。这适合“这个节点很危险，每次都要停一下”的固定策略。`interrupt()` 则是动态的，可以按金额、风险分数决定停不停。

两种都可以和 checkpointer 配合。动态中断更适合业务分支，静态断点更适合调试或统一审批卡点。

## 应用场景

财务机器人准备调用转账 API 前，把收款人、金额、用途放进 `interrupt()` 的 payload。前端展示这些字段，出纳可以批准、拒绝，或改金额后再批准。恢复值可以是布尔值，也可以是包含修改结果的对象。图根据恢复值走执行节点或取消节点，而不是在节点里自己 `print` 然后死等标准输入。

并行节点可能同时中断。这时恢复值要用“中断 ID -> 回复”的映射一次性提交，确保每个中断对上自己的答案。

## 容易混淆的边界

人机交互不是条件边。条件边只选择下一个节点，不等待人类。需要外部输入时用中断。

人机交互也不是把错误抛给调用方。抛异常会让这次运行失败；`interrupt()` 是预期内的暂停，状态是可恢复的。用户缺信息、需要审批，属于中断；数据库连接失败属于错误处理。

还有一个容易踩坑的点：恢复后，发生中断的那个节点会从函数开头重新执行，`interrupt()` 之前的代码会再跑一遍。把不可重复的副作用放在 `interrupt()` 之前，可能导致重复扣款或重复发邮件。审批通过后再执行外部写入，或保证这些操作幂等。
