from __future__ import annotations

import os
import uuid

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .log.trace_logging import TraceContext, current_trace, mask_sensitive
from .orchestration.graph import build_conversational_graph
from .orchestration.router import IntentRouter
from .orchestration.user_profile import UserProfileSummarizer, is_profile_query
from .storage.repository import AgentRepository

load_dotenv()
repo = AgentRepository(os.getenv("BLOOD_AGENT_DB", "./blood_agent.db"))
router = IntentRouter.from_env()
graph = build_conversational_graph(repo, router=router)
profile_summarizer = UserProfileSummarizer(repo, router.response_llm)
app = FastAPI(title="智慧献血多智能体系统", version="0.1.0")


def _refresh_profile_after_response(user_id: str, session_id: str = "",
                                    trace_id: str | None = None) -> None:
    """后台任务绑定本次 graph 的对话库，不误读进程中其他/旧数据库。"""
    conversation_store = getattr(graph, "conversation_store", None)
    if conversation_store is None:
        return
    service = (profile_summarizer if conversation_store is repo
               else UserProfileSummarizer(conversation_store, router.response_llm))
    context = TraceContext(trace_id or uuid.uuid4().hex, user_id, session_id, "api_profile")
    token = current_trace.set(context)
    try:
        service.refresh(user_id)
    finally:
        current_trace.reset(token)


class ChatRequest(BaseModel):
    user_id: str = Field(default="U1001", min_length=1)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    message: str = Field(min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    answer: str
    intent: str
    trace_id: str
    result: dict = Field(default_factory=dict)
    model_info: dict = Field(default_factory=dict)
    response_info: dict = Field(default_factory=dict)
    tool_plan_info: dict = Field(default_factory=dict)
    session_id: str
    tool_trace: list[dict] = Field(default_factory=list)
    conversation_persistence: dict = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    """返回当前服务的健康检查结果。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks,
         x_user_id: str = Header(default="")) -> ChatResponse:
    """接收一轮 API 对话请求，并返回模型答复及结构化调试信息。"""
    # 生产环境应由 API Gateway 验证登录态后注入 X-User-ID。
    # request.user_id 仅作无鉴权环境下的兜底；模型和用户消息正文都不能决定身份。
    authenticated_user_id = x_user_id or request.user_id
    # 画像查询重试上轮未完成的增量提取，避免把暂未汇总说成用户没有爱好。
    if is_profile_query(request.message):
        _refresh_profile_after_response(authenticated_user_id, request.session_id)
    # API 与网页共用 ConversationGraph；返回的 session_id 需由调用方在下一轮原样传回。
    state = graph.invoke(
        {"messages": [HumanMessage(content=request.message)], "user_id": authenticated_user_id},
        config={"configurable": {"thread_id": request.session_id},
                "metadata": {"entrypoint": "api"}},
    )
    # FastAPI 在响应发送后执行画像增量摘要；不增加业务回复的首屏等待。
    if state.get("conversation_persistence", {}).get("status") == "saved":
        background_tasks.add_task(
            _refresh_profile_after_response, authenticated_user_id,
            request.session_id, state.get("trace_id"),
        )
    # answer 是用户可见文本；result/轨迹是调试视图，不应被调用方当作操作成功证明。
    return ChatResponse(
        answer=mask_sensitive(state["messages"][-1].content),
        intent=state["intent"],
        trace_id=state["trace_id"],
        result=mask_sensitive(state.get("result", {})),
        model_info=state.get("model_info", {}),
        response_info=state.get("response_info", {}),
        tool_plan_info=state.get("tool_plan_info", {}),
        session_id=request.session_id,
        tool_trace=state.get("tool_trace", []),
        conversation_persistence=state.get("conversation_persistence", {}),
    )


def mount_gradio_ui() -> None:
    """按需挂载 UI，缺少可选依赖时 API 仍能独立运行。"""
    try:
        import gradio as gr

        from .web import ChatApplication, build_ui
    except ImportError:
        return
    ui = build_ui(ChatApplication(os.getenv("BLOOD_AGENT_DB", "./blood_agent.db")))
    gr.mount_gradio_app(app, ui, path="/ui")


mount_gradio_ui()
