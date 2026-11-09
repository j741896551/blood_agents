from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from .orchestration.graph import build_conversational_graph
from .storage.repository import AgentRepository


def main() -> None:
    """启动命令行多轮对话入口。"""
    load_dotenv()
    repo = AgentRepository(os.getenv("BLOOD_AGENT_DB", "./blood_agent.db"))
    graph = build_conversational_graph(repo)
    print("智慧献血助手已启动（输入 quit 退出）")
    while True:
        text = input("你：").strip()
        if text.lower() in {"quit", "exit", "退出"}:
            break
        if not text:
            continue
        result = graph.invoke(
            {"messages": [HumanMessage(content=text)], "user_id": "U1001"},
            config={"configurable": {"thread_id": "cli-session"},
                    "metadata": {"entrypoint": "cli"}},
        )
        print("助手：" + result["messages"][-1].content)


if __name__ == "__main__":
    main()
