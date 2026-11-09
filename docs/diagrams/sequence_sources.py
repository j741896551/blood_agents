"""时序图文本源：与实际图入口、子图、工具和模型边界对应，不执行任何业务。"""

SEQUENCES = [
    {
        "id": "13-runtime-sequence",
        "title": "一轮消息：主导航与各领域分支",
        "sources": [
            ("blood_agents/orchestration/conversation.py", "def invoke(self, state"),
            ("blood_agents/orchestration/graph.py", "def build_graph"),
        ],
        "guide": "alt代表不同领域分支，不是一个请求同时运行全部Agent。每轮结束后保存状态，后续输入再续接；OCR只保留原型注册。",
        "text": """%%{init: {'theme': 'base', 'themeVariables': {'actorBkg': '#EAF2FF', 'actorBorder': '#3575C5', 'actorTextColor': '#24364C', 'noteBkgColor': '#FFF4DE'}}}%%
sequenceDiagram
    autonumber
    actor User as 用户
    participant C as ConversationGraph
    participant S as supervisor
    participant A as 预约子图
    participant P as 积分子图
    participant R as 记录子图
    participant F as FAQ只读循环
    participant T as ToolRegistry
    participant M as ModelGateway
    participant DB as 会话库与checkpoint
    User->>+C: user_id / session_id / message
    C->>DB: 隔离线程键，恢复流程状态
    Note over C,S: 严格静态缓存命中时直接返回；以下为未命中路径
    C->>+S: 当前消息、活动领域和任务快照
    S->>M: 模型导航与账户访问范围
    M-->>S: NavigationDecision
    S-->>-C: 单领域或顺序任务队列
    alt appointment
        C->>+A: action/slots理解与只读规划
        A->>M: 预约语义理解、工具规划
        A->>T: 查询资料、历史和号源
        T-->>A: 本轮业务事实
        opt 已交付完整预览且用户最终确认
            A->>T: submit_appointment，入口身份和幂等键
            T-->>A: 成功或拒绝回执
        end
        A->>M: 依据事实生成预约回复
        A-->>-C: 草稿、result、handoff
    else points
        C->>+P: 积分动作与待办状态
        P->>T: 余额、目录、地址、订单查询
        T-->>P: 当前事实
        opt 当前写操作已展示并获确认
            P->>T: 地址/兑换/订单写入
            T-->>P: 接口回执
        end
        P->>M: 依据事实生成积分回复
        P-->>-C: 待办或完成结果
    else records
        C->>+R: 当前用户记录查询
        R->>T: query_records
        T-->>R: 献血记录或错误
        R->>M: 依据记录生成答复
        R-->>-C: 查询结果
    else faq
        C->>+F: 组合只读目标
        loop 默认最多8步
            F->>M: 工具目录和已取得观察
            M-->>F: call_tool 或 finish
            opt 请求白名单业务/知识工具
                F->>T: 校验参数并注入身份
                T-->>F: 观察结果
            end
            opt date_offset内置计算
                F->>F: 验证日期与规则来源后运算
            end
        end
        F->>M: 证据辅助答复与核验
        F-->>-C: 观察、证据和答复
    else general
        C->>M: 问候、能力说明或范围拒绝
        M-->>C: 模型回复
    end
    Note over S,F: OCR注册项保留但当前暂缓；任务队列按顺序暂停和续接
    C->>DB: 保存checkpoint及脱敏对话
    C-->>-User: 本轮回复、trace_id与调试信息
""",
    },
    {
        "id": "14-concurrency-sequence",
        "title": "跨会话并发与平台主备边界",
        "sources": [
            ("blood_agents/orchestration/conversation.py", "def _session_lock"),
            ("blood_agents/llm/model_gateway.py", "def call(self, messages"),
        ],
        "guide": "并行发生在不同会话请求间；相同会话由同一进程内锁串行。示例表达可配置在途预算，不表示默认额度能同时容纳所有请求，也没有多Worker分布式锁。",
        "text": """%%{init: {'theme': 'base', 'themeVariables': {'actorBkg': '#EAF2FF', 'actorBorder': '#3575C5', 'actorTextColor': '#24364C', 'noteBkgColor': '#FFF4DE'}}}%%
sequenceDiagram
    autonumber
    actor U1 as 用户A
    actor U2 as 用户B
    participant C as ConversationGraph
    participant G as ModelGateway
    participant P as 主平台API
    participant B as 备用平台API
    Note over C,G: 同一进程；发送并发和主备在途数分别配置
    par 会话A
        U1->>C: 发送A的消息
        C->>C: 获得A的会话锁
        C->>G: A的模型调用
        G->>P: 可用预算内尝试主服务
        P-->>G: 返回或失败
        G-->>C: A的结果
    and 会话B
        U2->>C: 发送B的消息
        C->>C: 获得B的会话锁
        C->>G: B的模型调用
        alt 主服务有在途额度且未熔断
            G->>P: 尝试主服务
            P-->>G: 返回或失败
        else 主服务忙/熔断/错误
            G->>B: 剩余预算内尝试备用
            B-->>G: 返回或仍不可用
        end
        G-->>C: B的结果或明确不可用
    end
    Note over C: 同会话的后续消息须等待前一轮释放锁
    C-->>U1: A的回复
    C-->>U2: B的回复
    Note over G,B: 等待超时不保证底层请求已取消；不能因此自动重放业务写入
""",
    },
]
