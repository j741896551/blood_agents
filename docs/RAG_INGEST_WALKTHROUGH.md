# RAG 文档下载、审核与入库：可跟做演示

本教程把四件事严格分开：**下载官方原文 → 提取待审核草稿 → 审核并登记短篇 Markdown → 建立向量索引、检索验收**。下载目录不是运行时知识库；FAQ 只读取 `blood_agents/knowledge/corpus.json` 白名单中的文档。公开资料用于一般知识，不提供个人献血资格诊断；个人记录、积分、预约等仍必须查询业务工具。

## 本次已下载的原文

| 本地文件（在 `data/knowledge_sources/`） | 官方来源 | 用途与状态 |
|---|---|---|
| `血站技术操作规程-2026版.pdf` | [辽宁省卫健委转载国家卫健委通知及附件](https://wsjk.ln.gov.cn/wsjk/zfxxgk/zc/gjjwj/2026070215022620516/index.shtml) | 2026 版规程原文；PDF 第 130—131 页含献血间隔。已下载并抽取待审核草稿，**PDF 全文未入库**。SHA-256：`e582e963e22bd15fb518334adc12198a676c09e50be40eebe0c88567628edeb0` |
| `中华人民共和国献血法.pdf` | [阜新市卫健委公开 PDF](https://wsjk.fuxin.gov.cn/wjw/file/2022-01-24/1643011577007ff8080817de469ed173017e8b1f14af7ac8.pdf) | 法律原文供核对；**PDF 全文未入库**。SHA-256：`a4eb3716f6367c23817db490092f5e4e95c5b0065440afc84469550c805dd178` |
| `全国电子无偿献血证查询流程.pdf` | [国家卫健委公开 PDF](https://www.nhc.gov.cn/yzygj/c100068/202006/df0e888f92e642c68c13cb2976503b97/files/1732872943779_97980.pdf) | 电子证流程原文供核对；**PDF 全文未入库**。SHA-256：`d39c96a026bf341430924fce5e29ceee04feb2f8a49504bb25e624618dd845a8` |

本次只下载了这三份可获取的代表性 PDF。`corpus.json` 中其余公开来源已有人工摘要和原始网页链接，**没有声称它们的原文都已下载**。国家标准 `GB 18467-2011` 的原文另受版权标注约束，这里未镜像全文；只保留[国家标准平台信息页](https://std.samr.gov.cn/gb/search/gbDetailed?id=71F772D7DDF2D3A7E05397BE0A0AB82A)供核对。

下载目录、待审核草稿和教学索引均在 `data/` 下，Git 忽略；`corpus.json` 与通过审核的短摘要在项目代码中。不要将下载 PDF 或个人业务数据直接提交到公开仓库。

这三份文件已经下载好；如果要从头复现下载，可在项目根目录执行（覆盖同名文件前先自行备份/核对）：

```bash
mkdir -p data/knowledge_sources
curl -fL 'https://wsjk.ln.gov.cn/wsjk/zfxxgk/zc/gjjwj/2026070215022620516/2026070215012524766.pdf' \
  -o data/knowledge_sources/血站技术操作规程-2026版.pdf
curl -fL 'https://wsjk.fuxin.gov.cn/wjw/file/2022-01-24/1643011577007ff8080817de469ed173017e8b1f14af7ac8.pdf' \
  -o data/knowledge_sources/中华人民共和国献血法.pdf
curl -fL 'https://www.nhc.gov.cn/yzygj/c100068/202006/df0e888f92e642c68c13cb2976503b97/files/1732872943779_97980.pdf' \
  -o data/knowledge_sources/全国电子无偿献血证查询流程.pdf
shasum -a 256 data/knowledge_sources/*.pdf
```

下载后比较上表的 SHA-256，并确认文件能作为 PDF 打开；如果官方以后更新原文，哈希变化时应重新核对版本，而不是盲目沿用旧摘要。

## 跟做第一步：检查并提取原文

在项目根目录，使用 Python 3.12 环境：

```bash
.venv312/bin/python -m pip install -e '.[rag,rag_ingest]'

.venv312/bin/python -m blood_agents.knowledge.ingest inspect \
  data/knowledge_sources/血站技术操作规程-2026版.pdf --pages 130-131

.venv312/bin/python -m blood_agents.knowledge.ingest stage \
  data/knowledge_sources/血站技术操作规程-2026版.pdf --pages 130-131 \
  --source-url 'https://wsjk.ln.gov.cn/wsjk/zfxxgk/zc/gjjwj/2026070215022620516/2026070215012524766.pdf' \
  --output data/knowledge_sources/drafts/我的献血间隔草稿.md
```

`inspect` 应显示原 PDF 共 218 页、选中 PDF **物理页码** 130—131、提取 737 个字符（不同 `pypdf` 版本可能略有差异）。`stage` 使用排他创建：目标文件已存在时会拒绝覆盖，请换文件名。先打开草稿，对照 PDF 同页核对数字、单位、例外条件、版本与日期；扫描件、表格和断行可能提取错误。草稿中的“待审核”标记不是装饰，它不能直接参与问答。

## 第二步：批准短篇资料，而不是把 PDF 全文塞进模型

当前已经有经过人工改写、白名单登记的[献血间隔摘要](../blood_agents/knowledge/documents/industry/donation_interval_rules.md)，对应 [corpus.json](../blood_agents/knowledge/corpus.json) 的 `donation_interval_rules`。其公开引用来源是福建省卫健委文章；本次下载的官方 PDF 第 130—131 页可用来**复核**摘要中的全血→全血 180 天、全血→单采血小板 90 天、单采血小板→全血 28 天、单采血小板→单采血小板 14 天等数字。该摘要不是 PDF 自动抽取结果，二者来源不可混称。例外条件和最终献血资格需现场评估，不应只凭四个数字作个体建议。

以后自己新增资料时：

1. 从已核实的原文挑选一个明确主题，人工核对后写成 `blood_agents/knowledge/documents/industry/<主题>.md` 短文，保留适用范围和限制。不要直接复制大段原文，也不要混入未公开公司资料。
2. 在 `blood_agents/knowledge/corpus.json` 增加一条记录，至少填写唯一 `id`、`title`、`source`（HTTPS 原文地址）、`source_kind=public`、`published_at`、`checked_at`、`provenance` 和相对 `path`。同一来源 URL 的不同主题可以分成多条记录，片段 ID 由记录 ID 区分。
3. 增加正例和无依据负例，先跑测试/检索；人工检查片段与原文是否一致，再运行下方入库命令。修改已批准资料会产生新的内容指纹 collection；旧 collection 不自动删除。

建议新资料先由你审阅并确认后再写入白名单。本次的 PDF 草稿**尚未新增为白名单条目**；教学入库用的仍是当前 15 篇已审核文档。

## 第三步：切分、编码、写入本地向量数据库

```bash
GRPC_ENABLE_FORK_SUPPORT=0 .venv312/bin/python -m blood_agents.knowledge.ingest index \
  --uri data/rag_ingest_walkthrough_20261025.db \
  --collection blood_knowledge_walkthrough \
  --query '全血献血后多久可以再献单采血小板'
```

流程：`corpus.json` 读取已批准 Markdown → 按标题/段落/180 字滑窗（重叠 30 字）切分并附标题、来源、时间等元数据 → 本地 `BAAI/bge-large-zh-v1.5` 将每段编码为向量 → Milvus Lite 写入内容指纹 collection → 用同一编码模型编码问题、进行向量与关键词混合召回。**这里不调用 DeepSeek 等生成模型 API**；首次缺少 BGE 权重时，下载模型仍需要网络。Milvus Lite 在本机可能将以 `.db` 结尾的路径创建为**目录**，不必假定它是 SQLite 单文件；它也不是存对话/业务数据的 SQLite 库。

本机 2026-10-25 的实际结果：15 篇已批准资料，切分后 70 段，Milvus 返回 70 个已索引 ID；检索问题命中间隔摘要。重复执行相同命令可复用同一内容指纹 collection，不重复插入。这个验收只说明入库与检索链路可运行，**不等于 FAQ 最终答复正确或医学可靠**。

若本机没有 `127.0.0.1` 监听权限，Milvus Lite 会报 `Failed to bind to address 127.0.0.1:0`；请在本机终端而非限制本地端口的隔离环境执行。已预先下载模型时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 避免再次联网。

## 第四步：让网页演示读取同一个索引

在启动 Gradio **之前**显式设置相同路径和 collection 基名：

```bash
GRPC_ENABLE_FORK_SUPPORT=0 VECTOR_BACKEND=milvus \
RAG_MILVUS_URI="$(pwd)/data/rag_ingest_walkthrough_20261025.db" \
MILVUS_COLLECTION=blood_knowledge_walkthrough \
.venv312/bin/python -m blood_agents.web
```

网页仍需你自己的平台模型配置才能完成导航、证据核验和自然语言答复；不配置模型不影响上述**本地入库**演示。可以先问“全血献血后多久可以献单采血小板”，查看 FAQ 的工具轨迹/来源，确认用的是新索引。默认 `VECTOR_BACKEND=memory` 不会打开这个 Milvus 数据库；如果页面已经在运行，改环境变量后须重启进程。

## 你可以如何验证

- 原文层：核对本地 PDF 第 130—131 页与草稿四个主要间隔数字，不用模型替你确认事实。
- 白名单层：打开 `corpus.json`，确认只有明确列出的已审核 Markdown 参与运行。
- 入库层：`index` 输出 `chunks` 与 `indexed_chunks` 相等，记录 `collection` 名；第二次运行后名称和数量不变。
- 检索层：输出 `hits` 里出现预期的献血间隔摘要和官方来源；“召回”仍要经过 FAQ 证据核验，不能自动当作答案。

如想直接观察分块代码，看 [`MultiStrategyChunker`](../blood_agents/knowledge/rag.py)、[`load_approved_corpus`](../blood_agents/knowledge/rag.py) 和 [`MilvusVectorStore`](../blood_agents/knowledge/adapters.py)；下载/提取/教学入库入口在 [`ingest.py`](../blood_agents/knowledge/ingest.py)。
