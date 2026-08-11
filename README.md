# LangChain + 火山方舟 Agent Plan 搭建 2-Step RAG

## 1. 项目介绍

本项目使用 Python、LangChain、火山方舟 Agent Plan 和 Chroma 构建本地知识库问答系统，支持以下文档格式：

- PDF
- TXT
- Markdown

项目采用经典的 2-Step RAG（Retrieval-Augmented Generation，检索增强生成）架构，分为两个主要阶段：

1. 向量知识库初始化：解析、清洗和切分文档，调用火山方舟 Embedding，将向量、原文和元数据写入 Chroma。
2. 在线知识库问答：向量化用户问题，从 Chroma 检索相关切片，构造 Prompt，调用火山方舟大语言模型生成带来源引用的回答。

本项目Codex生成，供初学者了解langchain和RAG基础知识和流程，模型和API可以根据个人使用习惯进行修改

> 运行要求：Python 3.10 或更高版本，推荐使用 Python 3.12。

### 主要组件

| 组件 | 作用 |
| --- | --- |
| Python | 项目开发语言 |
| LangChain | 编排文档、Embedding、Prompt 和模型调用 |
| 火山方舟 Agent Plan | 提供 Embedding 和对话模型 API |
| Chroma | 持久化和检索文档向量(向量库) |
| pypdf | 从 PDF 文件中提取文本 |
| python-dotenv | 从 `.env` 文件加载环境变量 |

## 2. 目录结构

```text
/home/RAG/
├── data/                              # PDF、TXT、Markdown 原始文档
│   └── test_intro.pdf
│
├── chroma_db/                         # Chroma 持久化向量数据库
│   └── chroma.sqlite3
│
├── utils/                             # 公共配置模块
│   ├── __init__.py                    # 将 utils 标记为 Python 包
│   ├── config.py                      # 路径、模型、切片和检索配置
│   ├── .env                           # API Key 和运行参数
│   └── requirements.txt               # Python 依赖列表
│
├── vector_init/                       # 向量知识库初始化模块
│   ├── __init__.py                    # 将 vector_init 标记为 Python 包
│   ├── document_processor.py          # 解析 PDF、TXT 和 Markdown
│   ├── text_cleaner.py                # 清洗文本并过滤空白文档
│   ├── text_chunker.py                # 将长文本拆分成 Chunk
│   ├── ark_embedding.py               # 创建并验证方舟 Embedding 客户端
│   └── vector_chroma_writer.py        # 将向量和原文写入 Chroma
│
├── rag_chat/                          # 在线问答流水线模块
│   ├── __init__.py                    # 将 rag_chat 标记为 Python 包
│   ├── query_vectorizer.py            # 将用户问题转换为向量
│   ├── chroma_retriever.py            # 使用 Chroma 执行 MMR 检索
│   ├── context_formatter.py           # 整理切片正文、来源和页码
│   ├── prompt_builder.py              # 构造知识库问答 Prompt
│   ├── ark_chat_model.py              # 调用火山方舟对话模型
│   └── answer_generator.py            # 生成最终回答
│
├── rag_vector_init.py                 # 向量知识库初始化入口
├── rag_chat_init.py                   # 在线知识库问答入口
└── README.MD                          # 项目说明
```

### 2.1 公共配置模块

#### `utils/__init__.py`

将 `utils` 目录标记为 Python 包，使项目可以通过以下方式导入配置：

```python
from utils.config import ARK_BASE_URL
```

#### `utils/config.py`

统一管理项目配置，主要负责：

- 显式加载 `utils/.env`
- 解析项目根目录
- 解析 `data/` 和 `chroma_db/` 的绝对路径
- 读取火山方舟 API 地址和模型名称
- 读取文本切片参数
- 读取 Chroma 检索参数
- 检查必需的环境变量

#### `utils/.env`

保存项目运行时的环境变量，包括：

- 火山方舟 API Key
- Agent Plan Base URL
- Embedding 模型
- 对话模型
- 数据目录和向量库目录
- 文本切片参数
- 检索参数

不要将包含真实 API Key 的 `.env` 提交到 Git 仓库。

#### `utils/requirements.txt`

记录项目需要安装的 Python 依赖。

### 2.2 向量知识库初始化模块

#### `vector_init/__init__.py`

将 `vector_init` 目录标记为 Python 包。

#### `vector_init/document_processor.py`

负责遍历 `data/` 目录并解析受支持的文档。

主要功能：

- 使用 `pypdf` 按页提取 PDF 文本
- 读取 TXT 文件
- 读取 Markdown 文件
- 尝试 UTF-8、UTF-8-SIG 和 GB18030 等编码
- 保存来源路径、页码和文件类型等元数据
- 单页解析失败时跳过该页，避免整个初始化流程中断

生成的文档元数据示例：

```python
{
    "source": "data/test_intro.pdf",
    "page": 0,
    "file_type": "pdf"
}
```

其中 PDF 页码从 `0` 开始存储，展示时转换为从 `1` 开始。

#### `vector_init/text_cleaner.py`

负责清理解析后的文本。

主要功能：

- 删除空字符
- 统一 Windows 和 Linux 换行符
- 合并多余空格
- 删除每行首尾空格
- 合并多余空行
- 过滤没有有效正文的页面或文件

#### `vector_init/text_chunker.py`

使用 LangChain 的 `RecursiveCharacterTextSplitter` 将长文档拆分为适合检索的文本切片。

默认配置：

```text
chunk_size = 800
chunk_overlap = 150
```

参数含义：

- `chunk_size`：每个切片的目标大小
- `chunk_overlap`：相邻切片之间保留的重叠内容

切片重叠可以降低关键信息恰好在切片边界处被截断的风险。

每个切片会增加 `chunk_index` 元数据：

```python
{
    "source": "data/test_intro.pdf",
    "page": 0,
    "file_type": "pdf",
    "chunk_index": 0
}
```

#### `vector_init/ark_embedding.py`

负责创建火山方舟 Agent Plan Embedding 客户端。

默认配置：

```text
Base URL: https://ark.cn-beijing.volces.com/api/plan/v3
Model: doubao-embedding-vision
```

在写入 Chroma 前，程序会先发送一条测试文本，检查：

- API Key 是否有效
- Base URL 是否正确
- Embedding 模型是否可用
- 接口是否返回非空向量
- 返回向量的维度

当前 `doubao-embedding-vision` 返回的向量维度为 2048。

#### `vector_init/vector_chroma_writer.py`

负责将文本切片写入 Chroma。

主要功能：

- 根据来源、页码、切片编号和正文生成 SHA-256 ID
- 分批调用 Embedding 接口
- 将向量、原文和元数据写入 Chroma
- 使用相同 ID 更新已有记录，减少重复切片
- 返回 Chroma 中的记录总数

#### `rag_vector_init.py`

向量知识库初始化入口。

按照以下顺序编排初始化流程：

```text
文档解析
   ↓
文本清洗
   ↓
文本切片
   ↓
Embedding 向量化
   ↓
写入 Chroma
```

运行命令：

```bash
python rag_vector_init.py
```

### 2.3 在线知识库问答模块

#### `rag_chat/__init__.py`

将 `rag_chat` 目录标记为 Python 包。

#### `rag_chat/query_vectorizer.py`

使用与建库时相同的 Embedding 模型，将用户问题转换为向量。

例如：

```text
XXX擅长哪些技术？
        ↓
[0.012, -0.083, 0.027, ..., 0.041]
```

建库和查询必须使用相同的 Embedding 模型，否则向量不在相同的语义空间内。

#### `rag_chat/chroma_retriever.py`

使用问题向量从 Chroma 中检索相关文本切片。

项目使用 MMR（Maximal Marginal Relevance）检索，默认配置：

```text
k = 4
fetch_k = 12
lambda_mult = 0.7
```

参数含义：

- `fetch_k`：首先从 Chroma 召回的候选切片数量
- `k`：最终返回给大语言模型的切片数量
- `lambda_mult`：相关性与结果多样性之间的平衡参数

MMR 可以减少多个检索结果内容高度重复的问题。

#### `rag_chat/context_formatter.py`

将检索结果整理成带编号的参考资料。

示例：

```text
[1]
来源：data/test_intro.pdf
位置：第 1 页
内容：
XXX是一名对人工智能和云计算充满热情的技术从业者……
```

整理后的信息包括：

- 引用编号
- 原始文件路径
- PDF 页码
- 文本切片正文

#### `rag_chat/prompt_builder.py`

将用户问题和检索到的参考资料构造成 Prompt。

Prompt 要求模型：

- 只能根据参考资料回答
- 不得使用资料之外的知识补充事实
- 资料不足时回答“根据当前知识库无法确定”
- 在关键结论后标注 `[1]`、`[2]` 等引用
- 不得编造人物、时间、数字或经历
- 使用简洁、清晰的中文回答

#### `rag_chat/ark_chat_model.py`

通过 OpenAI 兼容协议调用火山方舟 Agent Plan 对话模型。

主要配置：

```text
temperature = 0.1
timeout = 120
max_retries = 2
```

较低的 `temperature` 可以降低回答随机性，使知识库问答更加稳定。

#### `rag_chat/answer_generator.py`

负责：

1. 调用 `PromptBuilder` 构造 Prompt
2. 调用 `ArkChatModel` 请求大语言模型
3. 提取模型返回内容
4. 转换为最终字符串回答

#### `rag_chat_init.py`

在线知识库问答入口。

负责：

- 检查 Chroma 数据库是否存在
- 加载 Embedding 客户端
- 加载 Chroma
- 接收用户问题
- 执行问题向量化
- 执行 Chroma 检索
- 整理参考资料
- 构造 Prompt
- 调用火山方舟 LLM
- 展示回答和引用来源

运行命令：

```bash
python rag_chat_init.py
```

## 3. 2-Step RAG 工作流程

2-Step RAG 分为“向量知识库初始化”和“在线知识库问答”两个阶段。

### 3.1 第一阶段：向量知识库初始化

```text
PDF / TXT / Markdown
        ↓
document_processor.py
解析文档并提取原始文本
        ↓
text_cleaner.py
清洗文本并过滤空白内容
        ↓
text_chunker.py
拆分为带重叠的文本 Chunk
        ↓
ark_embedding.py
调用火山方舟 Embedding 生成向量
        ↓
vector_chroma_writer.py
将向量、正文、来源和页码写入 Chroma
        ↓
chroma_db/
```

执行入口：

```bash
python rag_vector_init.py
```

以下情况需要执行向量知识库初始化：

- 首次创建知识库
- 新增知识库文档
- 修改已有文档
- 修改文本切片参数
- 更换 Embedding 模型
- 重建 Chroma 数据库

建库和查询必须使用相同的 Embedding 模型及向量维度。

### 3.2 第二阶段：在线知识库问答

```text
用户问题
   ↓
query_vectorizer.py
使用方舟 Embedding 将问题转换为向量
   ↓
chroma_retriever.py
在 Chroma 中执行 MMR 相似度检索
   ↓
取得最相关的文本切片
   ↓
context_formatter.py
整理切片正文、来源、页码和引用编号
   ↓
prompt_builder.py
组合用户问题、回答规则和参考资料
   ↓
ark_chat_model.py
调用火山方舟 Agent Plan 对话模型
   ↓
answer_generator.py
生成严格基于知识库且带引用的回答
```

执行入口：

```bash
python rag_chat_init.py
```

这种架构称为 2-Step RAG，是因为一次问答包含两个核心步骤：

1. Retrieval：从知识库检索相关资料。
2. Generation：根据检索资料生成最终回答。

大语言模型不会直接访问 Chroma。Python 程序先从 Chroma 检索相关内容，再将内容放入 Prompt 交给大语言模型。

完整调用关系：

```text
rag_chat_init.py
    ↓
QueryVectorizer.vectorize()
    ↓
ChromaRetriever.retrieve()
    ↓
format_context()
    ↓
PromptBuilder.build()
    ↓
ArkChatModel.invoke()
    ↓
AnswerGenerator.generate()
    ↓
显示回答和来源
```

## 4. 使用样例

### 4.1 环境要求

运行环境要求：

- Linux
- Python 3.10 或更高版本
- 推荐使用 Python 3.12
- 已开通火山方舟 Agent Plan
- 已获取有效的 Agent Plan API Key
- 已配置 Agent Plan 支持的对话模型
- 服务器可以正常访问火山方舟 API

查看当前 Python 版本：

```bash
python --version
```

输出示例：

```text
Python 3.12.3
```

如果 Python 版本低于 3.10，建议升级后再创建虚拟环境。

### 4.2 创建虚拟环境

进入项目目录：

```bash
cd /home/RAG
```

创建虚拟环境：

```bash
python3 -m venv .venv
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

确认虚拟环境中的 Python 版本：

```bash
python --version
```

### 4.3 安装依赖

升级 pip：

```bash
pip install --upgrade pip
```

安装项目依赖：

```bash
pip install -r utils/requirements.txt
```

`utils/requirements.txt` 示例：

```text
langchain
langchain-core
langchain-openai
langchain-text-splitters
langchain-chroma
chromadb
pypdf
python-dotenv
```

### 4.4 配置环境变量

编辑配置文件：

```bash
vim utils/.env
```

配置内容：

```dotenv
ARK_API_KEY=你的Agent_Plan_API_Key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/plan/v3
ARK_EMBEDDING_MODEL=doubao-embedding-vision
ARK_CHAT_MODEL=你的Agent_Plan对话模型名称

RAG_DATA_DIR=data
RAG_CHROMA_DIR=chroma_db
RAG_COLLECTION_NAME=knowledge_base

RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=150
RAG_EMBEDDING_BATCH_SIZE=64

RAG_RETRIEVAL_TOP_K=4
RAG_RETRIEVAL_FETCH_K=12
RAG_RETRIEVAL_LAMBDA_MULT=0.7
```

不要将包含真实 API Key 的 `.env` 提交到 Git 仓库。

建议在 `.gitignore` 中加入：

```gitignore
.venv/
__pycache__/
*.pyc
utils/.env
chroma_db/
```

### 4.5 添加知识库文档

将文档放入 `data/` 目录：

```text
data/
└── test_intro.pdf
```

当前支持：

```text
.pdf
.txt
.md
```

### 4.6 检查代码

在 `/home/RAG` 目录下执行：

```bash
python -m py_compile \
  utils/*.py \
  vector_init/*.py \
  rag_chat/*.py \
  rag_vector_init.py \
  rag_chat_init.py
```

命令没有输出表示语法和基础导入检查通过。

### 4.7 初始化向量知识库

执行：

```bash
python rag_vector_init.py
```

成功输出示例：

```text
开始初始化 RAG 向量知识库
数据目录：/home/RAG/data

解析得到的原始文档单元：4
清洗后的有效文档单元：4
生成的有效文本切片：4

正在测试 Agent Plan Embedding 接口……
Base URL：https://ark.cn-beijing.volces.com/api/plan/v3
Embedding 模型：doubao-embedding-vision
Embedding 测试成功
向量维度：2048

开始生成向量并写入 Chroma……
正在写入切片 1～4，总数 4

RAG 向量知识库初始化完成
原始文档单元：4
有效文档单元：4
文本切片：4
向量维度：2048
Chroma 记录数：4
向量库位置：/home/RAG/chroma_db
```

### 4.8 启动知识库问答

执行：

```bash
python rag_chat_init.py
```

启动成功示例：

```text
正在启动 RAG 问答系统……
Chroma 加载成功，记录数：4

RAG 问答系统启动成功
Base URL：https://ark.cn-beijing.volces.com/api/plan/v3
Embedding 模型：doubao-embedding-vision
对话模型：当前配置的对话模型
向量库：/home/RAG/chroma_db
输入 exit、quit 或 q 退出。
```

问答示例：

```text
问题：XXX擅长哪些技术？

[1/5] 正在将问题转换为向量……
问题向量化完成，维度：2048
[2/5] 正在检索 Chroma……
检索完成，得到 4 个文本切片
[3/5] 正在整理参考资料……
[4/5] 正在构造 Prompt……
[5/5] 正在调用火山方舟模型……

回答：
XXX擅长大模型部署、分布式推理、KV Cache 优化、
云原生技术以及性能评测与调优。[1]

检索来源：
[1] data/test_intro.pdf，第 1 页
```

如果知识库资料不足，系统应回答：

```text
根据当前知识库无法确定。
```

输入以下任意命令退出：

```text
exit
quit
q
```

