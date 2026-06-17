# 📚 RAG 文档问答系统

> **Retrieval-Augmented Generation** — 基于检索增强生成的智能文档问答系统  
> 暑期实习项目 · 面试准备版

---

## 🎯 项目概述

这是一个**完整的 RAG（检索增强生成）系统**，用户可以上传 PDF/Word/TXT/Markdown 文档，然后对文档内容进行自然语言提问，系统会基于文档内容给出准确的回答并标注来源。

```
┌─────────────────────────────────────────────────────┐
│                    📚 RAG 系统架构                    │
│                                                      │
│  📥 文档上传 (Ingest)                                │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐         │
│  │ PDF  │→  │ 纯文本│→  │ 文本块│→  │ 向量  │        │
│  │ Word │   │      │   │(chunk)│   │(384维)│        │
│  │ TXT  │   │      │   │+ 元数据│   │      │        │
│  └──────┘   └──────┘   └──────┘   └───┬──┘         │
│                                        │             │
│                                  ┌─────▼──────┐      │
│                                  │  ChromaDB  │      │
│                                  │ (向量数据库) │      │
│                                  └─────┬──────┘      │
│                                        │             │
│  🔍 问答 (Query)                      │             │
│  ┌──────┐   ┌──────┐   ┌──────────┐  │             │
│  │ 问题 │→  │ 向量  │→  │ 语义检索  │◄─┘             │
│  └──────┘   └──────┘   └─────┬────┘                │
│                               │ Top-K 相关块         │
│                        ┌──────▼──────┐              │
│                        │   LLM 生成   │              │
│                        │ (OpenAI API) │              │
│                        └──────┬──────┘              │
│                               │ 答案 + 来源引用       │
│                        ┌──────▼──────┐              │
│                        │   用户界面   │              │
│                        │ (Streamlit) │              │
│                        └─────────────┘              │
└─────────────────────────────────────────────────────┘
```

## 🚀 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 OpenAI API Key
# OPENAI_API_KEY=sk-你的密钥
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

### Docker 一键部署

```bash
# 构建镜像
docker build -t rag-qa-system .

# 启动容器
docker run -p 8000:8000 rag-qa-system
```

访问 `http://localhost:8000/docs` 查看 API 文档。

### 4. 使用流程

1. 在左侧边栏输入 OpenAI API Key
2. 上传你的文档（PDF/Word/TXT/Markdown）
3. 点击「处理文档」
4. 在对话框输入问题，等待回答

---

## 📁 项目结构

```
rag-project/
├── README.md                ← 项目文档 + 面试准备
├── requirements.txt         ← Python 依赖
├── .env.example             ← 环境变量模板
├── app.py                   ← Streamlit Web UI 入口
├── data/                    ← 上传文档临时存储
├── chroma_db/               ← 向量数据库持久化文件
└── src/                     ← 核心代码
    ├── loader.py            ← ① 文档加载器
    ├── chunker.py           ← ② 文本切分器
    ├── embedder.py          ← ③ 向量化模块
    ├── vector_store.py      ← ④ 向量数据库
    ├── retriever.py         ← ⑤ 检索模块
    ├── generator.py         ← ⑥ LLM 生成模块
    └── pipeline.py          ← ⑦ 完整管道编排
```

---

## 🔧 技术栈

| 层级 | 技术 | 选择理由 |
|------|------|----------|
| **文档加载** | PyPDF2 / python-docx | 纯 Python，无需系统依赖 |
| **文本切分** | LangChain RecursiveCharacterTextSplitter | 递归按分隔符切分，保持语义完整性 |
| **向量化** | sentence-transformers `all-MiniLM-L6-v2` | 免费本地运行，384维，速度快（~80MB） |
| **向量库** | ChromaDB | 轻量级，Python 原生，自带持久化 |
| **LLM** | OpenAI GPT-3.5/4 | 效果最好，API 简洁 |
| **UI** | Streamlit | 纯 Python，快速构建 Web 界面 |

---

## 🎤 面试准备指南

### 核心概念

#### Q1: 什么是 RAG？为什么要用 RAG？

**回答模板：**

> RAG（Retrieval-Augmented Generation）是一种结合**信息检索**和**文本生成**的 AI 架构。它的核心思想是：在让大语言模型回答问题之前，先从外部知识库中检索相关信息，作为上下文一起喂给模型。
>
> **为什么需要 RAG？**
> 1. **解决幻觉问题**：LLM 会编造不存在的信息，RAG 强制它基于真实文档回答
> 2. **知识更新**：LLM 训练数据有截止日期，RAG 可以检索最新文档
> 3. **私有知识**：企业内部文档无法被公开 LLM 训练，RAG 可以接入私有知识库
> 4. **可追溯性**：每个答案都能追溯到原文出处，便于验证

#### Q2: RAG 的核心流程是什么？

> 分为两个阶段：
>
> **离线阶段（Ingest 文档上传）**
> 1. **加载** — 解析 PDF/Word 等格式 → 纯文本
> 2. **切分** — 将长文本按语义边界切成小块（chunk）
> 3. **向量化** — 用 embedding 模型将每个块转成向量（384维的数字数组）
> 4. **存储** — 将文本块和向量存入 ChromaDB
>
> **在线阶段（Query 问答）**
> 1. **向量化** — 将用户问题也转成向量
> 2. **检索** — 在向量库中找 top-K 个最相似的文档块（余弦相似度）
> 3. **生成** — 将检索到的块 + 问题一起发送给 LLM，生成答案
> 4. **引用** — 在答案中标注信息来源

#### Q3: 文本切分的 chunk_size 怎么选？overlap 为什么重要？

> **chunk_size = 500 字符**（本项目选择）
> - 太小（100字）：丢失上下文，检索到的片段不完整
> - 太大（2000字）：噪声多，检索精度下降，超出 LLM 上下文
> - 500 是经验值，适合中文段落（约 300-500 字/段）
>
> **chunk_overlap = 50 字符**
> - 防止关键信息落在两个块的边界被切断
> - 比如一句话的前半句在块1末尾，后半句在块2开头，有 overlap 就能被覆盖

#### Q4: 为什么 embedding 用 sentence-transformers 而不是 OpenAI API？

> - **免费**：本地运行，不需要付费
> - **低延迟**：不走网络，毫秒级响应
> - **隐私**：数据不出本地
> - **效果**：`all-MiniLM-L6-v2` 是公认的性价比之王，384维平衡了精度和速度
>
> **面试加分**：可以补充说"生产环境中会用 OpenAI `text-embedding-3-small` 或 `text-embedding-3-large`，因为它们支持更高的维度（256-3072），效果更好，且价格很低（$0.02/1M tokens）"

#### Q5: 为什么不直接用关键词搜索，要用向量检索？

> **关键词搜索的局限**：
> - 查"怎么学习 Python"找不到"Python 学习方法"
> - 查"天气"找不到"气候"
> - 多语言场景完全失效
>
> **向量检索的优势**：
> - 理解语义：相似含义的文本在向量空间中距离近
> - 跨语言：中文问题和英文文档也能匹配
> - 模糊匹配：拼写错误也能找到相关内容

#### Q6: ChromaDB vs FAISS vs Milvus 怎么选？

| | ChromaDB | FAISS | Milvus |
|---|---|---|---|
| 类型 | 向量数据库 | 向量索引库 | 分布式向量数据库 |
| 部署 | pip install | pip install | Docker/K8s |
| 元数据 | ✅ 内置 | ❌ 需额外管理 | ✅ 内置 |
| 适用 | 原型/小规模 | 研究/中等规模 | 生产/大规模 |
| 本项目选择 | ✅ | | |

#### Q7: 如果检索结果不相关怎么办？

> 回答思路（展示解决问题的思维）：
> 1. **调整 chunk_size**：太大或太小都影响检索精度
> 2. **引入 rerank**：在检索后用 Cross-Encoder 模型对结果重新排序
> 3. **混合检索**：BM25（关键词）+ 向量检索，取并集
> 4. **优化 embedding 模型**：换更强的模型如 `bge-large-zh`
> 5. **Query 改写**：用 LLM 将用户问题改写成更适合检索的形式
> 6. **提高 top_k**：增加候选数量

#### Q8: 如何评估 RAG 系统的效果？

> 三个维度：
> 1. **检索质量**：Recall@K（正确答案在前K个结果中的比例）、MRR
> 2. **生成质量**：答案是否忠实于上下文（Faithfulness）、是否回答正确（Correctness）
> 3. **端到端**：用户满意度、回答相关性
>
> 常用工具：RAGAS 框架、LangSmith

### 技术亮点（面试必说）

1. **模块化设计**：6 个独立模块，各司其职，遵循单一职责原则
2. **Pipeline 编排**：将分散模块组合成完整业务流程，易于扩展
3. **单例模式**：Embedding 模型全局只加载一次（节省内存和启动时间）
4. **流式输出**：LLM 回答逐字出现，提升用户体验
5. **来源引用**：每个回答都标注出处，可追溯可验证
6. **混合方案**：本地免费 embedding + API LLM，兼顾成本和效果

### 面试常见追问

**"如果让你改进这个项目，你会怎么做？"**
- 加入 **Hybrid Search**（BM25 + 向量）
- 加入 **Cross-Encoder Rerank**
- 支持**多轮对话**（带历史记录的 RAG）
- 加入**评测框架**（RAGAS）
- **多模态 RAG**：支持图片、表格
- 用 **Agent 模式**：让 LLM 自主判断是否需要检索

**"这个系统能处理多大的文档量？"**
- 当前 ChromaDB 单机可处理 10 万+ 块
- 更大规模可迁移到 Milvus 或 Pinecone
- 可通过分片、索引优化进一步提升

---

## 🧪 测试

### 命令行测试（不需要 UI）

```bash
# 测试文档加载
python -m src.loader

# 测试文本切分
python -m src.chunker

# 测试向量化
python -m src.embedder

# 一次下载完 embedding 模型（约80MB）

# 测试完整管道（不需要 API Key）
python -m src.pipeline
```

---

## 📋 依赖说明

```
PyPDF2              — PDF 解析
python-docx         — Word 文档解析
langchain-text-splitters — 文本切分
sentence-transformers    — 本地 embedding（免费）
chromadb            — 向量数据库
openai              — LLM API 调用
streamlit           — Web UI
python-dotenv       — 环境变量
tiktoken            — Token 计数
```

---

## 💡 扩展方向

- [ ] 多轮对话支持
- [ ] Hybrid Search（BM25 + 向量）
- [ ] Cross-Encoder Reranking
- [ ] RAGAS 评测集成
- [ ] 多模态支持（图片 OCR + 表格解析）
- [ ] Docker 部署
- [ ] REST API（FastAPI）

---

**Built for Internship Prep · Summer 2026**
