# 📚 RAG 文档问答系统

> **Retrieval-Augmented Generation** — 基于检索增强生成的智能文档问答系统  
> 🌐 在线体验：**[jiavo18-rag--system.streamlit.app](https://jiavo18-rag--system.streamlit.app/)**

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

