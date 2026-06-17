"""
RAG 文档问答系统 — Streamlit UI

启动方式: streamlit run app.py

面试要点：Streamlit 的特点
- 纯 Python，无需前端 HTML/CSS/JS
- 适合快速原型和数据演示
- 每个用户交互会重新运行整个脚本（st.session_state 保持状态）
"""

import streamlit as st
import os
import tempfile
import sys
from pathlib import Path

# 确保 src 目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline
from src.loader import load_document
from src.chunker import chunk_text

# ============================================
# 页面配置
# ============================================
st.set_page_config(
    page_title="RAG 文档问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# 样式
# ============================================
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .source-score {
        color: #4CAF50;
        font-weight: bold;
    }
    .answer-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 初始化 Session State
# ============================================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []



# ============================================
# 侧边栏
# ============================================
with st.sidebar:
    st.title("⚙️ 配置")

    # --- API 配置 ---
    st.header("🔑 API 设置")

    # 尝试从环境变量读取
    import dotenv
    dotenv.load_dotenv(override=True)

    default_key = os.getenv("OPENAI_API_KEY", "")
    default_base = os.getenv("OPENAI_BASE_URL", "")

    api_key = st.text_input(
        "OpenAI API Key",
        value=default_key,
        type="password",
        placeholder="sk-...",
        help="从 platform.deepseek.com/api_keys 获取（或其他兼容 API）",
    )
    base_url = st.text_input(
        "API Base URL（可选）",
        value=default_base,
        placeholder="留空使用默认地址",
        help="使用代理或兼容 API 时填写",
    )

    # 每次都同步 API 配置到 session state（修复：之前只在 init 时保存，改了 Key 不会更新）
    if api_key:
        st.session_state.api_key = api_key
    if base_url:
        st.session_state.base_url = base_url
    else:
        st.session_state.base_url = "https://api.deepseek.com/v1"  # 默认 DeepSeek
    st.session_state.model = "deepseek-chat"

    # 首次使用时初始化管道
    if not st.session_state.pipeline:
        st.session_state.pipeline = RAGPipeline(persist_dir="./chroma_db")

    st.divider()

    # --- 文档上传 ---
    st.header("📤 上传文档")

    uploaded_files = st.file_uploader(
        "选择文档（PDF / Word / TXT / Markdown）",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="支持 PDF、Word (.docx)、纯文本、Markdown 格式",
    )

    if uploaded_files and st.button("🚀 处理文档", type="primary", use_container_width=True):
        if not api_key:
            st.error("请先在左侧输入 API Key")
        else:
            with st.spinner("正在处理文档..."):
                for uploaded_file in uploaded_files:
                    # 保存临时文件
                    suffix = Path(uploaded_file.name).suffix
                    os.makedirs("./data", exist_ok=True)  # 确保目录存在
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix, dir="./data"
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    try:
                        result = st.session_state.pipeline.ingest(tmp_path)
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.session_state.kb_ready = True

                        st.success(
                            f"✅ {uploaded_file.name}\n"
                            f"→ {result['chunk_count']} 个文本块已入库"
                        )
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")

    st.divider()

    # --- 知识库状态 ---
    st.header("📊 知识库状态")

    if st.session_state.pipeline and st.session_state.pipeline.vector_store.get_count() > 0:
        stats = st.session_state.pipeline.get_stats()
        st.metric("文档块总数", stats["total_chunks"])
        st.metric("来源文件数", len(stats["source_files"]))

        if stats["source_files"]:
            st.write("**已加载文件：**")
            for f in stats["source_files"]:
                st.write(f"  📄 {f}")

        # 清空按钮
        if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
            st.session_state.pipeline.clear_knowledge_base()
            st.session_state.messages = []
            st.session_state.kb_ready = False
            st.session_state.ingested_files = []
            st.rerun()
    else:
        st.info("知识库为空，请上传文档")

    st.divider()

    # --- 检索设置 ---
    st.header("🔍 检索设置")
    top_k = st.slider(
        "检索结果数量 (top_k)",
        min_value=1,
        max_value=10,
        value=4,
        help="每次从知识库中取回的文档片段数量。值越大上下文越丰富，但也可能引入噪声"
    )

    st.divider()

# ============================================
# 主区域 — 对话界面
# ============================================
st.title("📚 RAG 文档问答系统")
st.caption("上传文档 → 提问 → 获得基于文档的智能回答")

# 知识库状态提示
if not st.session_state.kb_ready:
    st.info("👈 请先在左侧上传文档，然后开始提问")

st.divider()

# --- 对话历史 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 显示来源引用
        if message.get("sources"):
            with st.expander("📚 查看来源"):
                for i, src in enumerate(message["sources"]):
                    score = src.get("score", 0)
                    score_color = (
                        "🟢" if score is None or score < 0.5
                        else "🟡" if score < 1.0
                        else "🟠"
                    )
                    st.markdown(
                        f"""<div class="source-box">
                        {score_color} <span class="source-score">相似度: {score:.4f}</span> |
                        来源: {src['metadata'].get('source', '未知')} |
                        块 #{src['metadata'].get('chunk_index', '?')}
                        <br>{src['text'][:200]}...
                        </div>""",
                        unsafe_allow_html=True,
                    )

# --- 输入框 ---
if question := st.chat_input(
    "输入你的问题...",
    disabled=not st.session_state.kb_ready,
):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # 生成回答
    with st.chat_message("assistant"):
        if not st.session_state.pipeline:
            st.error("请先初始化管道")
        elif not st.session_state.get("api_key"):
            st.error("请设置 API Key")
        else:
            message_placeholder = st.empty()
            full_response = ""
            context_chunks = []

            # 流式生成（面试加分项）
            try:
                generator = st.session_state.pipeline.query(
                    question=question,
                    top_k=top_k,
                    stream=True,
                    api_key=st.session_state.get("api_key"),
                    base_url=st.session_state.get("base_url"),
                    model=st.session_state.get("model", "deepseek-chat"),
                )

                for item in generator:
                    if isinstance(item, tuple) and item[0] == "__CONTEXT__":
                        # 这是上下文信息
                        context_chunks = item[1]
                    elif isinstance(item, str):
                        full_response += item
                        message_placeholder.markdown(full_response + "▌")

                # 显示最终结果
                message_placeholder.markdown(full_response)

                # 显示来源
                if context_chunks:
                    with st.expander("📚 查看检索到的来源片段"):
                        for i, src in enumerate(context_chunks):
                            st.markdown(f"**片段 {i+1}** (相似度: {src['score']:.4f})")
                            st.text(src["text"][:300])
                            st.divider()

            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                full_response = f"❌ 错误: {str(e)}"

            # 保存对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": context_chunks,
            })

# ============================================
# 底部
# ============================================
st.divider()
st.caption(
    "Built with ❤️ using Streamlit · "
    "Embedding: sentence-transformers (free) · "
    "LLM: OpenAI API · "
    "Vector DB: ChromaDB"
)
