"""
RAG 文档问答系统 — FastAPI REST API

启动方式: uvicorn api:app --reload --port 8000
面试要点：将 RAG 系统包装为 REST API，使其可被任何前端/App/其他服务调用

API 端点:
  POST /upload   — 上传文档并入库
  POST /query    — 流式问答（SSE）
  GET  /stats    — 知识库统计
  DELETE /clear  — 清空知识库
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 确保 src 在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.pipeline import RAGPipeline

# ============================================
# 应用初始化
# ============================================
app = FastAPI(
    title="RAG 文档问答 API",
    description="基于检索增强生成的文档问答系统，支持 PDF/Word/TXT/MD",
    version="2.0.0",
)

# CORS — 允许任何来源访问（生产环境需限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 Pipeline 单例（首次使用时初始化）
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """获取全局 RAG 管道（延迟初始化，Singleton）"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            persist_dir="./chroma_db",
            collection_name="rag_documents",
        )
    return _pipeline


# ============================================
# 数据模型
# ============================================
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(4, ge=1, le=20, description="检索结果数量")
    api_key: str = Field(..., description="API Key（DeepSeek / OpenAI）")
    base_url: Optional[str] = Field(None, description="API Base URL")
    model: str = Field("deepseek-chat", description="模型名称")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class StatsResponse(BaseModel):
    total_chunks: int
    source_files: list[str]
    persist_dir: str


class UploadResponse(BaseModel):
    file_name: str
    text_length: int
    chunk_count: int
    stored_count: int
    total_in_store: int


# ============================================
# API 端点
# ============================================

@app.post("/upload", response_model=UploadResponse, tags=["文档管理"])
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档并处理入库

    - 支持格式: PDF, Word (.docx), TXT, Markdown (.md)
    - 流程: 解析 → 切分 → 向量化 → 存入 ChromaDB

    面试要点: FastAPI 的 UploadFile 支持流式接收大文件
    """
    # 校验文件类型
    allowed_ext = {".pdf", ".txt", ".md", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}。支持: {', '.join(allowed_ext)}",
        )

    # 保存到临时目录
    os.makedirs("./data", exist_ok=True)
    save_path = f"./data/{file.filename}"

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    # 处理入库
    try:
        pipeline = get_pipeline()
        result = pipeline.ingest(save_path)
        return UploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {e}")


@app.post("/query", tags=["问答"])
async def ask_question(req: QueryRequest):
    """
    向知识库提问（流式返回 SSE）

    返回格式: Server-Sent Events
      data: {"token": "文"}  — 增量文本
      data: {"sources": [...]} — 检索到的来源片段（最后一条）

    面试要点: SSE 流式输出 — 用户不用等完整答案，逐字出现
    """
    pipeline = get_pipeline()

    if pipeline.vector_store.get_count() == 0:
        raise HTTPException(status_code=400, detail="知识库为空，请先上传文档")

    async def generate():
        import json

        try:
            generator = pipeline.query(
                question=req.question,
                top_k=req.top_k,
                stream=True,
                api_key=req.api_key,
                base_url=req.base_url,
                model=req.model,
            )

            for item in generator:
                if isinstance(item, tuple) and item[0] == "__CONTEXT__":
                    # 最后的上下文信息
                    sources = [
                        {
                            "text": s["text"][:300],
                            "score": s["score"],
                            "source": s["metadata"].get("source", "未知"),
                        }
                        for s in item[1]
                    ]
                    yield f"data: {json.dumps({'sources': sources}, ensure_ascii=False)}\n\n"
                elif isinstance(item, str):
                    yield f"data: {json.dumps({'token': item}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


@app.get("/stats", response_model=StatsResponse, tags=["知识库"])
async def get_stats():
    """查看知识库统计信息"""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    return StatsResponse(**stats)


@app.delete("/clear", tags=["知识库"])
async def clear_knowledge_base():
    """清空知识库"""
    pipeline = get_pipeline()
    pipeline.clear_knowledge_base()
    return {"message": "知识库已清空", "total_chunks": 0}


# ============================================
# 健康检查
# ============================================
@app.get("/", tags=["系统"])
async def root():
    return {
        "service": "RAG 文档问答 API",
        "version": "2.0.0",
        "docs": "/docs",  # FastAPI 自动生成的 Swagger UI
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "stats": "GET /stats",
            "clear": "DELETE /clear",
        },
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 RAG 文档问答 API 服务...")
    print("   Swagger 文档: http://localhost:8000/docs")
    print("   API 入口:     http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
