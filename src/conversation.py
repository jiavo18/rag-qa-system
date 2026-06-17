"""
对话管理器 (Conversation Manager)

功能：管理多轮对话的会话和消息记录
面试要点：
  - 为什么用 SQLite？→ 重启不丢数据，比内存 dict 可靠
  - session_id 用 UUID → 全局唯一，分布式友好
  - max_turns 限制 → 防止上下文过长超出 LLM 窗口
"""

import uuid
from datetime import datetime
from typing import List, Tuple, Optional
from .database import get_db


class ConversationManager:
    """多轮对话管理"""

    def create_session(self, title: str = "", user_id: int = None) -> str:
        """创建新对话，返回 session_id"""
        session_id = str(uuid.uuid4())[:8]  # 短 ID，易读
        db = get_db()
        db.execute(
            "INSERT INTO sessions (id, title, user_id, created_at) VALUES (?, ?, ?, ?)",
            (session_id, title, user_id, datetime.now().isoformat()),
        )
        db.commit()
        db.close()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """追加一条消息到对话"""
        db = get_db()
        db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )
        # 自动更新对话标题（取用户第一条消息的前20字）
        if role == "user":
            current_title = db.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if current_title and not current_title["title"]:
                title = content[:20] + ("..." if len(content) > 20 else "")
                db.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (title, session_id),
                )
        db.commit()
        db.close()

    def get_history(
        self, session_id: str, max_turns: int = 5
    ) -> List[Tuple[str, str]]:
        """
        获取最近 N 轮对话历史

        Args:
            session_id: 对话 ID
            max_turns: 最大轮数（1轮 = 1问1答 = 2条消息）

        Returns:
            [(role, content), ...]  例如 [("user", "..."), ("assistant", "...")]
        """
        db = get_db()
        limit = max_turns * 2  # 每轮2条消息
        rows = db.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        db.close()
        # 反转回时间顺序
        return [(r["role"], r["content"]) for r in reversed(rows)]

    def list_sessions(self, user_id: int = None) -> List[dict]:
        """列出对话（可按用户过滤）"""
        db = get_db()
        if user_id is not None:
            rows = db.execute(
                "SELECT id, title, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        db.close()
        return [{"id": r["id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]

    def delete_session(self, session_id: str):
        """删除一个对话及其所有消息"""
        db = get_db()
        db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        db.commit()
        db.close()


# 全局单例
_conversation_manager: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """获取全局对话管理器（单例）"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager


if __name__ == "__main__":
    cm = ConversationManager()

    # 测试创建对话
    sid = cm.create_session()
    print(f"✅ 创建对话: {sid}")

    # 测试添加消息
    cm.add_message(sid, "user", "什么是机器学习？")
    cm.add_message(sid, "assistant", "机器学习是人工智能的一个分支，使计算机能从数据中学习。")
    cm.add_message(sid, "user", "它跟深度学习什么关系？")

    # 测试获取历史
    history = cm.get_history(sid)
    print(f"✅ 对话历史 ({len(history)} 条消息):")
    for role, content in history:
        print(f"   [{role}] {content[:50]}...")

    # 测试列表
    sessions = cm.list_sessions()
    print(f"\n✅ 对话列表: {len(sessions)} 个")
    for s in sessions:
        print(f"   {s['id']}: {s['title']}")

    # 测试删除
    cm.delete_session(sid)
    print(f"\n✅ 已删除对话 {sid}")
    print(f"   剩余对话: {len(cm.list_sessions())} 个")
