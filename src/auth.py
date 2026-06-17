"""
用户认证模块 (Auth)

功能：注册、登录、JWT Token 验证
面试要点：
  - 为什么 bcrypt？→ 自带盐值 + 慢哈希，防彩虹表和暴力破解
  - 为什么 JWT？→ 无状态，服务端不用存 session，水平扩展友好
  - Token 过期 → 防止 token 泄漏后永久有效
"""

import jwt
import bcrypt
import os
from datetime import datetime, timedelta
from typing import Optional
from .database import get_db

# JWT 密钥（生产环境应放在环境变量中）
JWT_SECRET = os.getenv("JWT_SECRET", "rag-system-secret-key-change-in-production")
JWT_EXPIRE_DAYS = 7
JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """对密码进行 bcrypt 哈希"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(
        password.encode("utf-8"),
        hashed.encode("utf-8"),
    )


def create_token(user_id: int) -> str:
    """创建 JWT Token"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[int]:
    """
    验证 JWT Token，返回 user_id 或 None

    面试要点：异常处理——过期、无效签名等都要 catch
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        return None  # Token 过期
    except jwt.InvalidTokenError:
        return None  # Token 无效


def register_user(username: str, password: str) -> Optional[dict]:
    """
    注册新用户

    Returns:
        {"id": ..., "username": ..., "created_at": ...} 或 None（用户名已存在）
    """
    if len(username) < 3:
        raise ValueError("用户名至少 3 个字符")
    if len(password) < 4:
        raise ValueError("密码至少 4 个字符")

    db = get_db()
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        db.close()
        return None  # 用户名已存在

    hashed = hash_password(password)
    created_at = datetime.now().isoformat()
    cursor = db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, hashed, created_at),
    )
    user_id = cursor.lastrowid
    db.commit()
    db.close()
    return {"id": user_id, "username": username, "created_at": created_at}


def login_user(username: str, password: str) -> Optional[str]:
    """
    登录验证，成功返回 JWT Token

    Returns:
        JWT token 字符串 或 None（用户名或密码错误）
    """
    db = get_db()
    user = db.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    db.close()

    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None

    return create_token(user["id"])


def get_user_by_id(user_id: int) -> Optional[dict]:
    """根据 ID 获取用户信息"""
    db = get_db()
    user = db.execute(
        "SELECT id, username, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    db.close()
    if user:
        return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}
    return None


if __name__ == "__main__":
    # 测试注册/登录流程
    print("=== 用户系统测试 ===\n")

    # 注册
    user = register_user("alice", "password123")
    print(f"✅ 注册成功: {user}")

    # 重复注册
    dup = register_user("alice", "password123")
    print(f"   重复注册: {dup} (应为 None)")

    # 登录
    token = login_user("alice", "password123")
    print(f"✅ 登录成功 Token: {token[:30]}...")

    # 错误密码
    bad_login = login_user("alice", "wrong")
    print(f"   错误密码: {bad_login} (应为 None)")

    # 验证 Token
    user_id = verify_token(token)
    print(f"✅ Token 验证: user_id={user_id}")

    # 获取用户
    info = get_user_by_id(user_id)
    print(f"✅ 用户信息: {info}")

    # 清理测试数据
    db = get_db()
    db.execute("DELETE FROM users WHERE username = 'alice'")
    db.commit()
    db.close()
    print(f"\n✅ 测试完成，已清理")
