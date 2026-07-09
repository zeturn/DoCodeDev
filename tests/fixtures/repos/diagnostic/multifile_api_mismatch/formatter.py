from __future__ import annotations


def format_user(user: dict[str, str]) -> str:
    return f"{user['first']} {user['last']}"
