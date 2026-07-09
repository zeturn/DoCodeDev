from __future__ import annotations

from formatter import format_user


def build_profile(user: dict[str, str]) -> dict[str, str]:
    formatted = format_user(user)
    return {"display_name": formatted["display_name"], "slug": formatted["slug"]}
