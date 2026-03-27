from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

USERS_FILE = Path("data/users.json")


def load_users() -> Dict[str, Any]:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(db: Dict[str, Any]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_or_create_user(db: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    if user_id not in db:
        db[user_id] = {
            "basic_info": {
                "name": "匿名",
                "age": None,
                "gender": None,
            },
            "current_state": {
                "syndrome": "未知",
                "confidence": 0.0,
                "last_update": "",
            },
            "history": [],
            "trajectory": [],
        }
    return db[user_id]


def update_user_record(
    db: Dict[str, Any],
    user_id: str,
    user_features: Dict[str, List[str]],
    top_result: Any,
) -> None:
    user = get_or_create_user(db, user_id)
    today = str(datetime.now().date())

    record = {
        "date": today,
        "symptoms": user_features.get("symptoms", []),
        "tongue": user_features.get("tongue", []),
        "pulse": user_features.get("pulse", []),
        "diagnosis": getattr(top_result, "name", ""),
        "score": getattr(top_result, "score", 0.0),
        "treatment": getattr(top_result, "treatment", ""),
        "formulas": getattr(top_result, "formulas", []),
    }
    user["history"].append(record)

    old_state = user.get("current_state", {}).get("syndrome", "未知")
    new_state = getattr(top_result, "name", "")

    if old_state != new_state and old_state != "未知":
        user["trajectory"].append(
            {
                "from": old_state,
                "to": new_state,
                "date": today,
            }
        )

    user["current_state"] = {
        "syndrome": new_state,
        "confidence": getattr(top_result, "score", 0.0),
        "last_update": today,
    }


def build_user_summary(user: Dict[str, Any]) -> str:
    current_state = user.get("current_state", {})
    history = user.get("history", [])[-5:]
    trajectory = user.get("trajectory", [])[-5:]

    history_lines = []
    for h in history:
        history_lines.append(
            f"{h.get('date', '')}：{h.get('diagnosis', '')}（{', '.join(h.get('symptoms', []))}）"
        )

    trajectory_lines = []
    for t in trajectory:
        trajectory_lines.append(
            f"{t.get('date', '')}：{t.get('from', '')} -> {t.get('to', '')}"
        )

    return f"""当前状态：{current_state.get('syndrome', '未知')}，置信度 {current_state.get('confidence', 0)}

最近记录：
{chr(10).join(history_lines) if history_lines else '无'}

状态变化：
{chr(10).join(trajectory_lines) if trajectory_lines else '无'}
"""