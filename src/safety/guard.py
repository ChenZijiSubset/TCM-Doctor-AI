def has_risk(text: str) -> bool:
    risks = ["胸痛", "昏迷", "呼吸困难", "大出血", "抽搐", "意识障碍", "剧烈腹痛"]
    return any(r in text for r in risks)