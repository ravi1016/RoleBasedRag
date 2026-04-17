ROLE_MAP = {
    "hr": ["hr", "general"],
    "finance": ["finance", "general"],
    "marketing": ["marketing", "general"],
    "engineering": ["engineering", "general"],
    "exec": ["hr", "finance", "marketing", "engineering", "general"]
}

def get_allowed_depts(role: str):
    return ROLE_MAP.get(role, ["general"])