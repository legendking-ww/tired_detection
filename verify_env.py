"""Check that .env next to this script loads (does not print full secrets)."""
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Run: pip install python-dotenv")
    raise SystemExit(1)

import os

root = Path(__file__).resolve().parent
path = root / ".env"
print("dotenv path:", path)
print("exists:", path.is_file())
if not path.is_file():
    print("Create .env next to main.py in project root.")
    raise SystemExit(2)

ok = load_dotenv(path, encoding="utf-8-sig")
print("load_dotenv ok:", ok)
for k in (
    "SILICONFLOW_API_KEY",
    "MULTIMODAL_API_KEY",
    "GROQ_API_KEY",
    "TIRED_MULTIMODAL",
    "GROQ_API_BASE",
):
    v = os.getenv(k, "")
    st = v.strip()
    if "KEY" in k:
        hint = f"len={len(st)} prefix={st[:7]}..." if len(st) > 8 else ("(empty)" if not st else "set(short)")
    else:
        hint = (st[:60] + "...") if len(st) > 60 else (st if st else "(empty)")
    print(f"  {k}: {hint}")
