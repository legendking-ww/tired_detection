"""危险疲劳时：调用与多模态相同的 OpenAI 兼容 Chat API，让 LLM 输出 JSON 行动计划。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    groq_api_key,
    groq_chat_model,
    is_agent_local_tts_enabled,
    llm_agent_chat_model,
    llm_agent_max_tokens,
    llm_agent_timeout,
)
from .groq_audio import _endpoints, _http_retry_post

_SYSTEM = (
    "你是车载主动安全 Agent 的决策模块。根据用户给出的「当前检测状态」JSON，"
    "输出**仅包含一个 JSON 对象**的应答，不要 Markdown、不要代码围栏、不要任何 JSON 以外的文字。\n"
    "JSON 结构必须为：\n"
    '{"reasoning_zh":"一两句中文说明决策依据","actions":[动作,...]}\n'
    "每个动作是对象，字段 type 只能是以下之一（可组合 2~5 个，按顺序执行）：\n"
    '1) {"type":"notify","text":"不超过120字，给驾驶员看的醒目文字提醒"}\n'
    '2) {"type":"speak","text":"中文播报稿，不超过80字；仅在用户环境允许本地 TTS 时有效"}\n'
    '3) {"type":"open_url","url":"https://... 必须是 https，且主机名为常见地图站"}\n'
    '4) {"type":"search_maps","query":"地图搜索词，如附近服务区、加油站、医院急诊"}\n'
    "原则：危险疲劳时**至少**包含 notify 或 search_maps/open_url 之一，让用户无需只听语音也能感知。"
    "优先 notify 给出明确休息建议，再用 search_maps 找休息点。"
    "不要输出危险协议（禁止 file:// javascript: data:）。"
)

_NO_SPEAK_TAIL = (
    "\n补充：当前运行环境已关闭本地语音播报，**禁止**输出 {\"type\":\"speak\"}；"
    "请用 notify 写清提醒，并至少包含 search_maps 或 open_url 之一。"
)


def _extract_json_object(raw: str) -> Optional[dict]:
    s = (raw or "").strip()
    if not s:
        return None
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def request_agent_plan(context: Dict[str, Any]) -> Tuple[Optional[dict], Optional[str]]:
    """调用 LLM 返回计划 dict；失败返回 (None, err_msg)。"""
    key = groq_api_key()
    if not key:
        return None, "未配置 API 密钥，无法运行 LLM Agent"
    _, c_url = _endpoints()
    model = llm_agent_chat_model()
    user_blob = json.dumps(context, ensure_ascii=False, indent=2)
    system = _SYSTEM + (_NO_SPEAK_TAIL if not is_agent_local_tts_enabled() else "")
    try:
        r = _http_retry_post(
            c_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "当前状态：\n" + user_blob},
                ],
                "temperature": 0.2,
                "max_tokens": llm_agent_max_tokens(),
            },
            timeout=llm_agent_timeout(),
        )
        if r.status_code != 200:
            return None, f"Agent Chat HTTP {r.status_code}: {r.text[:400]}"
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            return None, "模型返回内容非文本"
        plan = _extract_json_object(content)
        if plan is None:
            return None, "无法解析模型输出为 JSON"
        if "actions" in plan and not isinstance(plan["actions"], list):
            return None, "JSON 中 actions 必须为数组"
        return plan, None
    except Exception as e:
        return None, repr(e)


def normalize_actions(plan: dict) -> List[dict]:
    acts = plan.get("actions")
    if not isinstance(acts, list):
        return []
    out: List[dict] = []
    for a in acts:
        if isinstance(a, dict) and isinstance(a.get("type"), str):
            out.append(a)
    return out[:8]
