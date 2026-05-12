"""解析 LLM 计划并执行本地动作（TTS、打开地图等），在后台线程中运行。"""
from __future__ import annotations

import threading
import urllib.parse
import webbrowser
from typing import Any, Callable, Dict, List, Optional

from . import state as mm_state
from .config import is_agent_local_tts_enabled
from .llm_agent import normalize_actions, request_agent_plan

LogFn = Callable[[str], None]

_ALLOWED_MAP_HOSTS = frozenset(
    {
        "map.baidu.com",
        "www.amap.com",
        "amap.com",
        "ditu.amap.com",
        "maps.google.com",
        "www.google.com",
        "google.com",
    }
)


def _host_allowed(host: str) -> bool:
    h = (host or "").lower().strip(".")
    if not h:
        return False
    if h in _ALLOWED_MAP_HOSTS:
        return True
    for suf in (".baidu.com", ".amap.com", ".google.com"):
        if h.endswith(suf):
            return True
    return False


def _safe_https_url(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u.lower().startswith("https://"):
        return None
    try:
        p = urllib.parse.urlparse(u)
    except Exception:
        return None
    if p.scheme != "https" or not p.netloc:
        return None
    host = p.netloc.split("@")[-1].split(":")[0].lower()
    if not _host_allowed(host):
        return None
    return u


def _filter_actions(actions: List[dict]) -> List[dict]:
    out: List[dict] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        t = (a.get("type") or "").strip().lower()
        if t == "speak" and not is_agent_local_tts_enabled():
            continue
        out.append(a)
    return out


def _speak_worker(text: str, log: Optional[LogFn]) -> None:
    t = (text or "").strip()[:400]
    if not t:
        return
    try:
        import pyttsx3  # noqa: F401

        eng = pyttsx3.init()
        try:
            eng.setProperty("rate", 172)
        except Exception:
            pass
        eng.say(t)
        eng.runAndWait()
    except ImportError:
        if log:
            log("[Agent] 未安装 pyttsx3：无法朗读。`pip install pyttsx3` 后重启；播报文案已写入流水。")
        mm_state.append_voice_log("Agent", "未安装 pyttsx3，跳过朗读")
    except Exception as e:
        if log:
            log(f"[Agent] 语音播报异常：{e!s}")
        mm_state.append_voice_log("Agent", f"播报异常：{e!s}"[:120])


def _run_action(act: dict, log: LogFn, notes: List[str]) -> None:
    typ = (act.get("type") or "").strip().lower()
    if typ == "notify":
        txt = act.get("text")
        if isinstance(txt, str) and txt.strip():
            t = txt.strip()[:400]
            log(f"[Agent·提示] {t}")
            mm_state.append_voice_log("Agent·提示", t[:200])
            notes.append(f"· 界面提示：{t}")
        return
    if typ == "speak":
        txt = act.get("text")
        if isinstance(txt, str) and txt.strip():
            t = txt.strip()[:200]
            log(f"[Agent·播报文案] {t}")
            mm_state.append_voice_log("Agent·播报文案", t)
            notes.append(f"· 播报：{t}")
            if is_agent_local_tts_enabled():
                threading.Thread(target=_speak_worker, args=(txt, log), daemon=True).start()
            else:
                log("[Agent] 已跳过本地语音（TIRED_AGENT_LOCAL_TTS=0）。")
                mm_state.append_voice_log("Agent", "未朗读（已关闭本地 TTS）")
        return
    if typ == "open_url":
        url = act.get("url")
        if not isinstance(url, str):
            return
        safe = _safe_https_url(url)
        if not safe:
            log(f"[Agent] 已忽略不安全的 URL：{url[:120]}")
            notes.append("· 已忽略不安全链接")
            return
        log(f"[Agent] 打开链接：{safe}")
        mm_state.append_voice_log("Agent", "打开 " + safe[:120])
        notes.append(f"· 已打开链接：{safe[:80]}")
        threading.Thread(target=webbrowser.open, args=(safe,), kwargs={"new": 2}, daemon=True).start()
        return
    if typ == "search_maps":
        q = act.get("query")
        if not isinstance(q, str) or not q.strip():
            q = "附近服务区"
        q = q.strip()[:80]
        enc = urllib.parse.quote(q, safe="")
        baidu = f"https://map.baidu.com/search/{enc}"
        log(f"[Agent] 地图搜索：{q}")
        mm_state.append_voice_log("Agent", "地图 " + q)
        notes.append(f"· 地图搜索：{q}")
        threading.Thread(target=webbrowser.open, args=(baidu,), kwargs={"new": 2}, daemon=True).start()
        return
    log(f"[Agent] 忽略未知动作类型：{typ}")
    notes.append(f"· 忽略未知动作：{typ}")


def execute_plan(actions: List[dict], log: LogFn) -> List[str]:
    notes: List[str] = []
    for act in actions:
        if isinstance(act, dict):
            _run_action(act, log, notes)
    return notes


def _default_map_fallback(log: LogFn) -> List[str]:
    q = "附近服务区"
    enc = urllib.parse.quote(q, safe="")
    baidu = f"https://map.baidu.com/search/{enc}"
    log(f"[Agent] 无可用动作，默认打开地图：{q}")
    mm_state.append_voice_log("Agent", "默认地图 " + q)
    threading.Thread(target=webbrowser.open, args=(baidu,), kwargs={"new": 2}, daemon=True).start()
    return [f"· 已默认打开地图：{q}"]


def run_llm_fatigue_agent(ctx: Dict[str, Any], log: LogFn) -> None:
    mm_state.set_agent_summary("")
    mm_state.set_agent_status("正在请求 LLM 生成行动计划…")
    plan, err = request_agent_plan(ctx)
    if err:
        log(f"[Agent] LLM 计划失败：{err}")
        mm_state.append_voice_log("Agent", err[:200])
        mm_state.set_agent_status("")
        mm_state.set_agent_summary(f"失败：{err[:420]}")
        return

    rz = plan.get("reasoning_zh")
    reasoning = rz.strip() if isinstance(rz, str) and rz.strip() else ""

    actions = _filter_actions(normalize_actions(plan))
    if not actions:
        log("[Agent] 模型未返回可执行动作（或仅剩已过滤的 speak），尝试默认地图。")
        mm_state.append_voice_log("Agent", "无动作，尝试默认地图")
        fb_notes = _default_map_fallback(log)
        parts: List[str] = []
        if reasoning:
            parts.append(f"推理：{reasoning}")
        parts.extend(fb_notes)
        parts.append("（若仍无反应请检查浏览器是否为默认程序）")
        mm_state.set_agent_summary("\n".join(parts))
        mm_state.set_agent_status("")
        return

    mm_state.set_agent_status("正在执行动作（地图 / 链接 / 播报）…")
    exec_notes = execute_plan(actions, log)

    parts2: List[str] = []
    if reasoning:
        parts2.append(f"推理：{reasoning}")
    parts2.extend(exec_notes)
    if any(("链接" in n) or ("地图" in n) for n in exec_notes):
        parts2.append("（若未自动打开浏览器，请检查系统默认浏览器设置。）")
    mm_state.set_agent_summary("\n".join(parts2)[:900])
    mm_state.set_agent_status("")


def schedule_llm_fatigue_agent(ctx: Dict[str, Any], log: LogFn) -> None:
    threading.Thread(target=run_llm_fatigue_agent, args=(ctx, log), daemon=True).start()
