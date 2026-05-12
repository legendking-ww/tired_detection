"""跨线程共享：语音分、转写摘要、最近一次融合结果（供 UI 与画面叠字）。"""
import threading
from datetime import datetime
from typing import Optional, Tuple

_lock = threading.Lock()
_MAX_VOICE_LOG = 80
_voice_log: list[str] = []
_last_audio: Optional[float] = None
_last_error: Optional[str] = None
_last_transcript: str = ""
_last_visual: Optional[float] = None
_last_fused: Optional[float] = None
_last_level: str = ""
_agent_status: str = ""
_agent_summary: str = ""


def set_last_audio_score(value: Optional[float], err: Optional[str] = None) -> None:
    global _last_audio, _last_error
    with _lock:
        _last_audio = value
        _last_error = err


def get_last_audio_score() -> Tuple[Optional[float], Optional[str]]:
    with _lock:
        return _last_audio, _last_error


def set_last_transcript(text: str) -> None:
    global _last_transcript
    with _lock:
        _last_transcript = (text or "").strip()[:800]


def get_last_transcript() -> str:
    with _lock:
        return _last_transcript


def append_voice_log(kind: str, message: str) -> None:
    """供后台线程追加一行带时间戳的语音流水线（Whisper / 打分 / 麦克风错误等）。"""
    global _voice_log
    msg = (message or "").strip().replace("\n", " ")
    if len(msg) > 220:
        msg = msg[:217] + "…"
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {kind}: {msg}"
    with _lock:
        _voice_log.append(line)
        if len(_voice_log) > _MAX_VOICE_LOG:
            _voice_log = _voice_log[-_MAX_VOICE_LOG:]


def get_voice_log_text() -> str:
    with _lock:
        return "\n".join(_voice_log) if _voice_log else "（尚无语音轮次记录，开始检测后每轮转写与打分将出现在此处。）"


def set_last_fusion(visual: float, fused: float, level: str) -> None:
    global _last_visual, _last_fused, _last_level
    with _lock:
        _last_visual = float(visual)
        _last_fused = float(fused)
        _last_level = (level or "").strip()


def get_last_fusion() -> Tuple[Optional[float], Optional[float], str]:
    with _lock:
        return _last_visual, _last_fused, _last_level


def set_agent_status(text: str) -> None:
    """供 Agent 线程写入一行状态，主界面定时刷新显示。"""
    global _agent_status
    t = (text or "").strip().replace("\n", " ")
    if len(t) > 420:
        t = t[:417] + "…"
    with _lock:
        _agent_status = t


def get_agent_status() -> str:
    with _lock:
        return _agent_status


def set_agent_summary(text: str) -> None:
    """最近一次 Agent 运行结束后的多行文本（推理 + 执行摘要），供侧栏固定展示。"""
    global _agent_summary
    t = (text or "").strip()
    if len(t) > 900:
        t = t[:897] + "…"
    with _lock:
        _agent_summary = t


def get_agent_summary() -> str:
    with _lock:
        return _agent_summary


def clear_fusion_display() -> None:
    global _last_transcript, _last_visual, _last_fused, _last_level, _voice_log, _agent_status, _agent_summary
    with _lock:
        _last_transcript = ""
        _last_visual = None
        _last_fused = None
        _last_level = ""
        _voice_log.clear()
        _agent_status = ""
        _agent_summary = ""
