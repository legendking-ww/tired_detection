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


def clear_fusion_display() -> None:
    global _last_transcript, _last_visual, _last_fused, _last_level, _voice_log
    with _lock:
        _last_transcript = ""
        _last_visual = None
        _last_fused = None
        _last_level = ""
        _voice_log.clear()
