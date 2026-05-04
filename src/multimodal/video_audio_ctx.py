"""检测线程写入当前视频路径与播放位置，供语音线程用 ffmpeg 抽音轨。"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_video_path: str = ""
_pos_msec: float = 0.0


def set_playback(video_path: str, pos_msec: float) -> None:
    global _video_path, _pos_msec
    with _lock:
        _video_path = (video_path or "").strip()
        _pos_msec = max(0.0, float(pos_msec))


def get_playback() -> tuple[str, float]:
    with _lock:
        return _video_path, _pos_msec


def clear_playback() -> None:
    set_playback("", 0.0)
