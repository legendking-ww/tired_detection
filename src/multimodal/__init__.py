"""多模态驾驶安全：语音疲劳（Groq）与视觉疲劳融合（可选）。"""

from .audio_loop import start_audio_loop, stop_audio_loop
from .config import is_multimodal_enabled, is_multimodal_mic
from .fusion import alert_level, fuse_visual_audio, fuse_visual_audio_dynamic
from .state import get_last_audio_score, set_last_audio_score

__all__ = [
    "is_multimodal_enabled",
    "is_multimodal_mic",
    "start_audio_loop",
    "stop_audio_loop",
    "fuse_visual_audio",
    "fuse_visual_audio_dynamic",
    "alert_level",
    "get_last_audio_score",
    "set_last_audio_score",
]
