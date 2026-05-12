"""多模态相关环境变量（不把 API Key 写进代码）。"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def is_multimodal_enabled() -> bool:
    return os.environ.get("TIRED_MULTIMODAL", "").strip().lower() in ("1", "true", "yes")


def _strip_bearer_prefix(token: str) -> str:
    s = (token or "").strip()
    if s.lower().startswith("bearer "):
        return s[7:].strip()
    return s


def _env_value_clean(name: str) -> str:
    """读取环境变量并去掉首尾空白、UTF-8 BOM、零宽字符（常见于 .env 被记事本编辑后）。"""
    raw = os.environ.get(name)
    if raw is None:
        return ""
    s = str(raw).strip().strip("\ufeff\u200b")
    return s


def groq_api_key() -> str:
    """Bearer Token。依次读取 ``MULTIMODAL_API_KEY``、``SILICONFLOW_API_KEY``、``GROQ_API_KEY``（与 .env 配合）。"""
    for name in ("MULTIMODAL_API_KEY", "SILICONFLOW_API_KEY", "GROQ_API_KEY"):
        v = _env_value_clean(name)
        if v:
            return _strip_bearer_prefix(v)
    return ""


def groq_api_base() -> str:
    """OpenAI 兼容 API 根路径，默认 Groq；可改为代理或其它厂商网关。"""
    b = os.environ.get("GROQ_API_BASE", "").strip().rstrip("/")
    if not b:
        return "https://api.groq.com/openai/v1"
    return b


def groq_whisper_model() -> str:
    return os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3").strip() or "whisper-large-v3"


def groq_chat_model() -> str:
    return os.environ.get("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"


def is_multimodal_mic() -> bool:
    """为真时从麦克风录制 WAV，不再轮询 TIRED_MULTIMODAL_WAV 固定文件。"""
    return os.environ.get("TIRED_MULTIMODAL_MIC", "").strip().lower() in ("1", "true", "yes")


def mic_record_sec() -> float:
    """单次麦克风录制时长（秒），默认 10。"""
    return float(os.environ.get("TIRED_MULTIMODAL_RECORD_SEC", "10"))


def multimodal_wav_path() -> str:
    """每轮分析的音频文件；默认可放 demo wav。"""
    p = os.environ.get("TIRED_MULTIMODAL_WAV", "").strip()
    if p and os.path.isfile(p):
        return p
    default = os.path.join(PROJECT_ROOT, "resources", "samples", "driver_demo.wav")
    if os.path.isfile(default):
        return default
    return ""


def visual_weight() -> float:
    return float(os.environ.get("TIRED_MULTIMODAL_W_VISUAL", "0.7"))


def audio_weight() -> float:
    return float(os.environ.get("TIRED_MULTIMODAL_W_AUDIO", "0.3"))


def audio_interval_sec() -> float:
    return float(os.environ.get("TIRED_MULTIMODAL_INTERVAL", "10"))


def is_multimodal_video_audio() -> bool:
    """为真且正在播放视频文件时，优先用 ffmpeg 从视频抽音轨做语音疲劳（需安装 ffmpeg）。"""
    return os.environ.get("TIRED_MULTIMODAL_VIDEO_AUDIO", "").strip().lower() in ("1", "true", "yes")


def alert_watch_threshold() -> float:
    """融合分 ≥ 此值进入「注意」级（新默认 0.30，与加权逻辑配套）。"""
    return float(os.environ.get("TIRED_ALERT_WATCH", "0.30"))


def alert_danger_threshold() -> float:
    """融合分 ≥ 此值进入「危险」级（新默认 0.65）。"""
    return float(os.environ.get("TIRED_ALERT_DANGER", "0.65"))


def alert_watch_mid() -> float:
    """日志文案：在 [注意, 危险) 内区分「轻微 / 中度」，默认 0.45。"""
    return float(os.environ.get("TIRED_ALERT_WATCH_MID", "0.45"))


def fatigue_logic_mode() -> str:
    """疲劳融合策略：weighted（默认，加权视觉+EMA+动态语音权重）或 legacy（原 max+固定权重）。"""
    v = os.environ.get("TIRED_FATIGUE_LOGIC", "weighted").strip().lower()
    if v in ("legacy", "old", "max", "0"):
        return "legacy"
    return "weighted"


def visual_sat_eye() -> float:
    return max(1e-6, float(os.environ.get("TIRED_VISUAL_SAT_EYE", "0.20")))


def visual_sat_yawn() -> float:
    return max(1e-6, float(os.environ.get("TIRED_VISUAL_SAT_YAWN", "0.12")))


def visual_sat_head() -> float:
    return max(1e-6, float(os.environ.get("TIRED_VISUAL_SAT_HEAD", "0.50")))


def visual_w_eye() -> float:
    return max(0.0, float(os.environ.get("TIRED_VISUAL_W_EYE", "0.65")))


def visual_w_yawn() -> float:
    return max(0.0, float(os.environ.get("TIRED_VISUAL_W_YAWN", "0.30")))


def visual_w_head() -> float:
    return max(0.0, float(os.environ.get("TIRED_VISUAL_W_HEAD", "0.05")))


def visual_smooth_alpha() -> float:
    """视觉分 EMA：新窗口权重，0~1；越大越跟当前窗、越小越平滑。"""
    return max(0.0, min(1.0, float(os.environ.get("TIRED_VISUAL_SMOOTH", "0.5"))))


def fusion_boost_visual_lt() -> float:
    """动态融合：平滑视觉分低于此值且语音高于 fusion_boost_audio_gt 时提高语音权重。"""
    return float(os.environ.get("TIRED_FUSION_BOOST_VISUAL_LT", "0.35"))


def fusion_boost_audio_gt() -> float:
    return float(os.environ.get("TIRED_FUSION_BOOST_AUDIO_GT", "0.65"))


def fusion_w_audio_boosted() -> float:
    return max(0.0, min(1.0, float(os.environ.get("TIRED_FUSION_W_AUDIO_BOOST", "0.6"))))


def danger_streak_windows() -> int:
    """危险级需连续多少个统计窗口才累加强告警计数（防单窗抖动），至少 1。"""
    return max(1, int(os.environ.get("TIRED_DANGER_STREAK_WINDOWS", "2")))


def mic_min_rms() -> float:
    """16bit PCM 均方根参考下限，低于则打「音量偏低」提示（仍上传转写）。"""
    return float(os.environ.get("TIRED_MIC_MIN_RMS", "280"))


def mic_rms_silent_abort() -> float:
    """RMS 低于此值视为接近数字静音，放弃本段上传（默认 12）。环境极安静可调低到 6~8。"""
    return float(os.environ.get("TIRED_MIC_RMS_SILENT_ABORT", "12"))


def is_llm_agent_enabled() -> bool:
    """为真时在强疲劳告警触发后异步调用 LLM 生成 JSON 行动计划并执行（TTS / 地图等）。"""
    return os.environ.get("TIRED_LLM_AGENT", "").strip().lower() in ("1", "true", "yes")


def llm_agent_cooldown_sec() -> float:
    """两次 Agent 调用之间的最小间隔（秒）。"""
    return max(30.0, float(os.environ.get("TIRED_AGENT_COOLDOWN_SEC", "120")))


def llm_agent_timeout() -> float:
    """Agent Chat 请求超时（秒）。"""
    return max(8.0, float(os.environ.get("TIRED_AGENT_TIMEOUT", "40")))


def llm_agent_max_tokens() -> int:
    return max(128, min(1024, int(os.environ.get("TIRED_AGENT_MAX_TOKENS", "512"))))


def llm_agent_chat_model() -> str:
    """留空则与 GROQ_CHAT_MODEL 相同。"""
    m = os.environ.get("TIRED_AGENT_CHAT_MODEL", "").strip()
    return m if m else groq_chat_model()


def is_agent_local_tts_enabled() -> bool:
    """为假时不执行本地 speak（不调用 pyttsx3）；LLM、打开地图、日志仍可用。环境变量 TIRED_AGENT_LOCAL_TTS=0 关闭。"""
    return os.environ.get("TIRED_AGENT_LOCAL_TTS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
