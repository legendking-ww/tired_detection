"""使用 Groq OpenAI 兼容接口：Whisper 转写 + 聊天模型输出 0~1 疲劳分。"""
from __future__ import annotations

import os
import re
import time
from typing import Optional, Tuple

import requests

from . import state as mm_state
from .config import groq_api_base, groq_api_key, groq_chat_model, groq_whisper_model


def _transcribe_form_fields() -> dict:
    """Groq 转写支持 response_format；多数其它网关（如 SiliconFlow）勿传该字段。"""
    d: dict = {"model": groq_whisper_model()}
    if "groq.com" in groq_api_base().lower():
        d["response_format"] = "json"
    return d


def _endpoints() -> Tuple[str, str]:
    base = groq_api_base().rstrip("/")
    return f"{base}/audio/transcriptions", f"{base}/chat/completions"


def _http_retry_post(url: str, **kwargs) -> requests.Response:
    """对 429/5xx 与网络异常做有限次指数退避重试。"""
    delays = (1.0, 2.0)
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            r = requests.post(url, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            return r
        except requests.RequestException:
            if attempt < max_attempts - 1:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            raise
    raise RuntimeError("_http_retry_post: unreachable")


def _parse_score_0_1(text: str) -> float:
    text = (text or "").strip()
    for pat in (r"\b(1\.0|1)\b", r"\b(0\.\d+)\b"):
        m = re.search(pat, text, re.I)
        if m:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
    return 0.0


def transcribe_wav(wav_path: str, timeout: float = 60.0) -> Tuple[Optional[str], Optional[str]]:
    key = groq_api_key()
    if not key:
        return None, "未设置 SILICONFLOW_API_KEY / MULTIMODAL_API_KEY / GROQ_API_KEY"
    t_url, _ = _endpoints()
    name = os.path.basename(wav_path) or "audio.wav"
    delays = (1.0, 2.0)
    last_err: Optional[str] = None
    for attempt in range(3):
        try:
            with open(wav_path, "rb") as f:
                r = requests.post(
                    t_url,
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (name, f, "audio/wav")},
                    data=_transcribe_form_fields(),
                    timeout=timeout,
                )
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            if r.status_code != 200:
                return None, f"Whisper HTTP {r.status_code}: {r.text[:500]}"
            data = r.json()
            return data.get("text") or "", None
        except requests.RequestException as e:
            last_err = repr(e)
            if attempt < 2:
                time.sleep(delays[min(attempt, len(delays) - 1)])
                continue
            return None, last_err
        except Exception as e:
            return None, repr(e)
    return None, last_err or "Whisper 重试耗尽"


def fatigue_score_from_text(transcript: str, timeout: float = 45.0) -> Tuple[Optional[float], Optional[str]]:
    key = groq_api_key()
    if not key:
        return None, "未设置 SILICONFLOW_API_KEY / MULTIMODAL_API_KEY / GROQ_API_KEY"
    _, c_url = _endpoints()
    prompt = (
        "你是车载安全助手。下面是一段「驾驶员说话内容」的语音识别文本（可能含口语、重复、停顿）。"
        "请结合语义判断：是否困倦、反应变慢、表达含糊、情绪低落、抱怨劳累、打哈欠相关语气等。"
        "若文本过短或只有语气词，可略保守给分。只输出一个 0 到 1 的小数，1 非常疲劳，0 基本清醒；"
        "不要输出其它文字或解释。\n\n"
        f"转写：{transcript[:4000]}"
    )
    try:
        r = _http_retry_post(
            c_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": groq_chat_model(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return None, f"Chat HTTP {r.status_code}: {r.text[:500]}"
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_score_0_1(content), None
    except requests.RequestException as e:
        return None, repr(e)
    except (KeyError, IndexError, TypeError) as e:
        return None, f"解析聊天响应失败: {e!r}"
    except Exception as e:
        return None, repr(e)


def score_fatigue_from_wav(wav_path: str) -> Tuple[Optional[float], Optional[str]]:
    """成功返回 (0~1, None)；API 失败返回 (-1.0, 说明)；无 Key 等无法调用返回 (None, 说明)。"""
    key = groq_api_key()
    if not key:
        return None, "未设置 SILICONFLOW_API_KEY / MULTIMODAL_API_KEY / GROQ_API_KEY"
    text, err = transcribe_wav(wav_path)
    if err:
        mm_state.set_last_transcript("")
        mm_state.append_voice_log("Whisper", err)
        return -1.0, err
    if text is None:
        mm_state.append_voice_log("Whisper", "转写返回为空")
        return -1.0, "转写失败"
    t = (text or "").strip()
    if not t:
        mm_state.set_last_transcript("")
        mm_state.append_voice_log("转写", "（无有效文本，可能静音或环境噪声）")
        return 0.0, None
    mm_state.set_last_transcript(t)
    preview = t if len(t) <= 160 else t[:157] + "…"
    mm_state.append_voice_log("转写", preview)
    s, err2 = fatigue_score_from_text(t)
    if err2:
        mm_state.append_voice_log("疲劳分", err2)
        return -1.0, err2
    if s is None:
        mm_state.append_voice_log("疲劳分", "模型无有效输出")
        return -1.0, "LLM 无有效疲劳分"
    mm_state.append_voice_log("疲劳分", f"{float(s):.2f}（0 清醒 ~ 1 很困）")
    return float(s), None
