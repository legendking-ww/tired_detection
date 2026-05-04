"""后台线程：按间隔分析 WAV（麦克风 / 视频伴音 ffmpeg / 固定文件）并更新语音疲劳分。"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from . import config, state
from .groq_audio import score_fatigue_from_wav
from . import video_audio_ctx
from .video_audio_extract import extract_wav_segment

_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def _loop() -> None:
    interval = max(3.0, config.audio_interval_sec())
    warned_no_wav = False
    while not _stop.is_set():
        t0 = time.monotonic()
        tmp_unlink: Optional[str] = None
        key = config.groq_api_key()

        vpath, pos_msec = video_audio_ctx.get_playback()
        use_video = (
            config.is_multimodal_video_audio()
            and vpath
            and os.path.isfile(vpath)
            and key
        )

        if use_video:
            rec = max(0.5, min(60.0, config.mic_record_sec()))
            start_sec = max(0.0, (pos_msec / 1000.0) - rec * 0.5)
            wav_path, err = extract_wav_segment(vpath, start_sec, rec)
            if err:
                state.append_voice_log("视频伴音", err)
                state.set_last_audio_score(-1.0, err)
            else:
                tmp_unlink = wav_path
                score, serr = score_fatigue_from_wav(wav_path)
                state.set_last_audio_score(score, serr)
        elif config.is_multimodal_mic():
            if not key:
                state.set_last_audio_score(None, "未设置 SILICONFLOW_API_KEY / MULTIMODAL_API_KEY / GROQ_API_KEY")
            else:
                from .mic_qt import MicRecordThread

                rec = max(0.5, min(60.0, config.mic_record_sec()))
                thr = MicRecordThread(rec)
                thr.start()
                wait_ms = int(rec * 1000) + 20000
                if not thr.wait(wait_ms):
                    state.append_voice_log("麦克风", "录制线程超时")
                    state.set_last_audio_score(-1.0, "麦克风录制线程超时")
                elif thr.error:
                    state.append_voice_log("麦克风", thr.error)
                    state.set_last_audio_score(-1.0, thr.error)
                elif thr.out_path:
                    tmp_unlink = thr.out_path
                    score, err = score_fatigue_from_wav(thr.out_path)
                    if getattr(thr, "low_volume_hint", False):
                        hint = "麦克风音量偏低，若几乎识别不到发言请靠近话筒或提高系统输入音量。"
                        err = f"{err}; {hint}" if err else hint
                    state.set_last_audio_score(score, err)
                else:
                    state.append_voice_log("麦克风", "录制未生成文件")
                    state.set_last_audio_score(-1.0, "麦克风录制未生成文件")
        else:
            wav = config.multimodal_wav_path()
            if not wav:
                if not warned_no_wav:
                    state.set_last_audio_score(
                        None,
                        "未配置 TIRED_MULTIMODAL_WAV 且默认 resources/samples/driver_demo.wav 不存在",
                    )
                    warned_no_wav = True
            elif not key:
                state.set_last_audio_score(None, "未设置 SILICONFLOW_API_KEY / MULTIMODAL_API_KEY / GROQ_API_KEY")
            else:
                score, err = score_fatigue_from_wav(wav)
                state.set_last_audio_score(score, err)

        if tmp_unlink:
            try:
                os.unlink(tmp_unlink)
            except OSError:
                pass

        elapsed = time.monotonic() - t0
        wait_sec = max(1.0, interval - elapsed)
        if _stop.wait(wait_sec):
            break


def start_audio_loop() -> None:
    global _thread
    stop_audio_loop()
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="MultimodalAudio", daemon=True)
    _thread.start()


def stop_audio_loop() -> None:
    global _thread
    _stop.set()
    t = _thread
    _thread = None
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    _stop.clear()
