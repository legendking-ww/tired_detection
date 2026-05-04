"""从视频文件截取一段音轨为 WAV（需系统已安装 ffmpeg）。"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Tuple


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_wav_segment(
    video_path: str,
    start_sec: float,
    duration_sec: float,
) -> Tuple[str, str]:
    """
    返回 (临时 wav 路径, 错误信息)。成功时错误为空字符串，调用方负责 unlink wav。
    """
    if not ffmpeg_available():
        return "", "未检测到 ffmpeg。请安装 ffmpeg 并加入 PATH，或关闭 TIRED_MULTIMODAL_VIDEO_AUDIO 改用麦克风/WAV。"
    duration_sec = max(0.5, min(120.0, float(duration_sec)))
    start_sec = max(0.0, float(start_sec))
    fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="tired_vid_")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=duration_sec + 90.0)
        if r.returncode != 0:
            err = (r.stderr or b"").decode(errors="replace")[:400]
            try:
                os.unlink(out_path)
            except OSError:
                pass
            return "", err or "ffmpeg 退出码非 0"
    except subprocess.TimeoutExpired:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return "", "ffmpeg 超时"
    except Exception as e:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return "", repr(e)
    return out_path, ""
