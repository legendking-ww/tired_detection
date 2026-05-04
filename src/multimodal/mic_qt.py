"""使用 PyQt5 QAudioInput 同步录制 PCM 到 WAV（在独立 QThread 内跑局部事件循环）。"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import wave
from typing import Optional

from . import config

from PyQt5.QtCore import QByteArray, QEventLoop, QIODevice, QThread, QTimer
from PyQt5.QtMultimedia import QAudioDeviceInfo, QAudioFormat, QAudioInput


class _PCMCollector(QIODevice):
    """接收 QAudioInput 写入的原始 PCM。"""

    def __init__(self) -> None:
        super().__init__()
        self._buf = QByteArray()

    def readData(self, maxlen: int) -> bytes:
        return b""

    def writeData(self, data) -> int:
        if data is None:
            return 0
        if not isinstance(data, bytes):
            data = bytes(data)
        self._buf.append(data)
        return len(data)

    def pcm_bytes(self) -> bytes:
        return bytes(self._buf)


def _pcm_rms16(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n < 8:
        return 0.0
    fmt = "<%dh" % n
    vals = struct.unpack(fmt, pcm[: n * 2])
    return math.sqrt(sum(float(v) * v for v in vals) / float(n))


class MicRecordThread(QThread):
    """阻塞录制：在 run() 内 exec 短时 QEventLoop，供非 GUI 线程通过 wait() 同步等待。"""

    def __init__(self, duration_sec: float, parent=None) -> None:
        super().__init__(parent)
        self.duration_sec = max(0.5, float(duration_sec))
        self.out_path: str = ""
        self.error: str = ""
        self.low_volume_hint = False

    def run(self) -> None:
        self.out_path = ""
        self.error = ""
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleSize(16)
        fmt.setCodec("audio/pcm")
        fmt.setByteOrder(QAudioFormat.LittleEndian)
        fmt.setSampleType(QAudioFormat.SignedInt)

        dev = QAudioDeviceInfo.defaultInputDevice()
        try:
            no_mic = dev.isNull()
        except AttributeError:
            no_mic = not dev.deviceName()
        if no_mic:
            self.error = "未找到默认麦克风设备"
            return

        if not dev.isFormatSupported(fmt):
            fmt = dev.nearestFormat(fmt)

        sink = _PCMCollector()
        if not sink.open(QIODevice.WriteOnly):
            self.error = "无法打开音频缓冲设备"
            return

        audio_in = QAudioInput(fmt, self)
        loop = QEventLoop()
        done = False

        def finish() -> None:
            nonlocal done
            if done:
                return
            done = True
            audio_in.stop()
            loop.quit()

        audio_in.start(sink)
        # 多给一点时间让驱动刷净缓冲区
        ms = int(self.duration_sec * 1000) + 150
        QTimer.singleShot(ms, finish)
        loop.exec_()

        pcm = sink.pcm_bytes()
        sink.close()
        if len(pcm) < fmt.sampleRate() * fmt.channelCount() * 2 // 4:
            self.error = f"录制数据过少（{len(pcm)} 字节），请检查麦克风权限与设备"
            return

        rms = _pcm_rms16(pcm)
        min_rms = config.mic_min_rms()
        silent_abort = config.mic_rms_silent_abort()
        if rms < silent_abort:
            self.error = (
                f"录到的音量极低（RMS≈{rms:.0f}），低于静默门限 {silent_abort:.0f}。"
                "请提高「设置-声音-输入」音量、选对麦克风并靠近说话；或在 .env 中调低 TIRED_MIC_RMS_SILENT_ABORT（不建议低于 6）。"
            )
            return
        self.low_volume_hint = rms < min_rms

        fd, path = tempfile.mkstemp(suffix=".wav", prefix="tired_mic_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(fmt.channelCount())
                wf.setsampwidth(2)
                wf.setframerate(fmt.sampleRate())
                wf.writeframes(pcm)
            self.out_path = path
        except OSError as e:
            self.error = f"写入 WAV 失败: {e}"
            try:
                os.unlink(path)
            except OSError:
                pass
