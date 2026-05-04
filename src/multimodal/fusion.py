"""视觉分与语音分加权融合及三级预警。

融合侧约定：``audio_s < 0``（如 API 失败写入的 -1）表示本周期无有效语音分，
调用方应退化为仅视觉分（见 ``worker_threads``）。
"""

from . import config


def fuse_visual_audio(visual: float, audio: float, w_visual: float = 0.7, w_audio: float = 0.3) -> float:
    wv, wa = max(0.0, w_visual), max(0.0, w_audio)
    s = wv + wa
    if s <= 1e-6:
        return max(0.0, min(1.0, visual))
    return max(0.0, min(1.0, (wv * visual + wa * audio) / s))


def fuse_visual_audio_dynamic(visual: float, audio: float) -> float:
    """视觉偏低且语音偏高时提高语音权重，便于遮挡/光线差时仍依赖说话内容。"""
    vlt = config.fusion_boost_visual_lt()
    agt = config.fusion_boost_audio_gt()
    wa_boost = config.fusion_w_audio_boosted()
    wv_def = config.visual_weight()
    wa_def = config.audio_weight()
    if visual < vlt and audio > agt:
        wa = wa_boost
        wv = 1.0 - wa
    else:
        wv, wa = wv_def, wa_def
    return fuse_visual_audio(visual, audio, wv, wa)


def alert_level(score: float) -> str:
    d = config.alert_danger_threshold()
    w = config.alert_watch_threshold()
    if w >= d:
        w = max(0.0, d - 0.05)
    if score >= d:
        return "danger"
    if score >= w:
        return "watch"
    return "normal"
