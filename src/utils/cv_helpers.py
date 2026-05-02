"""OpenCV 辅助：摄像头多后端打开、BGR 图像中文叠字（PIL）。"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 项目根目录（本文件位于 src/utils/）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def open_video_capture_by_index(index: int):
    """
    打开指定序号的摄像头。Windows 上部分环境 DirectShow(DSHOW) 会报
    VIDEOIO(DSHOW): ... can't be used to capture by index 且无法出图，
    与画面是否镜像无关。Windows 上优先 MSMF，再试 DSHOW（部分机器上 DSHOW
    会对某些 index 打印 VIDEOIO(DSHOW)... can't be used to capture 且无法出图）。
    """
    backends = []
    if sys.platform == "win32":
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for api in backends:
        cap = cv2.VideoCapture(index) if api is None else cv2.VideoCapture(index, api)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue
        ret, _ = cap.read()
        if ret:
            return cap
        try:
            cap.release()
        except Exception:
            pass
    return None


def draw_text_cn_on_bgr(img, text, position, font_size=20, color=(0, 255, 255)):
    """在 BGR 图像上绘制中文（PIL）。cv2.putText 不支持中文，会显示为问号。"""
    if not text:
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    fill_rgb = (int(color[2]), int(color[1]), int(color[0]))

    font_paths = [
        os.path.join(PROJECT_ROOT, "resources", "fonts", "msyh.ttc"),
        os.path.join(PROJECT_ROOT, "resources", "fonts", "simhei.ttf"),
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=fill_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
