"""主检测窗口（PyQt + Ui_MainWindow）。"""
import os
from urllib.parse import urlparse

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
    QFileDialog,
)

from src.ui.UI import Ui_MainWindow
from src.app.worker_threads import AdjustCamera_Thread, Start_Thread
from src.multimodal.config import (
    alert_danger_threshold,
    alert_watch_threshold,
    audio_interval_sec,
    groq_api_base,
    groq_chat_model,
    groq_whisper_model,
    is_multimodal_enabled,
    is_multimodal_mic,
    is_multimodal_video_audio,
    mic_record_sec,
)
from src.multimodal.state import get_last_audio_score, get_last_fusion, get_last_transcript, get_voice_log_text
from src.utils.cv_helpers import open_video_capture_by_index


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self._centered_once = False
        self.adjust_camera_Thread = AdjustCamera_Thread()
        self.start_Thread = Start_Thread()
        self.setupUi(self)
        # 必须先连接信号再填充下拉框，否则默认选中的摄像头不会同步到线程
        self.Cam_Select.currentIndexChanged.connect(self.change_Cam_Select)
        self.init_camera_list()
        self.Button_OpenVideo.clicked.connect(self.open_Video)
        self.Button_Start.clicked.connect(self.start)
        self.Button_End.clicked.connect(self.end)
        self.Button_AdjustCamera_Location.clicked.connect(self.adjust_camera_location)
        self.offDuty_Check.clicked.connect(self.change_OffDuty_Check_Status)
        self.offDuty_Time.valueChanged.connect(self.change_OffDuty_Value)
        self.video.clicked.connect(self.set_open_video)
        self.cam.clicked.connect(self.set_open_video)
        self.show_eye.clicked.connect(self.set_show_setting)
        self.show_head.clicked.connect(self.set_show_setting)
        self.show_mouth.clicked.connect(self.set_show_setting)
        self.show_key_point.clicked.connect(self.set_show_setting)

        self.start_Thread.msg.connect(self.show_Message)
        self.start_Thread.picture.connect(self.show_Image)
        self.start_Thread.window.connect(self.pop_window)
        self.adjust_camera_Thread.picture.connect(self.show_Image)
        self.adjust_camera_Thread.msg.connect(self.show_Message)
        self.adjust_camera_Thread.window.connect(self.pop_window)

        self._setup_multimodal_ui_panel()
        self._refresh_multimodal_live()
        self._mm_timer = QTimer(self)
        self._mm_timer.timeout.connect(self._refresh_multimodal_live)
        self._mm_timer.start(2000)

        self._preview_graphics_ready = False
        self._preview_scene = None
        self._preview_item = None
        self._preview_fit_pending = True

    def _ensure_preview_graphics(self) -> None:
        if self._preview_graphics_ready:
            return
        self._preview_scene = QGraphicsScene(self.graphicsView)
        self._preview_item = QGraphicsPixmapItem()
        self._preview_scene.addItem(self._preview_item)
        self.graphicsView.setScene(self._preview_scene)
        self.graphicsView.setOptimizationFlags(
            QGraphicsView.DontSavePainterState | QGraphicsView.DontAdjustForAntialiasing
        )
        self.graphicsView.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self._preview_graphics_ready = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._centered_once:
            return
        self._centered_once = True
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        scr = app.primaryScreen()
        if scr is None:
            return
        ag = scr.availableGeometry()
        fg = self.frameGeometry()
        fg.moveCenter(ag.center())
        self.move(fg.topLeft())

    def _setup_multimodal_ui_panel(self) -> None:
        if is_multimodal_enabled():
            mode = "麦克风采集" if is_multimodal_mic() else "音频文件轮询"
            base = groq_api_base()
            try:
                host = urlparse(base).netloc or base
            except Exception:
                host = base
            self.label_multimodal_badge.setText("多模态 · 已开启")
            self.label_multimodal_badge.setStyleSheet(
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1e88e5, stop:1 #6a1b9a);"
                "color: white; border-radius: 8px; padding: 10px; font-weight: bold;"
            )
            va = "已开（需 ffmpeg）" if is_multimodal_video_audio() else "关"
            self.label_multimodal_detail.setText(
                f"模式：{mode}\n"
                f"视频伴音：{va}\n"
                f"网关：{host}\n"
                f"语音周期约 {audio_interval_sec():g} s · 单次分析时长 {mic_record_sec():g} s\n"
                f"注意/危险阈值：{alert_watch_threshold():g} / {alert_danger_threshold():g}\n"
                f"转写：{groq_whisper_model()}\n"
                f"疲劳打分：{groq_chat_model()}"
            )
            self.label_multimodal_live.setVisible(True)
            self.label_mm_meter.setVisible(True)
            self.mm_fused_bar.setVisible(True)
            self.label_mm_transcript.setVisible(True)
            self.label_multimodal_live.setText("语音侧：开始检测后，此处每约 2 秒刷新。")
            title = self.windowTitle()
            if "多模态" not in title:
                self.setWindowTitle(f"{title}  ·  多模态")
            self.statusBar().showMessage("多模态已加载（视觉 + 语音）", 8000)
        else:
            self.label_multimodal_badge.setText("多模态 · 未开启")
            self.label_multimodal_badge.setStyleSheet(
                "background-color: #eceff1; color: #546e7a; border-radius: 8px; padding: 10px;"
                "border: 1px solid #cfd8dc;"
            )
            self.label_multimodal_detail.setText(
                "在 .env 中设置 TIRED_MULTIMODAL=1，并配置 SILICONFLOW_API_KEY（或 MULTIMODAL_API_KEY）、"
                "GROQ_API_BASE、麦克风或 WAV 后重启本程序。"
            )
            self.label_multimodal_live.setText("")
            self.label_multimodal_live.setVisible(False)
            self.label_mm_meter.setVisible(False)
            self.mm_fused_bar.setVisible(False)
            self.label_mm_transcript.setVisible(False)
            self.voice_log_timeline.setPlaceholderText("开启多模态后显示语音识别流水。")

    def _refresh_multimodal_live(self) -> None:
        if not is_multimodal_enabled():
            self.voice_log_timeline.setPlainText(
                "在 .env 中设置 TIRED_MULTIMODAL=1 并配置 API Key 后，此处将按时间列出每轮 Whisper 转写与疲劳分。"
            )
            return
        vis, fused, lvl = get_last_fusion()
        tx = get_last_transcript()
        if fused is not None:
            pct = int(max(0.0, min(1.0, fused)) * 100)
            self.mm_fused_bar.setValue(pct)
            emoji = {"danger": "🔴", "watch": "🟡", "normal": "🟢"}.get(lvl, "⚪")
            self.label_mm_meter.setText(f"{emoji} 融合疲劳 {fused:.2f}  ·  {lvl}")
            if lvl == "danger":
                chunk = "QProgressBar::chunk { border-radius: 6px; background-color: #c62828; }"
            elif lvl == "watch":
                chunk = "QProgressBar::chunk { border-radius: 6px; background-color: #f9a825; }"
            else:
                chunk = "QProgressBar::chunk { border-radius: 6px; background-color: #2e7d32; }"
            self.mm_fused_bar.setStyleSheet(
                "QProgressBar { border: 2px solid #90caf9; border-radius: 8px; height: 24px; text-align: center; } "
                + chunk
            )
        else:
            self.mm_fused_bar.setValue(0)
            self.label_mm_meter.setText("融合度 —（检测运行后更新）")
            self.mm_fused_bar.setStyleSheet(
                "QProgressBar { border: 2px solid #cfd8dc; border-radius: 8px; height: 24px; } "
                "QProgressBar::chunk { border-radius: 6px; background-color: #b0bec5; }"
            )
        if tx:
            show = tx[:320] + ("…" if len(tx) > 320 else "")
            self.label_mm_transcript.setText("▶ 本轮转写\n" + show)
        else:
            self.label_mm_transcript.setText("▶ 本轮转写\n暂无（等待语音轮次或音量过低）")

        score, err = get_last_audio_score()
        lines = []
        if score is None and not err:
            lines.append("语音分：等待本轮分析…")
        elif score is not None and score >= 0.0:
            lines.append(f"语音疲劳分：{score:.2f}（0 清醒 ~ 1 很困）")
        elif score is not None and score < 0:
            lines.append("语音分析失败，融合暂为纯视觉。")
        if vis is not None and fused is not None:
            lines.append(f"视觉分：{vis:.2f}")
        if err:
            short = err if len(err) <= 140 else err[:137] + "…"
            lines.append(f"详情：{short}")
        self.label_multimodal_live.setText("\n".join(lines) if lines else "—")

        self.voice_log_timeline.setPlainText(get_voice_log_text())
        self.voice_log_timeline.verticalScrollBar().setValue(self.voice_log_timeline.verticalScrollBar().maximum())

    def init_camera_list(self):
        self.Cam_Select.blockSignals(True)
        self.Cam_Select.clear()
        self.Cam_Select.addItem("选择摄像头")

        # 无效 index 在 Windows 上会多次打开后端并刷屏、易卡顿；默认少探几个，可用 TIRED_CAMERA_PROBE_MAX 加大
        max_cameras = max(1, min(8, int(os.environ.get("TIRED_CAMERA_PROBE_MAX", "3"))))
        for i in range(max_cameras):
            cap = open_video_capture_by_index(i, verify_frame=False)
            if cap is not None:
                self.Cam_Select.addItem(f"摄像头 {i}")
                cap.release()

        if self.Cam_Select.count() > 1:
            self.Cam_Select.setCurrentIndex(1)
        self.Cam_Select.blockSignals(False)
        if self.Cam_Select.currentIndex() > 0:
            self.change_Cam_Select()

    def set_show_setting(self):
        isChecked = self.sender().isChecked()
        if self.sender() == self.show_eye:
            self.start_Thread.set_show_eye(isChecked)
        elif self.sender() == self.show_mouth:
            self.start_Thread.set_show_mouth(isChecked)
        elif self.sender() == self.show_head:
            self.start_Thread.set_show_Head(isChecked)
        else:
            self.start_Thread.set_show_key_point(isChecked)

    def set_open_video(self):
        if self.video.isChecked():
            self.start_Thread.set_open_video(True)
        else:
            self.start_Thread.set_open_video(False)

    def change_OffDuty_Check_Status(self):
        self.start_Thread.change_OffDuty_Check_Status(self.offDuty_Check.isChecked())

    def change_OffDuty_Value(self):
        self.start_Thread.change_OffDuty_Value(self.offDuty_Time.value())

    def start(self):
        if self.start_Thread.isRunning():
            self.output_Window.append("检测已在运行，请先点击结束再开始")
            return
        idx = self.Cam_Select.currentIndex()
        if idx <= 0 and not self.start_Thread.isOpenVideo:
            self.output_Window.append("请先在下拉框中选择摄像头，或使用「打开视频文件」")
            QMessageBox.warning(self, "提示", "未选择有效摄像头。请从「摄像头设置」中选择设备后再开始检测。")
            return
        self.change_Cam_Select()
        self.start_Thread.change_OffDuty_Check_Status(self.offDuty_Check.isChecked())
        self.start_Thread.change_OffDuty_Value(self.offDuty_Time.value())
        self._preview_fit_pending = True
        self.start_Thread.start()

    def adjust_camera_location(self):
        self._preview_fit_pending = True
        self.adjust_camera_Thread.start()

    def end(self):
        self.adjust_camera_Thread.close()
        self.start_Thread.close()

    def change_Cam_Select(self):
        current_index = self.Cam_Select.currentIndex()
        if current_index > 0:
            cam_index = current_index - 1
            self.adjust_camera_Thread.change_cam_select(cam_index)
            self.start_Thread.change_cam_select(cam_index)
            self.output_Window.append(f"切换摄像头到: 摄像头 {cam_index}")
        else:
            self.output_Window.append("请选择一个有效的摄像头")

    def open_Video(self):
        filePath = QFileDialog.getOpenFileName(self, "打开视频文件", "", 'Video files(*.mp4)')
        self.output_Window.append("视频文件" + filePath[0] + "加载成功")
        self.start_Thread.set_filePath(filePath[0])
        self.video.setChecked(True)

    def show_Message(self, msg):
        self.output_Window.append(msg)

    def show_Image(self, image):
        """预览：复用 Scene/Item，避免每帧 new Scene + fitInView（原先是大卡顿来源）。"""
        max_h = int(os.environ.get("TIRED_PREVIEW_MAX_HEIGHT", "900"))
        if max_h > 0:
            h0, w0 = image.shape[:2]
            if h0 > max_h:
                scale = max_h / float(h0)
                image = cv2.resize(
                    image,
                    (max(1, int(w0 * scale)), max_h),
                    interpolation=cv2.INTER_AREA,
                )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        frame = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(frame)
        self._ensure_preview_graphics()
        self._preview_item.setPixmap(pix)
        if self._preview_fit_pending:
            self._preview_fit_pending = False
            self.graphicsView.fitInView(self._preview_item.boundingRect(), Qt.KeepAspectRatio)

    def pop_window(self, info):
        QMessageBox.warning(self, "提示", info, QMessageBox.Yes)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._preview_item is not None and not self._preview_item.pixmap().isNull():
            self.graphicsView.fitInView(self._preview_item.boundingRect(), Qt.KeepAspectRatio)
