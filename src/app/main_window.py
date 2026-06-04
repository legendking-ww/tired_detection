"""主检测窗口（PyQt + Ui_MainWindow）。"""
import os
from urllib.parse import urlparse

import cv2
import numpy as np
from pygame import mixer
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
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)

from src.ui.UI import Ui_MainWindow
from src.app.worker_threads import AdjustCamera_Thread, Start_Thread
from src.multimodal.audio_loop import stop_audio_loop
from src.multimodal.config import (
    alert_danger_threshold,
    alert_watch_threshold,
    audio_interval_sec,
    groq_api_base,
    groq_chat_model,
    groq_whisper_model,
    is_agent_local_tts_enabled,
    is_llm_agent_enabled,
    is_multimodal_enabled,
    is_multimodal_mic,
    is_multimodal_video_audio,
    llm_agent_cooldown_sec,
    mic_record_sec,
)
from src.multimodal.state import (
    clear_fusion_display,
    get_agent_status,
    get_agent_summary,
    get_last_audio_score,
    get_last_fusion,
    get_last_transcript,
    get_voice_log_text,
)
from src.utils.cv_helpers import open_video_capture_by_index
from src.utils.logger import get_logger

_log = get_logger(__name__)


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
        self._refresh_agent_strip()
        self._refresh_multimodal_live()
        self._mm_timer = QTimer(self)
        self._mm_timer.timeout.connect(self._refresh_multimodal_live)
        self._mm_timer.start(2000)

        self._preview_graphics_ready = False
        self._preview_scene = None
        self._preview_item = None
        self._preview_fit_pending = True

        # 动态添加「检测历史」按钮
        self._add_history_button()

    def _add_history_button(self) -> None:
        """在运行控制区末尾添加「检测历史」按钮。"""
        btn = QtWidgets.QPushButton("检测历史")
        btn.setFont(self.Button_AdjustCamera_Location.font())
        btn.setMinimumHeight(36)
        btn.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #7c3aed; color: white; border: none; "
            "border-radius: 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #6d28d9; }"
        )
        btn.clicked.connect(self._show_history_dialog)
        # 找到「调整摄像头」按钮所在的布局并追加
        parent_layout = self.Button_AdjustCamera_Location.parent().layout()
        if parent_layout is not None:
            parent_layout.addWidget(btn)
        self._history_button = btn

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

    def _apply_responsive_size(self) -> None:
        """根据屏幕可用空间自适应窗口大小，并调整分割条比例。"""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        scr = app.primaryScreen()
        if scr is None:
            return
        ag = scr.availableGeometry()
        # 窗口占屏幕可用区域的 90%，但不超过 1920×1040
        w = min(1920, int(ag.width() * 0.90))
        h = min(1040, int(ag.height() * 0.90))
        self.resize(w, h)
        # 自适应左侧面板最大宽度
        left_max = min(620, int(w * 0.32))
        left_panel = self.centralwidget.findChild(QtWidgets.QWidget, "leftWorkbench")
        if left_panel is not None:
            left_panel.setMaximumWidth(left_max)
        # 自适应分割条比例
        main_split = self.centralwidget.findChild(QSplitter, "mainSplit")
        if main_split is not None:
            total_w = main_split.width()
            if total_w > 100:
                main_split.setSizes([int(total_w * 0.26), int(total_w * 0.74)])
        right_h = self.centralwidget.findChild(QSplitter, "rightHSplit")
        if right_h is not None:
            rw = right_h.width()
            if rw > 100:
                right_h.setSizes([int(rw * 0.68), int(rw * 0.32)])

    def showEvent(self, event):
        super().showEvent(event)
        if not self._centered_once:
            self._centered_once = True
            self._apply_responsive_size()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                scr = app.primaryScreen()
                if scr is not None:
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
                "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e676, stop:0.45 #00bcd4, stop:1 #7c4dff);"
                "color: #0d1117; border-radius: 10px; padding: 12px 10px; font-weight: bold; font-size: 13px;"
            )
            va = "已开（需 ffmpeg）" if is_multimodal_video_audio() else "关"
            agent_line = ""
            if is_llm_agent_enabled():
                tts_note = (
                    "本地播报已关（TIRED_AGENT_LOCAL_TTS=0），仅地图/链接/日志。"
                    if not is_agent_local_tts_enabled()
                    else "本地播报可开（pip install pyttsx3）；也可设 TIRED_AGENT_LOCAL_TTS=0 关闭。"
                )
                agent_line = (
                    f"\n主动安全 Agent：已开启。触发：强疲劳弹窗并响提示音后调 LLM；冷却约 {llm_agent_cooldown_sec():g}s。"
                    f"\n侧栏顶部深色区展示推理与执行摘要；地图/链接在浏览器打开。"
                    f"\n{tts_note}"
                )
            self.label_multimodal_detail.setPlainText(
                f"模式：{mode}\n"
                f"视频伴音：{va}\n"
                f"网关：{host}\n"
                f"语音周期约 {audio_interval_sec():g} s · 单次分析时长 {mic_record_sec():g} s\n"
                f"注意/危险阈值：{alert_watch_threshold():g} / {alert_danger_threshold():g}\n"
                f"转写：{groq_whisper_model()}\n"
                f"疲劳打分：{groq_chat_model()}"
                f"{agent_line}"
            )
            self.label_multimodal_detail.verticalScrollBar().setValue(0)
            self.label_multimodal_live.setVisible(True)
            self.label_mm_meter.setVisible(True)
            self.mm_fused_bar.setVisible(True)
            self.label_mm_transcript.setVisible(True)
            self.label_multimodal_live.setPlainText("语音侧：开始检测后，此处每约 2 秒刷新。")
            self.label_multimodal_live.verticalScrollBar().setValue(0)
            title = self.windowTitle()
            if "多模态" not in title:
                self.setWindowTitle(f"{title}  ·  多模态")
            self.statusBar().showMessage("多模态已加载（视觉 + 语音）", 8000)
        else:
            self.label_multimodal_badge.setText("多模态 · 未开启")
            self.label_multimodal_badge.setStyleSheet(
                "background-color: #eceff1; color: #455a64; border-radius: 10px; padding: 12px 10px;"
                "border: 2px dashed #90a4ae; font-weight: bold;"
            )
            mm_txt = (
                "在 .env 中设置 TIRED_MULTIMODAL=1，并配置 SILICONFLOW_API_KEY（或 MULTIMODAL_API_KEY）、"
                "GROQ_API_BASE、麦克风或 WAV 后重启本程序。"
            )
            if is_llm_agent_enabled():
                mm_txt += (
                    f"\n\n主动安全 Agent 已开启：强疲劳弹窗后调 LLM；冷却 {llm_agent_cooldown_sec():g} s。"
                    f"\n可不装本地语音：将 TIRED_AGENT_LOCAL_TTS 设为 0，仅用地图与流水。"
                )
            self.label_multimodal_detail.setPlainText(mm_txt)
            self.label_multimodal_detail.verticalScrollBar().setValue(0)
            self.label_multimodal_live.setPlainText("")
            self.label_multimodal_live.setVisible(False)
            self.label_mm_meter.setVisible(False)
            self.mm_fused_bar.setVisible(False)
            self.label_mm_transcript.setVisible(False)
            self.voice_log_timeline.setPlaceholderText("开启多模态后显示语音识别流水。")

    def _refresh_agent_strip(self) -> None:
        """侧栏 Agent 深色区：进行中状态、最近一次完整摘要，或待机说明。"""
        st = get_agent_status()
        sm = get_agent_summary()
        if st:
            self.label_agent_status.setPlainText(f"进行中\n{st}")
            self.label_agent_status.verticalScrollBar().setValue(0)
            return
        if sm:
            self.label_agent_status.setPlainText(sm)
            self.label_agent_status.verticalScrollBar().setValue(0)
            return
        if is_llm_agent_enabled():
            tts = (
                "本地朗读已关（环境变量 TIRED_AGENT_LOCAL_TTS 为 0），以文字与地图为主。"
                if not is_agent_local_tts_enabled()
                else "安装 pyttsx3 后可将模型「播报」动作转为本机语音。"
            )
            self.label_agent_status.setPlainText(
                "已开启，等待触发：先点「开始检测」；仅在强疲劳弹窗并播放提示音后才会请求 LLM，"
                f"两次调用间隔不少于 {llm_agent_cooldown_sec():.0f} 秒。\n"
                f"{tts}\n"
                "触发后此处显示推理与执行摘要，右侧「语音与 Agent 流水」同步记录。"
            )
            self.label_agent_status.verticalScrollBar().setValue(0)
        else:
            self.label_agent_status.setPlainText(
                "未开启。请在项目根目录 .env 中增加：变量 TIRED_LLM_AGENT，取值 1；"
                "并配置与多模态相同的 API Key 后重启程序。"
            )
            self.label_agent_status.verticalScrollBar().setValue(0)

    def _refresh_multimodal_live(self) -> None:
        if not is_multimodal_enabled():
            self.voice_log_timeline.setPlainText(
                "在 .env 中设置 TIRED_MULTIMODAL=1 并配置 API Key 后，此处将按时间列出每轮 Whisper 转写与疲劳分。"
            )
            self._refresh_agent_strip()
            return
        vis, fused, lvl = get_last_fusion()
        tx = get_last_transcript()
        if fused is not None:
            pct = int(max(0.0, min(1.0, fused)) * 100)
            self.mm_fused_bar.setValue(pct)
            emoji = {"danger": "🔴", "watch": "🟡", "normal": "🟢"}.get(lvl, "⚪")
            self.label_mm_meter.setText(f"{emoji} 融合疲劳 {fused:.2f}  ·  {lvl}")
            if lvl == "danger":
                chunk = "QProgressBar::chunk { border-radius: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ff1744, stop:1 #b71c1c); }"
            elif lvl == "watch":
                chunk = "QProgressBar::chunk { border-radius: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ffea00, stop:1 #ff9100); }"
            else:
                chunk = "QProgressBar::chunk { border-radius: 8px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e676, stop:1 #00bcd4); }"
            self.mm_fused_bar.setStyleSheet(
                "QProgressBar { border: 2px solid #5e35b1; border-radius: 10px; height: 26px; text-align: center; "
                "font-weight: bold; color: #1a237e; background-color: #ede7f6; } "
                + chunk
            )
        else:
            self.mm_fused_bar.setValue(0)
            self.label_mm_meter.setText("融合度 —（检测运行后更新）")
            self.mm_fused_bar.setStyleSheet(
                "QProgressBar { border: 2px solid #b0bec5; border-radius: 10px; height: 26px; text-align: center; "
                "color: #546e7a; background-color: #eceff1; } "
                "QProgressBar::chunk { border-radius: 8px; background-color: #b0bec5; }"
            )
        if tx:
            show = tx[:1200] + ("…" if len(tx) > 1200 else "")
            self.label_mm_transcript.setPlainText("▶ 本轮转写\n" + show)
        else:
            self.label_mm_transcript.setPlainText("▶ 本轮转写\n暂无（等待语音轮次或音量过低）")
        self.label_mm_transcript.verticalScrollBar().setValue(0)

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
        self.label_multimodal_live.setPlainText("\n".join(lines) if lines else "—")
        self.label_multimodal_live.verticalScrollBar().setValue(0)

        self.voice_log_timeline.setPlainText(get_voice_log_text())
        self.voice_log_timeline.verticalScrollBar().setValue(self.voice_log_timeline.verticalScrollBar().maximum())
        self._refresh_agent_strip()

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

    def _show_history_dialog(self) -> None:
        """显示检测历史记录弹窗。"""
        from src.utils.history import DetectionHistory

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("检测历史记录")
        dlg.resize(900, 560)
        dlg.setMinimumSize(700, 400)
        layout = QtWidgets.QVBoxLayout(dlg)

        label = QtWidgets.QLabel("<h3 style='color:#1e293b'>近期检测会话</h3>")
        label.setAlignment(Qt.AlignLeft)
        layout.addWidget(label)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(
            ["开始时间", "持续(秒)", "窗口数", "平均融合分", "最高融合分", "危险次数", "操作"]
        )
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            "QTableWidget { border: 1px solid #cbd5e1; border-radius: 8px; gridline-color: #e2e8f0; } "
            "QHeaderView::section { background: #f1f5f9; padding: 6px; font-weight: bold; }"
        )

        hist = DetectionHistory("mrsoft.db")
        sessions = hist.list_sessions(limit=50)
        table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            import datetime
            start_dt = datetime.datetime.fromtimestamp(s["start_ts"]).strftime("%m/%d %H:%M:%S")
            duration = int(s["end_ts"] - s["start_ts"]) if s["end_ts"] > s["start_ts"] else 0
            table.setItem(i, 0, QTableWidgetItem(start_dt))
            table.setItem(i, 1, QTableWidgetItem(f"{duration}s"))
            table.setItem(i, 2, QTableWidgetItem(str(s["window_count"])))
            table.setItem(i, 3, QTableWidgetItem(f"{s['avg_fused']:.3f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{s['max_fused']:.3f}"))
            # 危险次数着色
            danger_item = QTableWidgetItem(str(s["danger_count"]))
            if s["danger_count"] > 0:
                danger_item.setForeground(Qt.red)
                f = danger_item.font()
                f.setBold(True)
                danger_item.setFont(f)
            table.setItem(i, 5, danger_item)
            btn_detail = QtWidgets.QPushButton("详情")
            btn_detail.clicked.connect(
                lambda _checked, sid=s["session_id"]: self._show_session_detail(sid)
            )
            table.setCellWidget(i, 6, btn_detail)
        table.resizeColumnsToContents()
        layout.addWidget(table)

        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setMinimumHeight(36)
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)

        dlg.exec_()

    def _show_session_detail(self, session_id: str) -> None:
        """弹窗显示单次会话的详细曲线数据。"""
        from src.utils.history import DetectionHistory

        hist = DetectionHistory("mrsoft.db")
        data = hist.get_session_data(session_id)
        alerts = hist.get_alert_events(session_id)

        if not data:
            QMessageBox.information(self, "提示", "该会话暂无数据。")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"会话详情: {session_id}")
        dlg.resize(800, 500)
        lyt = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel(
            f"<b>会话 ID:</b> {session_id}<br>"
            f"<b>窗口数:</b> {len(data)} &nbsp;&nbsp; <b>告警事件:</b> {len(alerts)}"
        )
        info.setStyleSheet("padding: 6px; color: #334155;")
        lyt.addWidget(info)

        # 文本摘要
        text = QtWidgets.QTextBrowser()
        text.setFont(self.font())
        lines = []
        import datetime
        for d in data:
            ts = datetime.datetime.fromtimestamp(d["timestamp"]).strftime("%H:%M:%S")
            lvl_mark = {"danger": "🔴", "watch": "🟡", "normal": "🟢"}.get(d["alert_level"], "⚪")
            lines.append(
                f"{ts}  {lvl_mark} fused={d['fused_score']:.3f}  "
                f"vis={d['visual_score']:.3f}  ear={d['ear']:.2f}  "
                f"pclos={d['perclos']:.2f}  blink={d['blink_rate']:.1f}/m"
            )
        text.setPlainText("\n".join(lines))
        lyt.addWidget(text)

        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.setMinimumHeight(36)
        close_btn.clicked.connect(dlg.accept)
        lyt.addWidget(close_btn)

        dlg.exec_()

    def end(self):
        _log.info("用户请求停止检测")
        self.adjust_camera_Thread.stop(timeout_ms=2000)
        self.start_Thread.stop(timeout_ms=3000)
        if is_multimodal_enabled():
            try:
                stop_audio_loop()
                clear_fusion_display()
            except Exception:
                pass
        self.output_Window.append("检测已停止。")

    def closeEvent(self, event):
        """有序退出：停音频 → 停线程 → 释放资源 → 清理 mixer。"""
        _log.info("MainWindow closeEvent: 开始有序退出…")
        try:
            self._mm_timer.stop()
        except Exception:
            pass
        if is_multimodal_enabled():
            try:
                stop_audio_loop()
                clear_fusion_display()
            except Exception as e:
                _log.warning("关闭多模态资源时出错: %s", e)
        self.adjust_camera_Thread.stop(timeout_ms=2000)
        self.start_Thread.stop(timeout_ms=3000)
        try:
            mixer.quit()
        except Exception:
            pass
        _log.info("MainWindow 退出完成")
        event.accept()

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
