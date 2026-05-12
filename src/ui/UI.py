#导入pyqt5模块
import os
from PyQt5.QtGui import QIcon, QFont, QFontInfo, QPalette, QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt
##QtCore模块包含了一些核心的非图形功能，例如事件循环和信号槽机制。
#QtGui模块包含了一些基本的图形功能，例如字体、颜色和绘图工具。
#QtWidgets模块包含了一些GUI元素，例如窗口、标签、按钮、复选框等，用于创建用户界面。

def _ui_cn_font(point_size: int, weight=QFont.Normal) -> QFont:
    """中文界面：优先 YaHei UI，全字距 hinting + 抗锯齿，利于小字号清晰。"""
    f = QFont("Microsoft YaHei UI", point_size, weight)
    if not QFontInfo(f).exactMatch():
        f = QFont("微软雅黑", point_size, weight)
    f.setHintingPreference(QFont.PreferFullHinting)
    f.setStyleStrategy(QFont.PreferAntialias)
    return f


#用于设计主窗口。创建各个控件，标签，线，按钮，复选框等，并指定了它们的字体，位置等属性
class Ui_MainWindow(object):
    #定义setupUi方法，该方法接收一个MainWindow对象作为参数。
    def setupUi(self, MainWindow):
        #给这个实例命了个名字（唯一标识符）。方便类外界访问。
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(1920, 1040)
        #设置窗口图标
        MainWindow.setWindowIcon(QIcon("resources/images/yjwj.png"))
        
        #创建一个QWidget对象，作为MainWindow对象的子控件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #设置中心部件的背景颜色
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 248, 252))
        self.centralwidget.setPalette(palette)
        MainWindow.setStyleSheet(
            """
            QMainWindow {
                background-color: #e2e8f0;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑";
            }
            QSplitter#mainSplit::handle {
                background: #cbd5e1;
                width: 5px;
                border-radius: 2px;
                margin: 2px 0;
            }
            QSplitter#rightHSplit::handle {
                background: #cbd5e1;
                width: 4px;
                border-radius: 2px;
            }
            QSplitter#rightVSplit::handle {
                background: #cbd5e1;
                height: 4px;
                border-radius: 2px;
            }
            /* 勿对 QGroupBox 本体设 font-weight:bold，否则会继承到组内所有控件，中文易糊、挤 */
            QGroupBox {
                font-weight: normal;
                border: 1px solid #cbd5e1;
                border-radius: 12px;
                margin-top: 12px;
                padding: 14px 10px 10px 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #0f172a;
                font-weight: bold;
            }
            QPushButton { border-radius: 8px; padding: 6px; }
            QTextBrowser { border-radius: 8px; }
            QPlainTextEdit { border-radius: 8px; }
            QGraphicsView { border-radius: 8px; }
            QComboBox, QSpinBox { border-radius: 6px; }
            QGroupBox#multimodalHero {
                border: 2px solid #5e35b1;
                border-radius: 14px;
                margin-top: 16px;
                padding: 14px 10px 10px 10px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ede7f6, stop:0.45 #e8eaf6, stop:1 #e3f2fd);
            }
            QGroupBox#multimodalHero::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #311b92;
                font-weight: bold;
                font-size: 12px;
            }
            QFrame#agentHeroFrame {
                background-color: #0f172a;
                border: 2px solid #22d3ee;
                border-radius: 12px;
            }
            QLabel#agentHeroHeading {
                color: #67e8f9;
                font-weight: bold;
                font-size: 11pt;
                padding: 0 2px 4px 2px;
                background: transparent;
            }
            QTextBrowser#labelAgentStatus {
                color: #e2e8f0;
                background-color: #1e293b;
                border: none;
                border-radius: 6px;
                padding: 4px 6px;
            }
            QTextBrowser#mmDetailBrowser, QTextBrowser#mmTranscriptBrowser, QTextBrowser#mmLiveBrowser {
                background-color: rgba(255,255,255,0.95);
                border-radius: 8px;
                padding: 6px;
            }
            """
        )
        
        # 主布局：左右可拖调节；左侧固定工作台（无滚动）；右侧「大预览 + 日志列」
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.setObjectName("mainSplit")
        main_split.setChildrenCollapsible(False)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("leftWorkbench")
        left_panel.setMinimumWidth(340)
        left_panel.setMaximumWidth(620)
        left_panel.setStyleSheet(
            "#leftWorkbench { background-color: #f8fafc; border-radius: 14px; border: 1px solid #e2e8f0; }"
        )
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(12, 12, 12, 14)

        # 顶栏：主标题 + 副标题（信息分区）
        header_wrap = QtWidgets.QWidget()
        header_v = QtWidgets.QVBoxLayout(header_wrap)
        header_v.setContentsMargins(2, 0, 2, 4)
        header_v.setSpacing(2)
        title_label = QtWidgets.QLabel("疲劳驾驶检测系统")
        title_label.setFont(_ui_cn_font(16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #0f172a; letter-spacing: 0.5px;")
        sub_title = QtWidgets.QLabel("实时视觉 · 语音融合 · 主动安全")
        sub_title.setFont(_ui_cn_font(9))
        sub_title.setAlignment(Qt.AlignCenter)
        sub_title.setWordWrap(True)
        sub_title.setStyleSheet("color: #64748b;")
        header_v.addWidget(title_label)
        header_v.addWidget(sub_title)
        left_layout.addWidget(header_wrap)

        # —— 检测与画面：设备、信号源、叠加 ——（原摄像头 + 画面叠加合并）
        device_group = QtWidgets.QGroupBox("检测与画面")
        device_group.setFont(_ui_cn_font(10))
        device_layout = QtWidgets.QVBoxLayout(device_group)
        device_layout.setSpacing(8)
        device_layout.setContentsMargins(10, 12, 10, 8)

        cam_select_layout = QtWidgets.QHBoxLayout()
        cam_label = QtWidgets.QLabel("摄像头")
        cam_label.setFont(_ui_cn_font(10))
        self.Cam_Select = QtWidgets.QComboBox()
        self.Cam_Select.setFont(_ui_cn_font(10))
        self.Cam_Select.addItem("选择摄像头")
        self.Cam_Select.setStyleSheet(
            "QComboBox { padding: 6px 8px; border: 1px solid #94a3b8; border-radius: 8px; background: #fff; }"
        )
        cam_select_layout.addWidget(cam_label)
        cam_select_layout.addWidget(self.Cam_Select, 1)
        device_layout.addLayout(cam_select_layout)

        self.Button_OpenVideo = QtWidgets.QPushButton("打开视频文件")
        self.Button_OpenVideo.setFont(_ui_cn_font(10))
        self.Button_OpenVideo.setMinimumHeight(36)
        self.Button_OpenVideo.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #059669; color: white; border: none; border-radius: 8px; font-weight: bold; } "
            "QPushButton:hover { background-color: #047857; }"
        )
        device_layout.addWidget(self.Button_OpenVideo)

        src_label = QtWidgets.QLabel("信号源")
        src_label.setFont(_ui_cn_font(9))
        src_label.setStyleSheet("color: #64748b; font-weight: bold; margin-top: 2px;")
        device_layout.addWidget(src_label)
        video_cam_layout = QtWidgets.QHBoxLayout()
        video_cam_layout.setSpacing(16)
        self.video = QtWidgets.QRadioButton("视频文件")
        self.video.setFont(_ui_cn_font(10))
        self.cam = QtWidgets.QRadioButton("摄像头")
        self.cam.setFont(_ui_cn_font(10))
        self.cam.setChecked(True)
        video_cam_layout.addWidget(self.video)
        video_cam_layout.addWidget(self.cam)
        video_cam_layout.addStretch(1)
        device_layout.addLayout(video_cam_layout)

        ov_row = QtWidgets.QHBoxLayout()
        ov_row.setSpacing(6)
        ov_label = QtWidgets.QLabel("画面叠加")
        ov_label.setFont(_ui_cn_font(9))
        ov_label.setStyleSheet("color: #64748b; font-weight: bold;")
        ov_row.addWidget(ov_label, 0)
        ov_row.addStretch(1)
        device_layout.addLayout(ov_row)
        chk_row = QtWidgets.QHBoxLayout()
        chk_row.setSpacing(4)
        self.show_eye = QtWidgets.QCheckBox("眼")
        self.show_eye.setFont(_ui_cn_font(9))
        self.show_eye.setChecked(True)
        self.show_eye.setToolTip("在画面上叠加眼部轮廓与开合相关显示")
        self.show_mouth = QtWidgets.QCheckBox("嘴")
        self.show_mouth.setFont(_ui_cn_font(9))
        self.show_mouth.setChecked(True)
        self.show_mouth.setToolTip("在画面上叠加嘴部轮廓与张嘴相关显示")
        self.show_head = QtWidgets.QCheckBox("头")
        self.show_head.setFont(_ui_cn_font(9))
        self.show_head.setToolTip("头部姿态（俯仰等）相关显示")
        self.show_key_point = QtWidgets.QCheckBox("点")
        self.show_key_point.setFont(_ui_cn_font(9))
        self.show_key_point.setToolTip("人脸关键点")
        for _cb in (self.show_eye, self.show_mouth, self.show_head, self.show_key_point):
            _cb.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            chk_row.addWidget(_cb, 1)
        device_layout.addLayout(chk_row)

        left_layout.addWidget(device_group)

        # —— 运行 ——
        control_group = QtWidgets.QGroupBox("运行")
        control_group.setFont(_ui_cn_font(10))
        control_layout = QtWidgets.QVBoxLayout(control_group)
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 14, 10, 10)

        self.Button_Start = QtWidgets.QPushButton("开始检测")
        self.Button_Start.setFont(_ui_cn_font(10))
        self.Button_Start.setMinimumHeight(42)
        self.Button_Start.setStyleSheet(
            "QPushButton { padding: 10px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #2563eb, stop:1 #1d4ed8); "
            "color: white; border: none; border-radius: 10px; font-weight: bold; } "
            "QPushButton:hover { background: #1d4ed8; }"
        )
        self.Button_End = QtWidgets.QPushButton("结束检测")
        self.Button_End.setFont(_ui_cn_font(10))
        self.Button_End.setMinimumHeight(42)
        self.Button_End.setStyleSheet(
            "QPushButton { padding: 10px; background-color: #dc2626; color: white; border: none; border-radius: 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #b91c1c; }"
        )
        start_end_row = QtWidgets.QHBoxLayout()
        start_end_row.setSpacing(8)
        start_end_row.addWidget(self.Button_Start, 1)
        start_end_row.addWidget(self.Button_End, 1)
        control_layout.addLayout(start_end_row)

        self.Button_AdjustCamera_Location = QtWidgets.QPushButton("调整摄像头")
        self.Button_AdjustCamera_Location.setFont(_ui_cn_font(10))
        self.Button_AdjustCamera_Location.setMinimumHeight(38)
        self.Button_AdjustCamera_Location.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #ea580c; color: white; border: none; border-radius: 10px; font-weight: bold; } "
            "QPushButton:hover { background-color: #c2410c; }"
        )
        control_layout.addWidget(self.Button_AdjustCamera_Location)

        left_layout.addWidget(control_group)

        # 多模态（视觉 + 语音）：强视觉层级（具体文案由 MainWindow 刷新）
        multimodal_group = QtWidgets.QGroupBox("多模态 · 视觉 + 语音")
        multimodal_group.setObjectName("multimodalHero")
        multimodal_group.setFont(_ui_cn_font(10))
        mm_layout = QtWidgets.QVBoxLayout(multimodal_group)
        mm_layout.setContentsMargins(6, 10, 6, 6)
        mm_layout.setSpacing(6)

        self.agent_hero_frame = QtWidgets.QFrame()
        self.agent_hero_frame.setObjectName("agentHeroFrame")
        agent_hero_layout = QtWidgets.QVBoxLayout(self.agent_hero_frame)
        agent_hero_layout.setContentsMargins(10, 10, 10, 10)
        agent_hero_layout.setSpacing(6)
        self.label_agent_heading = QtWidgets.QLabel("主动安全 Agent")
        self.label_agent_heading.setObjectName("agentHeroHeading")
        self.label_agent_heading.setFont(_ui_cn_font(11, QFont.Bold))
        self.label_agent_status = QtWidgets.QTextBrowser()
        self.label_agent_status.setObjectName("labelAgentStatus")
        self.label_agent_status.setReadOnly(True)
        self.label_agent_status.setOpenExternalLinks(False)
        self.label_agent_status.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.label_agent_status.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.label_agent_status.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.label_agent_status.setFont(_ui_cn_font(9))
        self.label_agent_status.setMinimumHeight(88)
        self.label_agent_status.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        agent_hero_layout.addWidget(self.label_agent_heading)
        agent_hero_layout.addWidget(self.label_agent_status, 1)
        self.agent_hero_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.MinimumExpanding,
        )

        self.label_multimodal_badge = QtWidgets.QLabel()
        self.label_multimodal_badge.setFont(_ui_cn_font(11, QFont.Bold))
        self.label_multimodal_badge.setAlignment(Qt.AlignCenter)
        self.label_multimodal_badge.setMinimumHeight(44)
        self.label_multimodal_badge.setStyleSheet(
            "color: #1a237e; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #7c4dff, stop:1 #00bcd4); "
            "border-radius: 10px; padding: 10px 8px; font-weight: bold;"
        )

        self.label_multimodal_detail = QtWidgets.QTextBrowser()
        self.label_multimodal_detail.setObjectName("mmDetailBrowser")
        self.label_multimodal_detail.setReadOnly(True)
        self.label_multimodal_detail.setOpenExternalLinks(False)
        self.label_multimodal_detail.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.label_multimodal_detail.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.label_multimodal_detail.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.label_multimodal_detail.setFont(_ui_cn_font(10))
        self.label_multimodal_detail.setMinimumHeight(110)
        self.label_multimodal_detail.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.label_multimodal_detail.setStyleSheet(
            "QTextBrowser#mmDetailBrowser { color: #263238; font-weight: normal; border: 1px solid #b39ddb; }"
        )
        self.label_mm_meter = QtWidgets.QLabel("融合度 —")
        self.label_mm_meter.setFont(_ui_cn_font(10, QFont.Bold))
        self.label_mm_meter.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.label_mm_meter.setMinimumHeight(42)
        self.label_mm_meter.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.label_mm_meter.setStyleSheet(
            "color: #4a148c; padding: 10px 8px; font-weight: bold; "
            "background-color: #f3e5f5; border-radius: 8px; border: 1px solid #ce93d8;"
        )
        self.mm_fused_bar = QtWidgets.QProgressBar()
        self.mm_fused_bar.setRange(0, 100)
        self.mm_fused_bar.setValue(0)
        self.mm_fused_bar.setFormat("%p% 融合疲劳")
        self.mm_fused_bar.setTextVisible(True)
        self.mm_fused_bar.setMinimumHeight(28)
        self.mm_fused_bar.setStyleSheet(
            "QProgressBar { border: 2px solid #5e35b1; border-radius: 10px; height: 26px; text-align: center; "
            "font-weight: bold; font-size: 10pt; color: #1a237e; background-color: #ede7f6; } "
            "QProgressBar::chunk { border-radius: 8px; "
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #d500f9, stop:0.5 #7c4dff, stop:1 #00e5ff); }"
        )
        self.label_mm_transcript = QtWidgets.QTextBrowser()
        self.label_mm_transcript.setObjectName("mmTranscriptBrowser")
        self.label_mm_transcript.setReadOnly(True)
        self.label_mm_transcript.setOpenExternalLinks(False)
        self.label_mm_transcript.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.label_mm_transcript.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.label_mm_transcript.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.label_mm_transcript.setFont(_ui_cn_font(10))
        self.label_mm_transcript.setPlainText("▶ 本轮转写\n—")
        self.label_mm_transcript.setMinimumHeight(72)
        self.label_mm_transcript.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.label_mm_transcript.setStyleSheet(
            "QTextBrowser#mmTranscriptBrowser { color: #311b92; border: 1px solid #9575cd; "
            "background-color: #ede7f6; font-weight: normal; }"
        )
        self.label_multimodal_live = QtWidgets.QTextBrowser()
        self.label_multimodal_live.setObjectName("mmLiveBrowser")
        self.label_multimodal_live.setReadOnly(True)
        self.label_multimodal_live.setOpenExternalLinks(False)
        self.label_multimodal_live.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.label_multimodal_live.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.label_multimodal_live.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        self.label_multimodal_live.setFont(_ui_cn_font(10, QFont.Bold))
        self.label_multimodal_live.setMinimumHeight(72)
        self.label_multimodal_live.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.label_multimodal_live.setStyleSheet(
            "QTextBrowser#mmLiveBrowser { color: #0d47a1; border: 1px solid #29b6f6; "
            "background-color: #e1f5fe; font-weight: bold; }"
        )
        multimodal_group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        mm_layout.addWidget(self.agent_hero_frame, 0)
        mm_layout.addWidget(self.label_multimodal_badge, 0)
        mm_layout.addWidget(self.label_multimodal_detail, 3)
        mm_layout.addWidget(self.label_mm_meter, 0)
        mm_layout.addWidget(self.mm_fused_bar, 0)
        mm_layout.addWidget(self.label_mm_transcript, 2)
        mm_layout.addWidget(self.label_multimodal_live, 2)
        left_layout.addWidget(multimodal_group, 1)

        offduty_group = QtWidgets.QGroupBox("脱岗提醒")
        offduty_group.setFont(_ui_cn_font(10))
        offduty_layout = QtWidgets.QVBoxLayout(offduty_group)
        offduty_layout.setSpacing(8)
        offduty_layout.setContentsMargins(10, 14, 10, 10)
        self.offDuty_Check = QtWidgets.QCheckBox("启用脱岗检测")
        self.offDuty_Check.setFont(_ui_cn_font(10))
        offduty_layout.addWidget(self.offDuty_Check)
        offduty_time_layout = QtWidgets.QHBoxLayout()
        offduty_time_label = QtWidgets.QLabel("离岗阈值(秒)")
        offduty_time_label.setFont(_ui_cn_font(10))
        self.offDuty_Time = QtWidgets.QSpinBox()
        self.offDuty_Time.setFont(_ui_cn_font(10))
        self.offDuty_Time.setRange(1, 60)
        self.offDuty_Time.setValue(5)
        self.offDuty_Time.setStyleSheet(
            "QSpinBox { padding: 6px 8px; border: 1px solid #94a3b8; border-radius: 8px; background: #fff; }"
        )
        offduty_time_layout.addWidget(offduty_time_label)
        offduty_time_layout.addWidget(self.offDuty_Time, 1)
        offduty_layout.addLayout(offduty_time_layout)
        left_layout.addWidget(offduty_group)

        main_split.addWidget(left_panel)

        # 右侧：横向「实时预览 | 语音+系统日志」；日志区内部可竖向调节比例
        right_panel = QtWidgets.QWidget()
        right_outer = QtWidgets.QHBoxLayout(right_panel)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(0)

        right_h_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        right_h_split.setObjectName("rightHSplit")
        right_h_split.setChildrenCollapsible(False)

        video_group = QtWidgets.QGroupBox("实时预览")
        video_group.setFont(_ui_cn_font(10))
        video_layout = QtWidgets.QVBoxLayout(video_group)
        video_layout.setContentsMargins(10, 14, 10, 10)
        self.graphicsView = QtWidgets.QGraphicsView()
        self.graphicsView.setMinimumHeight(260)
        self.graphicsView.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.graphicsView.setStyleSheet(
            "QGraphicsView { border: 1px solid #64748b; border-radius: 10px; background-color: #0f172a; }"
        )
        video_layout.addWidget(self.graphicsView)

        vo_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vo_split.setObjectName("rightVSplit")
        vo_split.setChildrenCollapsible(False)

        voice_group = QtWidgets.QGroupBox("语音与 Agent 流水")
        voice_group.setFont(_ui_cn_font(10))
        voice_layout = QtWidgets.QVBoxLayout(voice_group)
        voice_layout.setContentsMargins(10, 14, 10, 10)
        self.voice_log_timeline = QtWidgets.QPlainTextEdit()
        self.voice_log_timeline.setReadOnly(True)
        _mono = QFont("Cascadia Mono", 10)
        if not QFontInfo(_mono).exactMatch():
            _mono = QFont("Consolas", 10)
        self.voice_log_timeline.setFont(_mono)
        self.voice_log_timeline.setMinimumHeight(72)
        self.voice_log_timeline.setPlaceholderText("每轮 Whisper 转写、疲劳分与 Agent 动作将追加到此…")
        self.voice_log_timeline.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #38bdf8; border-radius: 10px; background-color: #f8fafc; padding: 8px; }"
        )
        voice_layout.addWidget(self.voice_log_timeline)

        output_group = QtWidgets.QGroupBox("系统日志")
        output_group.setFont(_ui_cn_font(10))
        output_layout = QtWidgets.QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 14, 10, 10)
        self.output_Window = QtWidgets.QTextBrowser()
        _mono2 = QFont("Cascadia Mono", 10)
        if not QFontInfo(_mono2).exactMatch():
            _mono2 = QFont("Consolas", 10)
        self.output_Window.setFont(_mono2)
        self.output_Window.setMinimumHeight(72)
        self.output_Window.setStyleSheet(
            "QTextBrowser { border: 1px solid #cbd5e1; border-radius: 10px; background-color: #ffffff; padding: 6px; }"
        )
        output_layout.addWidget(self.output_Window)

        vo_split.addWidget(voice_group)
        vo_split.addWidget(output_group)
        vo_split.setStretchFactor(0, 2)
        vo_split.setStretchFactor(1, 3)
        vo_split.setSizes([200, 280])

        right_h_split.addWidget(video_group)
        right_h_split.addWidget(vo_split)
        right_h_split.setStretchFactor(0, 5)
        right_h_split.setStretchFactor(1, 2)
        right_h_split.setSizes([960, 480])

        right_outer.addWidget(right_h_split)
        main_split.addWidget(right_panel)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([500, 1420])

        main_layout.addWidget(main_split)
        
        # 设置中心部件
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 创建状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        # 调用retranslateUi方法设置文本
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        #_translate 是一个用于翻译文本的函数，用于为界面的各个元素设置文本
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "疲劳驾驶检测系统"))
        # 尝试不同的路径
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        MainWindow.setWindowIcon(QIcon(icon_path))
        self.Cam_Select.setItemText(0, _translate("MainWindow", "选择摄像头"))
        self.Button_OpenVideo.setText(_translate("MainWindow", "打开视频文件"))
        self.Button_Start.setText(_translate("MainWindow", "开始检测"))
        self.Button_End.setText(_translate("MainWindow", "结束检测"))
        self.Button_AdjustCamera_Location.setText(_translate("MainWindow", "调整摄像头"))
        self.offDuty_Check.setText(_translate("MainWindow", "启用脱岗检测"))
        self.video.setText(_translate("MainWindow", "视频文件"))
        self.cam.setText(_translate("MainWindow", "摄像头"))
        self.show_eye.setText(_translate("MainWindow", "眼"))
        self.show_mouth.setText(_translate("MainWindow", "嘴"))
        self.show_head.setText(_translate("MainWindow", "头"))
        self.show_key_point.setText(_translate("MainWindow", "点"))
        self.label_agent_heading.setText(_translate("MainWindow", "主动安全 Agent"))
        self.statusbar.showMessage(_translate("MainWindow", "就绪"))
    def paintEvent(self, event):
        # 创建一个QPainter对象
        painter = QPainter(self)
        # 加载背景图片，尝试不同的路径
        background_path = 'resources/images/bkg.jpg'
        if not os.path.exists(background_path):
            background_path = 'bkg.jpg'
        pixmap = QPixmap(background_path)
        # 绘制背景图片
        painter.drawPixmap(self.rect(), pixmap)

