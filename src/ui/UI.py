#导入pyqt5模块
import os
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import Qt
##QtCore模块包含了一些核心的非图形功能，例如事件循环和信号槽机制。
#QtGui模块包含了一些基本的图形功能，例如字体、颜色和绘图工具。
#QtWidgets模块包含了一些GUI元素，例如窗口、标签、按钮、复选框等，用于创建用户界面。

#用于设计主窗口。创建各个控件，标签，线，按钮，复选框等，并指定了它们的字体，位置等属性
class Ui_MainWindow(object):
    #定义setupUi方法，该方法接收一个MainWindow对象作为参数。
    def setupUi(self, MainWindow):
        #给这个实例命了个名字（唯一标识符）。方便类外界访问。
        MainWindow.setObjectName("MainWindow")

        MainWindow.resize(1760, 1000)
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
            QMainWindow { background-color: #eef2f7; }
            /* 勿对 QGroupBox 本体设 font-weight:bold，否则会继承到组内所有控件，中文易糊、挤 */
            QGroupBox {
                font-weight: normal;
                border: 1px solid #c5d4e8;
                border-radius: 10px;
                margin-top: 14px;
                padding: 16px 10px 10px 10px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #1a365d;
                font-weight: bold;
            }
            QPushButton { border-radius: 8px; padding: 6px; }
            QTextBrowser { border-radius: 8px; }
            QPlainTextEdit { border-radius: 8px; }
            QGraphicsView { border-radius: 8px; }
            QComboBox, QSpinBox { border-radius: 6px; }
            """
        )
        
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左侧控制面板（加宽、无滚动区，避免文字被挤裁切）
        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(340)
        left_panel.setMaximumWidth(520)
        left_panel.setStyleSheet("background-color: #f8fafc; border-radius: 12px;")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(18, 18, 18, 18)
        
        # 创建标题标签
        title_label = QtWidgets.QLabel("疲劳驾驶检测系统")
        title_font = QFont()
        title_font.setFamily("微软雅黑")
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #1e3a5f; letter-spacing: 1px;")
        left_layout.addWidget(title_label)
        
        # 创建分隔线
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #CCCCCC;")
        left_layout.addWidget(line)
        
        # 创建摄像头选择区域
        cam_group = QtWidgets.QGroupBox("摄像头设置")
        cam_group.setFont(QFont("微软雅黑", 10))
        cam_layout = QtWidgets.QVBoxLayout(cam_group)
        cam_layout.setSpacing(15)
        
        # 摄像头选择下拉框
        cam_select_layout = QtWidgets.QHBoxLayout()
        cam_label = QtWidgets.QLabel("选择摄像头:")
        cam_label.setFont(QFont("微软雅黑", 9))
        self.Cam_Select = QtWidgets.QComboBox()
        self.Cam_Select.setFont(QFont("微软雅黑", 9))
        self.Cam_Select.addItem("选择摄像头")
        self.Cam_Select.setStyleSheet("QComboBox { padding: 5px; border: 1px solid #CCCCCC; border-radius: 4px; }")
        cam_select_layout.addWidget(cam_label)
        cam_select_layout.addWidget(self.Cam_Select)
        cam_layout.addLayout(cam_select_layout)
        
        # 打开视频按钮
        self.Button_OpenVideo = QtWidgets.QPushButton("打开视频文件")
        self.Button_OpenVideo.setFont(QFont("微软雅黑", 9))
        self.Button_OpenVideo.setStyleSheet("QPushButton { padding: 8px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; } QPushButton:hover { background-color: #45a049; }")
        cam_layout.addWidget(self.Button_OpenVideo)
        
        left_layout.addWidget(cam_group)
        
        # 创建检测控制区域
        control_group = QtWidgets.QGroupBox("检测控制")
        control_group.setFont(QFont("微软雅黑", 10))
        control_layout = QtWidgets.QVBoxLayout(control_group)
        control_layout.setSpacing(12)
        
        self.Button_Start = QtWidgets.QPushButton("开始检测")
        self.Button_Start.setFont(QFont("微软雅黑", 10))
        self.Button_Start.setMinimumHeight(40)
        self.Button_Start.setStyleSheet(
            "QPushButton { padding: 10px; background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1976d2, stop:1 #0d47a1); "
            "color: white; border: none; border-radius: 8px; font-weight: bold; } "
            "QPushButton:hover { background: #1565c0; }"
        )
        self.Button_End = QtWidgets.QPushButton("结束检测")
        self.Button_End.setFont(QFont("微软雅黑", 10))
        self.Button_End.setMinimumHeight(40)
        self.Button_End.setStyleSheet(
            "QPushButton { padding: 10px; background-color: #c62828; color: white; border: none; border-radius: 8px; font-weight: bold; } "
            "QPushButton:hover { background-color: #b71c1c; }"
        )
        start_end_row = QtWidgets.QHBoxLayout()
        start_end_row.setSpacing(10)
        start_end_row.addWidget(self.Button_Start, 1)
        start_end_row.addWidget(self.Button_End, 1)
        control_layout.addLayout(start_end_row)
        
        self.Button_AdjustCamera_Location = QtWidgets.QPushButton("调整摄像头")
        self.Button_AdjustCamera_Location.setFont(QFont("微软雅黑", 9))
        self.Button_AdjustCamera_Location.setMinimumHeight(36)
        self.Button_AdjustCamera_Location.setStyleSheet(
            "QPushButton { padding: 8px; background-color: #ff9800; color: white; border: none; border-radius: 8px; font-weight: bold; } "
            "QPushButton:hover { background-color: #f57c00; }"
        )
        control_layout.addWidget(self.Button_AdjustCamera_Location)
        
        left_layout.addWidget(control_group)
        
        # 创建脱岗检测区域
        offduty_group = QtWidgets.QGroupBox("脱岗检测")
        offduty_group.setFont(QFont("微软雅黑", 10))
        offduty_layout = QtWidgets.QVBoxLayout(offduty_group)
        offduty_layout.setSpacing(10)
        
        # 脱岗检测复选框
        self.offDuty_Check = QtWidgets.QCheckBox("启用脱岗检测")
        self.offDuty_Check.setFont(QFont("微软雅黑", 9))
        offduty_layout.addWidget(self.offDuty_Check)
        
        # 脱岗时间设置
        offduty_time_layout = QtWidgets.QHBoxLayout()
        offduty_time_label = QtWidgets.QLabel("脱岗时间(秒):")
        offduty_time_label.setFont(QFont("微软雅黑", 9))
        self.offDuty_Time = QtWidgets.QSpinBox()
        self.offDuty_Time.setFont(QFont("微软雅黑", 9))
        self.offDuty_Time.setRange(1, 60)
        self.offDuty_Time.setValue(5)
        self.offDuty_Time.setStyleSheet("QSpinBox { padding: 5px; border: 1px solid #CCCCCC; border-radius: 4px; }")
        offduty_time_layout.addWidget(offduty_time_label)
        offduty_time_layout.addWidget(self.offDuty_Time)
        offduty_layout.addLayout(offduty_time_layout)
        
        left_layout.addWidget(offduty_group)
        
        # 创建显示选项区域
        display_group = QtWidgets.QGroupBox("显示选项")
        display_group.setFont(QFont("微软雅黑", 10))
        display_layout = QtWidgets.QVBoxLayout(display_group)
        display_layout.setSpacing(8)
        
        # 视频/摄像头切换
        video_cam_layout = QtWidgets.QHBoxLayout()
        self.video = QtWidgets.QRadioButton("视频文件")
        self.video.setFont(QFont("微软雅黑", 9))
        self.cam = QtWidgets.QRadioButton("摄像头")
        self.cam.setFont(QFont("微软雅黑", 9))
        self.cam.setChecked(True)
        video_cam_layout.addWidget(self.video)
        video_cam_layout.addWidget(self.cam)
        display_layout.addLayout(video_cam_layout)
        
        # 显示选项复选框
        self.show_eye = QtWidgets.QCheckBox("显示眼睛")
        self.show_eye.setFont(QFont("微软雅黑", 9))
        self.show_eye.setChecked(True)
        display_layout.addWidget(self.show_eye)
        
        self.show_mouth = QtWidgets.QCheckBox("显示嘴巴")
        self.show_mouth.setFont(QFont("微软雅黑", 9))
        self.show_mouth.setChecked(True)
        display_layout.addWidget(self.show_mouth)
        
        self.show_head = QtWidgets.QCheckBox("显示头部姿态")
        self.show_head.setFont(QFont("微软雅黑", 9))
        display_layout.addWidget(self.show_head)
        
        self.show_key_point = QtWidgets.QCheckBox("显示关键点")
        self.show_key_point.setFont(QFont("微软雅黑", 9))
        display_layout.addWidget(self.show_key_point)
        
        left_layout.addWidget(display_group)

        # 多模态（视觉 + 语音）状态区：具体文案由 MainWindow 根据 .env 刷新
        multimodal_group = QtWidgets.QGroupBox("多模态 · 视觉 + 语音")
        multimodal_group.setFont(QFont("微软雅黑", 10))
        mm_layout = QtWidgets.QVBoxLayout(multimodal_group)
        mm_layout.setContentsMargins(6, 10, 6, 6)
        mm_layout.setSpacing(10)
        self.label_multimodal_badge = QtWidgets.QLabel()
        self.label_multimodal_badge.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.label_multimodal_badge.setAlignment(Qt.AlignCenter)
        self.label_multimodal_badge.setMinimumHeight(44)
        self.label_multimodal_detail = QtWidgets.QLabel()
        self.label_multimodal_detail.setFont(QFont("微软雅黑", 9))
        self.label_multimodal_detail.setWordWrap(True)
        self.label_multimodal_detail.setTextFormat(QtCore.Qt.PlainText)
        self.label_multimodal_detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_multimodal_detail.setMinimumHeight(128)
        self.label_multimodal_detail.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.label_multimodal_detail.setStyleSheet("color: #455a64; font-weight: normal;")
        self.label_mm_meter = QtWidgets.QLabel("融合度 —")
        self.label_mm_meter.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.label_mm_meter.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.label_mm_meter.setMinimumHeight(46)
        self.label_mm_meter.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.label_mm_meter.setStyleSheet(
            "color: #0d47a1; padding: 8px 6px; font-weight: bold; min-height: 40px;"
        )
        self.mm_fused_bar = QtWidgets.QProgressBar()
        self.mm_fused_bar.setRange(0, 100)
        self.mm_fused_bar.setValue(0)
        self.mm_fused_bar.setFormat("%p% 融合疲劳")
        self.mm_fused_bar.setTextVisible(True)
        self.mm_fused_bar.setStyleSheet(
            "QProgressBar { border: 2px solid #90caf9; border-radius: 8px; height: 22px; text-align: center; } "
            "QProgressBar::chunk { border-radius: 6px; background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #42a5f5, stop:1 #7e57c2); }"
        )
        self.label_mm_transcript = QtWidgets.QLabel("▶ 本轮转写\n—")
        self.label_mm_transcript.setFont(QFont("微软雅黑", 9))
        self.label_mm_transcript.setWordWrap(True)
        self.label_mm_transcript.setMinimumHeight(72)
        self.label_mm_transcript.setTextFormat(QtCore.Qt.PlainText)
        self.label_mm_transcript.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_mm_transcript.setStyleSheet(
            "color: #263238; background-color: #e3f2fd; border: 1px solid #90caf9; "
            "border-radius: 8px; padding: 8px; font-weight: normal;"
        )
        self.label_multimodal_live = QtWidgets.QLabel()
        self.label_multimodal_live.setFont(QFont("微软雅黑", 9))
        self.label_multimodal_live.setWordWrap(True)
        self.label_multimodal_live.setTextFormat(QtCore.Qt.PlainText)
        self.label_multimodal_live.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label_multimodal_live.setMinimumHeight(64)
        self.label_multimodal_live.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.label_multimodal_live.setStyleSheet("color: #1565c0; font-weight: normal;")
        mm_layout.addWidget(self.label_multimodal_badge)
        mm_layout.addWidget(self.label_multimodal_detail)
        mm_layout.addWidget(self.label_mm_meter)
        mm_layout.addWidget(self.mm_fused_bar)
        mm_layout.addWidget(self.label_mm_transcript)
        mm_layout.addWidget(self.label_multimodal_live)
        left_layout.addWidget(multimodal_group)
        
        # 添加伸缩空间
        left_layout.addStretch()

        # 创建右侧显示区域（可拖动分割：视频 / 语音流水 / 系统日志）
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(0)

        video_group = QtWidgets.QGroupBox("实时检测画面")
        video_group.setFont(QFont("微软雅黑", 10))
        video_layout = QtWidgets.QVBoxLayout(video_group)

        self.graphicsView = QtWidgets.QGraphicsView()
        self.graphicsView.setMinimumHeight(300)
        self.graphicsView.setStyleSheet(
            "QGraphicsView { border: 1px solid #b0bec5; border-radius: 8px; background-color: #0d1117; }"
        )
        video_layout.addWidget(self.graphicsView)

        voice_group = QtWidgets.QGroupBox("语音识别流水（按时间）")
        voice_group.setFont(QFont("微软雅黑", 10))
        voice_layout = QtWidgets.QVBoxLayout(voice_group)
        voice_layout.setContentsMargins(10, 14, 10, 10)
        self.voice_log_timeline = QtWidgets.QPlainTextEdit()
        self.voice_log_timeline.setReadOnly(True)
        self.voice_log_timeline.setFont(QFont("Consolas", 9))
        self.voice_log_timeline.setMinimumHeight(100)
        self.voice_log_timeline.setMaximumHeight(220)
        self.voice_log_timeline.setPlaceholderText("每轮 Whisper 转写与疲劳分将追加到此…")
        self.voice_log_timeline.setStyleSheet(
            "QPlainTextEdit { border: 1px solid #81d4fa; border-radius: 8px; background-color: #fafdff; padding: 6px; }"
        )
        voice_layout.addWidget(self.voice_log_timeline)

        output_group = QtWidgets.QGroupBox("系统运行日志")
        output_group.setFont(QFont("微软雅黑", 10))
        output_layout = QtWidgets.QVBoxLayout(output_group)

        self.output_Window = QtWidgets.QTextBrowser()
        self.output_Window.setFont(QFont("Consolas", 9))
        self.output_Window.setMinimumHeight(100)
        self.output_Window.setStyleSheet(
            "QTextBrowser { border: 1px solid #cfd8dc; border-radius: 8px; background-color: #ffffff; padding: 4px; }"
        )
        output_layout.addWidget(self.output_Window)

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(video_group)
        right_splitter.addWidget(voice_group)
        right_splitter.addWidget(output_group)
        right_splitter.setStretchFactor(0, 5)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setStretchFactor(2, 3)
        right_splitter.setSizes([520, 150, 210])
        right_layout.addWidget(right_splitter)
        
        main_layout.addWidget(left_panel, 0)
        main_layout.addWidget(right_panel, 1)
        
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
        self.show_eye.setText(_translate("MainWindow", "显示眼睛"))
        self.show_mouth.setText(_translate("MainWindow", "显示嘴巴"))
        self.show_head.setText(_translate("MainWindow", "显示头部姿态"))
        self.show_key_point.setText(_translate("MainWindow", "显示关键点"))
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

