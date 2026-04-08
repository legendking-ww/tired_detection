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

        #设置MainWindow对象的大小为1200x800 像素。
        MainWindow.resize(1200, 800)
        #设置窗口图标
        MainWindow.setWindowIcon(QIcon("resources/images/yjwj.png"))
        
        #创建一个QWidget对象，作为MainWindow对象的子控件
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        #设置中心部件的背景颜色
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.centralwidget.setPalette(palette)
        
        # 创建主布局
        main_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        
        # 创建左侧控制面板
        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(350)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建标题标签
        title_label = QtWidgets.QLabel("疲劳驾驶检测系统")
        title_font = QFont()
        title_font.setFamily("微软雅黑")
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333;")
        left_layout.addWidget(title_label)
        
        # 创建分隔线
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #CCCCCC;")
        left_layout.addWidget(line)
        
        # 创建摄像头选择区域
        cam_group = QtWidgets.QGroupBox("摄像头设置")
        cam_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
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
        control_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
        control_layout = QtWidgets.QVBoxLayout(control_group)
        control_layout.setSpacing(15)
        
        # 开始检测按钮
        self.Button_Start = QtWidgets.QPushButton("开始检测")
        self.Button_Start.setFont(QFont("微软雅黑", 10))
        self.Button_Start.setStyleSheet("QPushButton { padding: 10px; background-color: #2196F3; color: white; border: none; border-radius: 4px; font-weight: bold; }")
        control_layout.addWidget(self.Button_Start)
        
        # 结束检测按钮
        self.Button_End = QtWidgets.QPushButton("结束检测")
        self.Button_End.setFont(QFont("微软雅黑", 10))
        self.Button_End.setStyleSheet("QPushButton { padding: 10px; background-color: #f44336; color: white; border: none; border-radius: 4px; font-weight: bold; }")
        control_layout.addWidget(self.Button_End)
        
        # 调整摄像头按钮
        self.Button_AdjustCamera_Location = QtWidgets.QPushButton("调整摄像头")
        self.Button_AdjustCamera_Location.setFont(QFont("微软雅黑", 9))
        self.Button_AdjustCamera_Location.setStyleSheet("QPushButton { padding: 8px; background-color: #ff9800; color: white; border: none; border-radius: 4px; }")
        control_layout.addWidget(self.Button_AdjustCamera_Location)
        
        left_layout.addWidget(control_group)
        
        # 创建脱岗检测区域
        offduty_group = QtWidgets.QGroupBox("脱岗检测")
        offduty_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
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
        display_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
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
        
        # 添加伸缩空间
        left_layout.addStretch()
        
        # 创建右侧显示区域
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建视频显示区域
        video_group = QtWidgets.QGroupBox("视频显示")
        video_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
        video_layout = QtWidgets.QVBoxLayout(video_group)
        
        # 创建GraphicsView用于显示视频
        self.graphicsView = QtWidgets.QGraphicsView()
        self.graphicsView.setStyleSheet("QGraphicsView { border: 1px solid #CCCCCC; border-radius: 4px; background-color: #000000; }")
        video_layout.addWidget(self.graphicsView)
        right_layout.addWidget(video_group)
        
        # 创建输出信息区域
        output_group = QtWidgets.QGroupBox("系统信息")
        output_group.setFont(QFont("微软雅黑", 10, QFont.Bold))
        output_layout = QtWidgets.QVBoxLayout(output_group)
        
        # 创建文本浏览器用于显示输出信息
        self.output_Window = QtWidgets.QTextBrowser()
        self.output_Window.setFont(QFont("Consolas", 9))
        self.output_Window.setStyleSheet("QTextBrowser { border: 1px solid #CCCCCC; border-radius: 4px; background-color: #FFFFFF; }")
        output_layout.addWidget(self.output_Window)
        right_layout.addWidget(output_group)
        
        # 将左右面板添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
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
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "疲劳驾驶检测系统"))
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
        font.setPointSize(8)
        self.Button_AdjustCamera_Location.setFont(font)
        self.Button_AdjustCamera_Location.setObjectName("Button_AdjustCamera_Location")

        #创建widget部件，并将其添加到centralwidget部件中
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(0, 160, 141, 31))
        self.widget.setObjectName("widget")

        #创建了一个名为 video 的复选框，并将其添加到widget部件中
        self.video = QtWidgets.QCheckBox(self.widget)
        self.video.setGeometry(QtCore.QRect(10, 0, 87, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.video.setFont(font)
        self.video.setChecked(False)
        self.video.setAutoExclusive(True)
        self.video.setObjectName("video")

        #创建cam复选框
        self.cam = QtWidgets.QCheckBox(self.widget)
        self.cam.setGeometry(QtCore.QRect(70, 0, 87, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.cam.setFont(font)
        #设置了 cam 复选框的初始选中状态为选中。
        self.cam.setChecked(True)
        #设置了 cam 复选框的自动排他性，即当一个复选框被选中时，其他复选框会自动取消选中。
        self.cam.setAutoExclusive(True)
        self.cam.setObjectName("cam")

        #设置show_key_point复选框
        self.show_key_point = QtWidgets.QCheckBox(self.centralwidget)
        self.show_key_point.setGeometry(QtCore.QRect(10, 270, 87, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.show_key_point.setFont(font)
        self.show_key_point.setObjectName("show_key_point")

        #设置line_6线
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(80, 250, 201, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")

        #设置label_8标签
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(10, 250, 71, 16))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")

        #创建show_head复选框
        self.show_head = QtWidgets.QCheckBox(self.centralwidget)
        self.show_head.setGeometry(QtCore.QRect(220, 270, 61, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(8)
        self.show_head.setFont(font)
        self.show_head.setObjectName("show_head")

        #创建show_eye复选框
        self.show_eye = QtWidgets.QCheckBox(self.centralwidget)
        self.show_eye.setGeometry(QtCore.QRect(90, 270, 51, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(8)
        self.show_eye.setFont(font)
        self.show_eye.setObjectName("show_eye")
        self.show_eye.setChecked(True)

        #创建show_mouth复选框
        self.show_mouth = QtWidgets.QCheckBox(self.centralwidget)
        self.show_mouth.setGeometry(QtCore.QRect(160, 270, 51, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(8)
        self.show_mouth.setFont(font)
        self.show_mouth.setObjectName("show_mouth")
        self.show_mouth.setChecked(True)

        #将 MainWindow 的中心部件设置为 centralwidget。
        MainWindow.setCentralWidget(self.centralwidget)
        #为 actionExit 操作对象设置一个唯一对象名称。
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        #调用 retranslateUi 方法为 MainWindow 设置界面的文本。
        self.retranslateUi(MainWindow)
        #调用 QtCore.QMetaObject.connectSlotsByName 方法将对象和槽函数进行连接。
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #这段代码是用于设置界面的文本和按钮的显示文字
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

