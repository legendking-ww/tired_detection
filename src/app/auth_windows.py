"""登录、注册与人脸采集窗口。"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QFont
from PyQt5.QtWidgets import (
    QWidget,
    QFrame,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QListWidget,
    QSizePolicy,
)

from src.core.face_recognition import FaceRecognition
from src.utils.cv_helpers import draw_text_cn_on_bgr, open_video_capture_by_index
from src.app.main_window import MainWindow

# 登录 / 注册共用：毛玻璃半透明卡片与输入框（背景图透出）
_AUTH_GLASS_CARD_QSS = (
    "QFrame { background-color: rgba(255, 255, 255, 0.42); border-radius: 20px; "
    "border: 1px solid rgba(255, 255, 255, 0.38); }"
)
_AUTH_FIELD_QSS = (
    "QLineEdit { padding: 12px 14px; border: 2px solid rgba(69, 90, 100, 0.58); border-radius: 10px; "
    "font-size: 15px; background: rgba(255, 255, 255, 0.88); min-height: 22px; color: #102027; "
    "selection-background-color: #1976d2; selection-color: #ffffff; } "
    "QLineEdit:hover:!focus { border: 2px solid rgba(25, 118, 210, 0.55); background: rgba(255, 255, 255, 0.95); } "
    "QLineEdit:focus { border: 2px solid #1565c0; background: #ffffff; } "
    "QLineEdit:disabled { color: #9e9e9e; background: rgba(245, 245, 245, 0.85); border-color: #bdbdbd; }"
)


def _auth_labeled_field(caption: str, editor: QLineEdit) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    lab = QtWidgets.QLabel(caption)
    lab.setFont(QFont("微软雅黑", 10))
    lab.setStyleSheet("color: rgba(38, 50, 56, 0.92);")
    v.addWidget(lab)
    v.addWidget(editor)
    return w


def _center_widget_on_screen(widget: QWidget) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    scr = app.primaryScreen()
    if scr is None:
        return
    ag = scr.availableGeometry()
    fg = widget.frameGeometry()
    fg.moveCenter(ag.center())
    widget.move(fg.topLeft())


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_recognition = FaceRecognition('mrsoft.db')
        self.face_login_running = False
        self._center_once = False
        self._face_cap = None
        self._face_login_timer: QTimer | None = None
        self._face_recognize_interval = max(0.25, float(os.environ.get("TIRED_FACE_LOGIN_RECOGNIZE_SEC", "0.4")))
        self._last_face_recognize_mono = 0.0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统')
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))
        self.resize(1020, 680)

        main_layout = QHBoxLayout()
        main_layout.setSpacing(36)
        main_layout.setContentsMargins(40, 36, 40, 36)

        # —— 左侧：账号登录卡片 ——
        left_card = QFrame()
        left_card.setStyleSheet(_AUTH_GLASS_CARD_QSS)
        left_card.setMinimumWidth(360)
        left_card.setMaximumWidth(440)
        left_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_inner = QVBoxLayout(left_card)
        left_inner.setContentsMargins(32, 36, 32, 36)
        left_inner.setSpacing(0)

        title_label = QtWidgets.QLabel('用户登录')
        tf = QFont('微软雅黑', 22, QFont.Bold)
        title_label.setFont(tf)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('color: #1a365d; letter-spacing: 1px;')
        left_inner.addWidget(title_label)

        sub_title = QtWidgets.QLabel('疲劳驾驶检测系统')
        sub_title.setFont(QFont('微软雅黑', 10))
        sub_title.setAlignment(Qt.AlignCenter)
        sub_title.setStyleSheet('color: rgba(69, 90, 100, 0.9); margin-top: 4px;')
        left_inner.addWidget(sub_title)
        left_inner.addSpacing(28)

        self.username_line = QLineEdit()
        self.username_line.setPlaceholderText('请输入用户名')
        self.username_line.setMinimumHeight(44)
        self.username_line.setStyleSheet(_AUTH_FIELD_QSS)
        left_inner.addWidget(_auth_labeled_field('用户名', self.username_line))
        left_inner.addSpacing(18)

        self.password_line = QLineEdit()
        self.password_line.setPlaceholderText('请输入密码')
        self.password_line.setEchoMode(QLineEdit.Password)
        self.password_line.setMinimumHeight(44)
        self.password_line.setStyleSheet(_AUTH_FIELD_QSS)
        left_inner.addWidget(_auth_labeled_field('密码', self.password_line))
        left_inner.addSpacing(28)

        self.login_button = QPushButton('  账号登录  ')
        self.login_button.setFont(QFont('微软雅黑', 11, QFont.Bold))
        self.login_button.setMinimumHeight(46)
        self.login_button.setCursor(Qt.PointingHandCursor)
        self.login_button.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #42a5f5, stop:1 #1976d2); "
            "color: white; border: none; border-radius: 12px; } "
            "QPushButton:hover { background: #1e88e5; } QPushButton:pressed { background: #1565c0; }"
        )
        self.login_button.clicked.connect(self.login)
        left_inner.addWidget(self.login_button)
        left_inner.addSpacing(12)

        self.face_login_button = QPushButton('  人脸识别登录  ')
        self.face_login_button.setFont(QFont('微软雅黑', 11, QFont.Bold))
        self.face_login_button.setMinimumHeight(46)
        self.face_login_button.setCursor(Qt.PointingHandCursor)
        self.face_login_button.setStyleSheet(
            "QPushButton { background-color: #43a047; color: white; border: none; border-radius: 12px; } "
            "QPushButton:hover { background-color: #388e3c; } QPushButton:pressed { background-color: #2e7d32; }"
        )
        self.face_login_button.clicked.connect(self.face_login)
        left_inner.addWidget(self.face_login_button)
        left_inner.addSpacing(20)

        self.register_button = QPushButton('还没有账号？注册新用户')
        self.register_button.setFont(QFont('微软雅黑', 10))
        self.register_button.setMinimumHeight(40)
        self.register_button.setCursor(Qt.PointingHandCursor)
        self.register_button.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.22); color: #0d47a1; border: 1px solid rgba(144,202,249,0.65); "
            "border-radius: 10px; } "
            "QPushButton:hover { background-color: rgba(227, 242, 253, 0.72); color: #01579b; }"
        )
        left_inner.addWidget(self.register_button)
        left_inner.addStretch(1)

        # —— 右侧：人脸预览卡片 ——
        right_card = QFrame()
        right_card.setStyleSheet(_AUTH_GLASS_CARD_QSS)
        right_inner = QVBoxLayout(right_card)
        right_inner.setContentsMargins(28, 28, 28, 28)
        right_inner.setSpacing(14)

        preview_title = QtWidgets.QLabel('人脸登录 · 实时预览')
        preview_title.setFont(QFont('微软雅黑', 12, QFont.Bold))
        preview_title.setAlignment(Qt.AlignCenter)
        preview_title.setStyleSheet('color: rgba(55, 71, 79, 0.95);')
        right_inner.addWidget(preview_title)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(420, 315)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(
            'QLabel { border: 2px solid rgba(255,255,255,0.35); border-radius: 12px; '
            'background-color: rgba(13, 17, 23, 0.82); color: rgba(144, 164, 174, 0.95); }'
        )
        self.camera_label.setText('预览区')
        right_inner.addWidget(self.camera_label, 0, Qt.AlignHCenter)

        self.face_status_label = QtWidgets.QLabel('请点击左侧「人脸识别登录」')
        self.face_status_label.setFont(QFont('微软雅黑', 11))
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.face_status_label.setWordWrap(True)
        self.face_status_label.setMinimumHeight(44)
        self.face_status_label.setStyleSheet('color: rgba(69, 90, 100, 0.95); padding: 4px;')
        right_inner.addWidget(self.face_status_label)

        self.stop_face_login_button = QPushButton('停止识别')
        self.stop_face_login_button.setFont(QFont('微软雅黑', 10, QFont.Bold))
        self.stop_face_login_button.setMinimumHeight(42)
        self.stop_face_login_button.setMaximumWidth(280)
        self.stop_face_login_button.setCursor(Qt.PointingHandCursor)
        self.stop_face_login_button.setEnabled(False)
        self.stop_face_login_button.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; border: none; border-radius: 10px; } "
            "QPushButton:hover:enabled { background-color: #c62828; } "
            "QPushButton:disabled { background-color: rgba(255, 205, 210, 0.55); color: #757575; }"
        )
        self.stop_face_login_button.clicked.connect(self.stop_face_login)
        right_inner.addWidget(self.stop_face_login_button, 0, Qt.AlignHCenter)
        right_inner.addStretch(1)

        main_layout.addWidget(left_card, 0)
        main_layout.addWidget(right_card, 1)
        self.setLayout(main_layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 尝试不同的路径
        background_path = 'resources/images/bkg.jpg'
        if not os.path.exists(background_path):
            background_path = 'bkg.jpg'
        pixmap = QPixmap(background_path)
        painter.drawPixmap(self.rect(), pixmap)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._center_once:
            self._center_once = True
            _center_widget_on_screen(self)

    def login(self):
        username = self.username_line.text()
        password = self.password_line.text()

        if not username or not password:
            QMessageBox.warning(self, '警告', '用户名和密码不能为空！')
            return

        user_id = self.face_recognition.verify_user_login(username, password)
        if user_id is not None:
            QMessageBox.information(self, '成功', '登录成功！')
            self.new_window = MainWindow()
            self.new_window.show()
            self.close()
        else:
            QMessageBox.warning(self, '错误', '用户名或密码错误！')
        
    def _release_face_cap(self):
        if self._face_cap is not None:
            try:
                self._face_cap.release()
            except Exception:
                pass
            self._face_cap = None

    def _stop_face_login_timer(self):
        if self._face_login_timer is not None and self._face_login_timer.isActive():
            self._face_login_timer.stop()

    def face_login(self):
        """人脸识别登录（QTimer 驱动，避免阻塞主线程；识别算法节流调用）。"""
        if self._face_login_timer is not None and self._face_login_timer.isActive():
            return
        self._stop_face_login_timer()
        self._release_face_cap()

        self.face_login_running = True
        self.stop_face_login_button.setEnabled(True)
        self.face_status_label.setText('正在启动摄像头...')
        self._last_face_recognize_mono = 0.0

        cap = open_video_capture_by_index(0)
        if cap is None:
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.face_status_label.setText('摄像头打开失败')
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            return

        self._face_cap = cap
        self.camera_label.setText('')
        self.face_status_label.setText('请面对摄像头...')

        fd = self.face_recognition.fatigue_detector
        lb = getattr(fd, "_lm_backend", None)
        lm_ok = lb == "legacy_mesh" and getattr(fd, "_legacy_face_mesh", None) is not None
        lm_ok = lm_ok or (lb == "tasks" and getattr(fd, "_face_landmarker", None) is not None)
        if fd.yolo_face is None and not lm_ok:
            QMessageBox.warning(
                self,
                '错误',
                '未检测到可用的人脸定位（缺少 YOLO 权重且人脸关键点引擎未就绪）。',
            )
            self._release_face_cap()
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            self.face_status_label.setText('人脸识别不可用')
            return

        if self._face_login_timer is None:
            self._face_login_timer = QTimer(self)
            self._face_login_timer.timeout.connect(self._on_face_login_frame)
        self._face_login_timer.start(33)

    def _on_face_login_frame(self):
        if not self.face_login_running:
            self._stop_face_login_timer()
            self._release_face_cap()
            return

        cap = self._face_cap
        if cap is None:
            self._stop_face_login_timer()
            return

        fd = self.face_recognition.fatigue_detector
        ret, frame = cap.read()
        if not ret:
            self._stop_face_login_timer()
            self._release_face_cap()
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            QMessageBox.warning(self, '错误', '无法读取摄像头帧！')
            self.face_status_label.setText('请点击人脸识别登录按钮')
            return

        frame_bgr = cv2.resize(frame, (400, 300))
        face_img, loc_rect = fd.crop_face_for_recognition(frame_bgr)
        if loc_rect is not None:
            cv2.rectangle(
                frame_bgr,
                (loc_rect.left(), loc_rect.top()),
                (loc_rect.right(), loc_rect.bottom()),
                (0, 255, 0),
                2,
            )
            frame_bgr = draw_text_cn_on_bgr(
                frame_bgr,
                "识别中…",
                (loc_rect.left(), max(2, loc_rect.top() - 22)),
                font_size=14,
                color=(0, 255, 0),
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(image))

        if face_img is not None and face_img.size > 0:
            now = time.monotonic()
            if now - self._last_face_recognize_mono >= self._face_recognize_interval:
                self._last_face_recognize_mono = now
                name = self.face_recognition.recognize_face(face_img)
                if name:
                    self._stop_face_login_timer()
                    self._release_face_cap()
                    self.face_login_running = False
                    self.stop_face_login_button.setEnabled(False)
                    self.face_status_label.setText(f'识别成功: {name}')
                    QMessageBox.information(self, '成功', f'人脸识别成功！欢迎回来，{name}')
                    self.new_window = MainWindow()
                    self.new_window.show()
                    self.close()
                    return
                self.face_status_label.setText('人脸识别失败，请重试')
        else:
            self.face_status_label.setText('未检测到人脸，请调整位置/光线')

    def stop_face_login(self):
        """停止人脸识别登录"""
        self.face_login_running = False
        self._stop_face_login_timer()
        self._release_face_cap()
        self.stop_face_login_button.setEnabled(False)
        self.camera_label.clear()
        self.camera_label.setText('预览区')
        self.face_status_label.setText('已停止。可再次点击「人脸识别登录」')

    def closeEvent(self, event):
        self.face_login_running = False
        self._stop_face_login_timer()
        self._release_face_cap()
        event.accept()

class RegistrationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_recognition = FaceRecognition("mrsoft.db", eager_load_detector=False)
        self._center_once = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统 - 注册')
        self.resize(520, 560)
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))

        outer = QVBoxLayout()
        outer.setContentsMargins(40, 36, 40, 36)
        outer.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)

        card = QFrame()
        card.setStyleSheet(_AUTH_GLASS_CARD_QSS)
        card.setMinimumWidth(360)
        card.setMaximumWidth(400)
        inner = QVBoxLayout(card)
        inner.setContentsMargins(32, 36, 32, 36)
        inner.setSpacing(0)

        title = QtWidgets.QLabel('注册新用户')
        title.setFont(QFont('微软雅黑', 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('color: #1a365d; letter-spacing: 1px;')
        inner.addWidget(title)

        sub = QtWidgets.QLabel('创建账号后即可登录与人脸采集')
        sub.setFont(QFont('微软雅黑', 10))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet('color: rgba(69, 90, 100, 0.9); margin-top: 4px;')
        inner.addWidget(sub)
        inner.addSpacing(28)

        self.username_line = QLineEdit()
        self.username_line.setPlaceholderText('请输入用户名')
        self.username_line.setMinimumHeight(44)
        self.username_line.setStyleSheet(_AUTH_FIELD_QSS)
        inner.addWidget(_auth_labeled_field('用户名', self.username_line))
        inner.addSpacing(18)

        self.password_line = QLineEdit()
        self.password_line.setPlaceholderText('请输入密码（建议 8 位以上）')
        self.password_line.setEchoMode(QLineEdit.Password)
        self.password_line.setMinimumHeight(44)
        self.password_line.setStyleSheet(_AUTH_FIELD_QSS)
        inner.addWidget(_auth_labeled_field('密码', self.password_line))
        inner.addSpacing(28)

        self.register_button = QPushButton('  完成注册  ')
        self.register_button.setFont(QFont('微软雅黑', 11, QFont.Bold))
        self.register_button.setMinimumHeight(46)
        self.register_button.setCursor(Qt.PointingHandCursor)
        self.register_button.setStyleSheet(
            "QPushButton { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #42a5f5, stop:1 #1976d2); "
            "color: white; border: none; border-radius: 12px; } "
            "QPushButton:hover { background: #1e88e5; } QPushButton:pressed { background: #1565c0; }"
        )
        self.register_button.clicked.connect(self.register)
        inner.addWidget(self.register_button)
        inner.addStretch(1)

        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)
        self.setLayout(outer)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._center_once:
            self._center_once = True
            _center_widget_on_screen(self)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 尝试不同的路径
        background_path = 'resources/images/bkg.jpg'
        if not os.path.exists(background_path):
            background_path = 'bkg.jpg'
        pixmap = QPixmap(background_path)
        painter.drawPixmap(self.rect(), pixmap)

    def register(self):
        username = self.username_line.text()
        password = self.password_line.text()

        if not username or not password:
            QMessageBox.warning(self, '温馨提示', '您什么也没输入！')
            self.close()
            return

        user_id = self.face_recognition.register_user(username, password)
        if user_id is None:
            QMessageBox.warning(self, '错误', '用户名已存在或注册失败！')
            self.close()
            return

        QMessageBox.information(self, '成功', '注册成功！')
        self.new_window = FaceRegisterWindow(user_id=user_id, account_name=username)
        self.new_window.show()
        self.close()

    def Open(self):
        self.show()

class FaceRegisterWindow(QWidget):
    def __init__(self, user_id, account_name=""):
        super().__init__()
        self.user_id = user_id
        self.account_name = account_name
        self.face_recognition = FaceRecognition("mrsoft.db")
        self._center_once = False
        self._reg_cap = None
        self._reg_timer: QTimer | None = None
        self._reg_active = False
        self._reg_progress = 0
        self._reg_target_name = ""
        self._last_reg_db_mono = 0.0
        self._reg_db_interval = max(0.35, float(os.environ.get("TIRED_FACE_REGISTER_DB_SEC", "0.5")))
        self._reg_timeout_sec = max(30.0, float(os.environ.get("TIRED_FACE_REGISTER_TIMEOUT_SEC", "180")))
        self._reg_started_mono = 0.0
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统 - 人脸注册')
        # 尝试不同的路径
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))
        self.resize(880, 640)

        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 左侧注册区域
        left_widget = QtWidgets.QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(20)
        
        # 标题
        title_label = QtWidgets.QLabel('人脸注册')
        title_font = QFont()
        title_font.setFamily('微软雅黑')
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('color: #333333;')
        left_layout.addWidget(title_label)
        
        # 姓名输入
        name_layout = QHBoxLayout()
        name_label = QtWidgets.QLabel('姓名:')
        name_label.setFont(QFont('微软雅黑', 11))
        self.name_line = QLineEdit()
        self.name_line.setPlaceholderText('请输入您的姓名')
        self.name_line.setFixedSize(250, 40)
        self.name_line.setStyleSheet('QLineEdit { padding: 8px; border: 1px solid #CCCCCC; border-radius: 4px; font-size: 14px; }')
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_line)
        left_layout.addLayout(name_layout)
        
        # 注册按钮
        self.register_button = QPushButton('开始注册')
        self.register_button.clicked.connect(self.register_face)
        self.register_button.setFixedSize(250, 45)
        self.register_button.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #45a049; }')
        left_layout.addWidget(self.register_button)
        
        # 状态显示
        self.status_label = QtWidgets.QLabel('请输入姓名并点击开始注册')
        self.status_label.setFont(QFont('微软雅黑', 11))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(40)
        self.status_label.setStyleSheet('color: #666666;')
        left_layout.addWidget(self.status_label)
        
        # 已注册用户列表
        list_label = QtWidgets.QLabel('已注册用户:')
        list_label.setFont(QFont('微软雅黑', 11, QFont.Bold))
        left_layout.addWidget(list_label)
        
        self.result_list = QListWidget()
        self.result_list.setFixedSize(250, 200)
        self.result_list.setStyleSheet('QListWidget { border: 1px solid #CCCCCC; border-radius: 4px; font-size: 14px; }')
        left_layout.addWidget(self.result_list)
        
        # 右侧摄像头区域
        right_widget = QtWidgets.QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignCenter)
        
        # 摄像头显示
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(400, 300)
        self.camera_label.setStyleSheet('border: 1px solid #CCCCCC; border-radius: 4px; background-color: #000000;')
        right_layout.addWidget(self.camera_label)
        
        # 提示信息
        self.hint_label = QtWidgets.QLabel('提示: 请确保光线充足，正面面对摄像头')
        self.hint_label.setFont(QFont('微软雅黑', 10))
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setFixedHeight(30)
        self.hint_label.setStyleSheet('color: #666666;')
        right_layout.addWidget(self.hint_label)
        
        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedSize(400, 20)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # 将左右区域添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        self.setLayout(main_layout)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._center_once:
            self._center_once = True
            _center_widget_on_screen(self)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 尝试不同的路径
        background_path = 'resources/images/yjwj.jpg'
        if not os.path.exists(background_path):
            background_path = 'yjwj.jpg'
        pixmap = QPixmap(background_path)
        painter.drawPixmap(self.rect(), pixmap)

    def _release_reg_cap(self):
        if self._reg_cap is not None:
            try:
                self._reg_cap.release()
            except Exception:
                pass
            self._reg_cap = None

    def _stop_reg_timer(self):
        if self._reg_timer is not None and self._reg_timer.isActive():
            self._reg_timer.stop()

    def register_face(self):
        """人脸注册：QTimer 驱动，数据库写入节流，避免阻塞主界面。"""
        if self._reg_active:
            return
        name = self.name_line.text().strip()
        if not name:
            QMessageBox.warning(self, '温馨提示', '姓名不能为空！')
            return

        self._stop_reg_timer()
        self._release_reg_cap()

        self.status_label.setText(f'正在注册: {name}')
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.register_button.setEnabled(False)

        cap = open_video_capture_by_index(0)
        if cap is None:
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.status_label.setText('摄像头打开失败')
            self.progress_bar.setVisible(False)
            self.register_button.setEnabled(True)
            return

        self._reg_cap = cap

        fd = self.face_recognition.fatigue_detector
        lb = getattr(fd, "_lm_backend", None)
        lm_ok = lb == "legacy_mesh" and getattr(fd, "_legacy_face_mesh", None) is not None
        lm_ok = lm_ok or (lb == "tasks" and getattr(fd, "_face_landmarker", None) is not None)
        if fd.yolo_face is None and not lm_ok:
            QMessageBox.warning(
                self,
                '错误',
                '未检测到可用的人脸定位（缺少 YOLO 权重且人脸关键点引擎未就绪）。',
            )
            self._release_reg_cap()
            self.progress_bar.setVisible(False)
            self.status_label.setText('人脸注册不可用')
            self.register_button.setEnabled(True)
            return

        self.hint_label.setText('请面对摄像头，保持表情自然...')
        self._reg_target_name = name
        self._reg_progress = 0
        self._reg_active = True
        self._last_reg_db_mono = -1e9
        self._reg_started_mono = time.monotonic()

        if self._reg_timer is None:
            self._reg_timer = QTimer(self)
            self._reg_timer.timeout.connect(self._on_face_register_frame)
        self._reg_timer.start(33)

    def _on_face_register_frame(self):
        if not self._reg_active:
            self._stop_reg_timer()
            self._release_reg_cap()
            return

        cap = self._reg_cap
        if cap is None:
            self._finish_face_register_flow(False, '内部错误：摄像头未打开。')
            return

        if time.monotonic() - self._reg_started_mono > self._reg_timeout_sec:
            self._finish_face_register_flow(False, '注册超时：请调整光线与位置后重试。')
            return

        name = self._reg_target_name
        fd = self.face_recognition.fatigue_detector
        ret, frame = cap.read()
        if not ret:
            self._finish_face_register_flow(False, '无法读取摄像头帧！')
            return

        frame_bgr = cv2.resize(frame, (400, 300))
        face_img, loc_rect = fd.crop_face_for_recognition(frame_bgr)
        if loc_rect is not None:
            cv2.rectangle(
                frame_bgr,
                (loc_rect.left(), loc_rect.top()),
                (loc_rect.right(), loc_rect.bottom()),
                (0, 255, 0),
                2,
            )
            frame_bgr = draw_text_cn_on_bgr(
                frame_bgr,
                name,
                (loc_rect.left(), max(2, loc_rect.top() - 52)),
                font_size=16,
                color=(0, 255, 0),
            )
            frame_bgr = draw_text_cn_on_bgr(
                frame_bgr,
                "正在注册…",
                (loc_rect.left(), max(2, loc_rect.top() - 24)),
                font_size=14,
                color=(0, 255, 0),
            )

        if face_img is not None and face_img.size > 0:
            now = time.monotonic()
            if now - self._last_reg_db_mono >= self._reg_db_interval:
                self._last_reg_db_mono = now
                success = self.face_recognition.register_face(self.user_id, name, face_img)
                if success:
                    self._reg_progress = 100
                    self._finish_face_register_flow(True)
                    return
                self._reg_progress = min(90, self._reg_progress + 10)
        else:
            frame_bgr = draw_text_cn_on_bgr(
                frame_bgr,
                "请将人脸置于画面中央",
                (10, 270),
                font_size=14,
                color=(0, 0, 255),
            )

        self.progress_bar.setValue(self._reg_progress)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(image))

    def _finish_face_register_flow(self, success: bool, err_msg: str | None = None):
        self._reg_active = False
        self._stop_reg_timer()
        self._release_reg_cap()
        self.progress_bar.setVisible(False)
        self.register_button.setEnabled(True)
        self.hint_label.setText('提示: 请确保光线充足，正面面对摄像头')

        if success:
            QMessageBox.information(self, '成功', '人脸注册成功！')
            self.result_list.addItem(self._reg_target_name)
            self.name_line.clear()
            self.status_label.setText('注册成功，请继续注册其他用户或关闭窗口')
            return
        if err_msg:
            QMessageBox.warning(self, '提示', err_msg)
            self.status_label.setText(err_msg)
            return
        QMessageBox.warning(self, '失败', '人脸注册失败，请重试！')
        self.status_label.setText('注册失败，请重试')

    def closeEvent(self, event):
        self._reg_active = False
        self._stop_reg_timer()
        self._release_reg_cap()
        self.face_recognition.close_db()
        event.accept()
