"""登录、注册与人脸采集窗口。"""
import os

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QFont
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QListWidget,
)

from src.core.face_recognition import FaceRecognition
from src.utils.cv_helpers import draw_text_cn_on_bgr, open_video_capture_by_index
from src.app.main_window import MainWindow


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_recognition = FaceRecognition('mrsoft.db')
        self.face_login_running = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统')
        # 尝试不同的路径
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(500, 200, 900, 600)

        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 左侧登录区域
        left_widget = QtWidgets.QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(20)
        
        # 标题
        title_label = QtWidgets.QLabel('用户登录')
        title_font = QFont()
        title_font.setFamily('微软雅黑')
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('color: #333333;')
        left_layout.addWidget(title_label)
        
        # 用户名输入
        username_layout = QHBoxLayout()
        username_label = QtWidgets.QLabel('用户名:')
        username_label.setFont(QFont('微软雅黑', 11))
        self.username_line = QLineEdit()
        self.username_line.setPlaceholderText('请输入用户名')
        self.username_line.setFixedSize(250, 40)
        self.username_line.setStyleSheet('QLineEdit { padding: 8px; border: 1px solid #CCCCCC; border-radius: 4px; font-size: 14px; }')
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_line)
        left_layout.addLayout(username_layout)
        
        # 密码输入
        password_layout = QHBoxLayout()
        password_label = QtWidgets.QLabel('密码:')
        password_label.setFont(QFont('微软雅黑', 11))
        self.password_line = QLineEdit()
        self.password_line.setPlaceholderText('请输入密码')
        self.password_line.setEchoMode(QLineEdit.Password)
        self.password_line.setFixedSize(250, 40)
        self.password_line.setStyleSheet('QLineEdit { padding: 8px; border: 1px solid #CCCCCC; border-radius: 4px; font-size: 14px; }')
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_line)
        left_layout.addLayout(password_layout)
        
        # 登录按钮
        self.login_button = QPushButton('账号登录')
        self.login_button.clicked.connect(self.login)
        self.login_button.setFixedSize(250, 45)
        self.login_button.setStyleSheet('QPushButton { background-color: #2196F3; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #1976D2; }')
        left_layout.addWidget(self.login_button)
        
        # 人脸识别登录按钮
        self.face_login_button = QPushButton('人脸识别登录')
        self.face_login_button.clicked.connect(self.face_login)
        self.face_login_button.setFixedSize(250, 45)
        self.face_login_button.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; } QPushButton:hover { background-color: #45a049; }')
        left_layout.addWidget(self.face_login_button)
        
        # 注册按钮
        self.register_button = QPushButton('注册新用户')
        self.register_button.setFixedSize(250, 40)
        self.register_button.setStyleSheet('QPushButton { background-color: #FF9800; color: white; border: none; border-radius: 4px; font-size: 14px; } QPushButton:hover { background-color: #F57C00; }')
        left_layout.addWidget(self.register_button)
        
        # 右侧人脸识别区域
        right_widget = QtWidgets.QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setAlignment(Qt.AlignCenter)
        
        # 摄像头显示
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(400, 300)
        self.camera_label.setStyleSheet('border: 1px solid #CCCCCC; border-radius: 4px; background-color: #000000;')
        right_layout.addWidget(self.camera_label)
        
        # 人脸识别状态
        self.face_status_label = QtWidgets.QLabel('请点击人脸识别登录按钮')
        self.face_status_label.setFont(QFont('微软雅黑', 12))
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.face_status_label.setFixedHeight(40)
        right_layout.addWidget(self.face_status_label)
        
        # 停止按钮
        self.stop_face_login_button = QPushButton('停止')
        self.stop_face_login_button.setFixedSize(250, 40)
        self.stop_face_login_button.setStyleSheet('QPushButton { background-color: #f44336; color: white; border: none; border-radius: 4px; font-size: 14px; } QPushButton:hover { background-color: #d32f2f; }')
        self.stop_face_login_button.setEnabled(False)
        right_layout.addWidget(self.stop_face_login_button)
        self.stop_face_login_button.clicked.connect(self.stop_face_login)
        
        # 将左右区域添加到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        
        self.setLayout(main_layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        # 尝试不同的路径
        background_path = 'resources/images/bkg.jpg'
        if not os.path.exists(background_path):
            background_path = 'bkg.jpg'
        pixmap = QPixmap(background_path)
        painter.drawPixmap(self.rect(), pixmap)

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
        
    def face_login(self):
        """人脸识别登录"""
        self.face_login_running = True
        self.stop_face_login_button.setEnabled(True)
        self.face_status_label.setText('正在启动摄像头...')
        
        cap = open_video_capture_by_index(0)
        if cap is None:
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.face_status_label.setText('摄像头打开失败')
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            return
        
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
            cap.release()
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            self.face_status_label.setText('人脸识别不可用')
            return
        
        while self.face_login_running:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, '错误', '无法读取摄像头帧！')
                break
            
            # OpenCV 读取为 BGR；YOLO_face.detect 也按 BGR 输入处理（内部自行转 RGB 做推理）
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

            # 显示摄像头画面（仅展示层转 RGB）
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)

            if face_img is not None and face_img.size > 0:
                name = self.face_recognition.recognize_face(face_img)
                if name:
                    self.face_status_label.setText(f'识别成功: {name}')
                    QMessageBox.information(self, '成功', f'人脸识别成功！欢迎回来，{name}')
                    cap.release()
                    cv2.destroyAllWindows()
                    self.face_login_running = False
                    self.stop_face_login_button.setEnabled(False)
                    self.new_window = MainWindow()
                    self.new_window.show()
                    self.close()
                    return
                self.face_status_label.setText('人脸识别失败，请重试')
            else:
                self.face_status_label.setText('未检测到人脸，请调整位置/光线')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.face_login_running = False
        self.stop_face_login_button.setEnabled(False)
        self.face_status_label.setText('请点击人脸识别登录按钮')
    
    def stop_face_login(self):
        """停止人脸识别登录"""
        self.face_login_running = False
        self.stop_face_login_button.setEnabled(False)
        self.face_status_label.setText('人脸识别已停止')

class RegistrationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_recognition = FaceRecognition("mrsoft.db", eager_load_detector=False)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统')
        self.setGeometry(500, 300, 600, 400)
        # 尝试不同的路径
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))
        layout = QVBoxLayout()

        self.username_line = QLineEdit(self)
        self.username_line.setPlaceholderText('输入账户')
        layout.addWidget(self.username_line)
        self.username_line.setFixedSize(300, 40)
        self.username_line.setStyleSheet("QLineEdit { background-color: rgba(255, 255, 255, 220); }")

        self.password_line = QLineEdit(self)
        self.password_line.setPlaceholderText('输入密码')
        self.password_line.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_line)
        self.password_line.setFixedSize(300, 40)
        self.password_line.setStyleSheet("QLineEdit { background-color: rgba(255, 255, 255, 220); }")

        self.register_button = QPushButton('注册', self)
        self.register_button.clicked.connect(self.register)
        layout.addWidget(self.register_button)
        self.register_button.setFixedSize(280, 40)

        self.setLayout(layout)

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
        self.initUI()

    def initUI(self):
        self.setWindowTitle('疲劳驾驶检测系统 - 人脸注册')
        # 尝试不同的路径
        icon_path = 'resources/images/yjwj.png'
        if not os.path.exists(icon_path):
            icon_path = 'yjwj.png'
        self.setWindowIcon(QIcon(icon_path))
        self.setGeometry(500, 200, 800, 600)

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

    def paintEvent(self, event):
        painter = QPainter(self)
        # 尝试不同的路径
        background_path = 'resources/images/yjwj.jpg'
        if not os.path.exists(background_path):
            background_path = 'yjwj.jpg'
        pixmap = QPixmap(background_path)
        painter.drawPixmap(self.rect(), pixmap)

    def register_face(self):
        name = self.name_line.text().strip()

        if not name:
            QMessageBox.warning(self, '温馨提示', '姓名不能为空！')
            return

        self.status_label.setText(f'正在注册: {name}')
        self.progress_bar.setVisible(True)
        
        cap = open_video_capture_by_index(0)
        if cap is None:
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.status_label.setText('摄像头打开失败')
            self.progress_bar.setVisible(False)
            return

        self.hint_label.setText('请面对摄像头，保持表情自然...')

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
            cap.release()
            self.progress_bar.setVisible(False)
            self.status_label.setText('人脸注册不可用')
            return
        
        # 注册成功标志
        registered = False
        # 进度计数器
        progress = 0
        
        while progress < 100:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, '错误', '无法读取摄像头帧！')
                break

            # 显示摄像头画面
            frame = cv2.resize(frame, (400, 300))

            face_img, loc_rect = fd.crop_face_for_recognition(frame)
            if loc_rect is not None:
                cv2.rectangle(
                    frame,
                    (loc_rect.left(), loc_rect.top()),
                    (loc_rect.right(), loc_rect.bottom()),
                    (0, 255, 0),
                    2,
                )
                frame = draw_text_cn_on_bgr(
                    frame,
                    name,
                    (loc_rect.left(), max(2, loc_rect.top() - 52)),
                    font_size=16,
                    color=(0, 255, 0),
                )
                frame = draw_text_cn_on_bgr(
                    frame,
                    "正在注册…",
                    (loc_rect.left(), max(2, loc_rect.top() - 24)),
                    font_size=14,
                    color=(0, 255, 0),
                )

            if face_img is not None and face_img.size > 0:
                success = self.face_recognition.register_face(self.user_id, name, face_img)
                if success:
                    registered = True
                    progress = 100
                else:
                    progress += 10
            else:
                frame = draw_text_cn_on_bgr(
                    frame,
                    "请将人脸置于画面中央",
                    (10, 270),
                    font_size=14,
                    color=(0, 0, 255),
                )
            
            # 更新进度条
            self.progress_bar.setValue(progress)
            
            # 显示画面
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)
            
            # 刷新界面
            QtWidgets.QApplication.processEvents()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if registered:
            QMessageBox.information(self, '成功', '人脸注册成功！')
            self.result_list.addItem(name)
            self.name_line.clear()
            self.status_label.setText('注册成功，请继续注册其他用户或关闭窗口')
        else:
            QMessageBox.warning(self, '失败', '人脸注册失败，请重试！')
            self.status_label.setText('注册失败，请重试')
        
        self.progress_bar.setVisible(False)
        self.hint_label.setText('提示: 请确保光线充足，正面面对摄像头')

    def closeEvent(self, event):
        self.face_recognition.close_db()
        event.accept()
