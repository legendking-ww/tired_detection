import sys
import os
import cv2
import sqlite3
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QFont
from PyQt5.QtWidgets import QMainWindow, QGraphicsPixmapItem, QGraphicsScene, QMessageBox, QFileDialog, QLabel, QListWidget, QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from pygame import mixer
import winsound

from src.ui.UI import Ui_MainWindow
from src.core.fatigue_detection import FatigueDetector
from src.core.face_recognition import FaceRecognition

class BaseThread(QThread):
    picture = pyqtSignal(object)
    msg = pyqtSignal(str)
    window = pyqtSignal(str)

    def __init__(self):
        super(BaseThread, self).__init__()
        self.fatigue_detector = FatigueDetector()
        self.face_recognition = FaceRecognition()
        self.cap = None
        self.camSelect = 0
        self.isClose = False

    def change_cam_select(self, camSelect):
        self.camSelect = camSelect

    def close(self):
        self.isClose = True
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                pass

    def process_frame(self, frame):
        return self.fatigue_detector.process_frame(frame)

class AdjustCamera_Thread(BaseThread):
    def run(self):
        self.isClose = False
        self.window.emit("请调整摄像头位置，使人脸位于显示框内。调整后请按关闭结束")
        
        try:
            self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.window.emit("摄像头打开失败，请检查摄像头是否连接")
                return
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame, gray, rects = self.process_frame(frame)
                
                for rect in rects:
                    shape = self.fatigue_detector.predictor(gray, rect)
                    shape = self.fatigue_detector.face_utils.shape_to_np(shape)
                    reprojectdst, euler_angle = self.fatigue_detector.get_head_pose(shape)
                    pitch = euler_angle[0, 0]
                    yaw = euler_angle[1, 0]
                    roll = euler_angle[2, 0]
                    
                    for start, end in self.fatigue_detector.line_pairs:
                        start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                        end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
                    
                    cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                self.picture.emit(frame)
                
                if self.isClose:
                    break
        except Exception as e:
            error_msg = f"摄像头调整失败: {str(e)}"
            print(error_msg)
            self.msg.emit(error_msg)
        finally:
            if self.cap is not None:
                self.cap.release()
            self.window.emit("摄像头位置调整结束")

class Start_Thread(BaseThread):
    def __init__(self):
        super(Start_Thread, self).__init__()
        self.offDutyTime = 0
        self.filePath = None
        self.isOffDutyCheck = False
        self.isOpenVideo = False
        self.isShowEye = True
        self.isShowMouth = True
        self.isShowHead = False
        self.isShowKeyPoint = False

    def set_show_eye(self, isShowEye):
        self.isShowEye = isShowEye

    def set_show_mouth(self, isShowMouth):
        self.isShowMouth = isShowMouth

    def set_show_Head(self, isShowHead):
        self.isShowHead = isShowHead

    def set_show_key_point(self, isShowKeyPoint):
        self.isShowKeyPoint = isShowKeyPoint

    def change_OffDuty_Check_Status(self, isOffDutyCheck):
        self.isOffDutyCheck = isOffDutyCheck

    def change_OffDuty_Value(self, offDutyTime):
        self.offDutyTime = offDutyTime

    def set_filePath(self, filePath):
        self.isOpenVideo = True
        self.filePath = filePath

    def set_open_video(self, isOpenVideo):
        self.isOpenVideo = isOpenVideo

    @staticmethod
    def playMusic():
        mixer.init()
        mixer.music.load('resources/sounds/warning.mp3')
        mixer.music.play()
    
    @staticmethod
    def put_text_cn(img, text, position, font_size=20, color=(0, 255, 255)):
        """在图像上绘制中文字符"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试不同的字体路径
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 备选字体
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
        
        # 如果没有找到字体，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 将PIL图像转换回OpenCV图像
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def run(self):
        self.window.emit("开始程序")
        self.isClose = False

        try:
            if self.isOpenVideo:
                if self.filePath is None:
                    self.window.emit("未加载视频，请加载视频后再点击")
                    return
                
                self.cap = cv2.VideoCapture(self.filePath)
                if not self.cap.isOpened():
                    self.window.emit("视频文件打开失败，请检查文件路径")
                    return
                self.msg.emit("视频读取成功")
            else:
                self.cap = cv2.VideoCapture(self.camSelect, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.window.emit("摄像头打开失败，请检查摄像头是否连接")
                    return
                self.msg.emit("相机打开成功")

            test_time = 0
            TEST_TIMES = 100
            ear_sum = 0
            mar_sum = 0
            har_sum = 0

            Detected_TIME_LIMIT = 60
            closed_times = 0
            yawning_times = 0
            pitch_times = 0
            warning_time = 0

            EAR_THRESH = 0.32
            MAR_THRESH = 0.55
            HAR_THRESH = 0
            FATIGUE_THRESH = 0.4
            PITCH_THRESH = 5
            offDutyTime = 0

            self.msg.emit("程序正在加载中，请您耐心等待")
            self.window.emit("程序正在加载中，请您耐心等待")

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self.isOpenVideo:
                        self.window.emit("视频播放结束")
                    print('视频结束')
                    break

                frame, gray, rects = self.process_frame(frame)

                if not rects:
                    if self.isOffDutyCheck:
                        offDutyTime += 1
                        if offDutyTime >= self.offDutyTime * 30:
                            self.msg.emit("您已经脱岗，请立刻回到岗位")
                            self.window.emit("您已经脱岗，请立刻回到岗位")
                            offDutyTime = 0
                else:
                    offDutyTime = 0

                for rect in rects:
                    shape = self.fatigue_detector.predictor(gray, rect)

                    if self.isShowKeyPoint:
                        for point in shape.parts():
                            point_position = (point.x, point.y)
                            cv2.circle(frame, point_position, 3, (255, 8, 0), -1)

                    shape = self.fatigue_detector.face_utils.shape_to_np(shape)
                    (lStart, lEnd) = self.fatigue_detector.face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                    (rStart, rEnd) = self.fatigue_detector.face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                    (mStart, mEnd) = self.fatigue_detector.face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]
                    
                    leftEAR = self.fatigue_detector.eye_aspect_ratio(leftEye)
                    rightEAR = self.fatigue_detector.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    mar = self.fatigue_detector.mouth_aspect_ratio(mouth)

                    reprojectdst, euler_angle = self.fatigue_detector.get_head_pose(shape)
                    pitch = euler_angle[0, 0]
                    yaw = euler_angle[1, 0]
                    roll = euler_angle[2, 0]
                    har = pitch

                    if self.isShowHead:
                        for start, end in self.fatigue_detector.line_pairs:
                            start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                            end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                    # 人脸识别
                    left = rect.left()
                    top = rect.top()
                    right = rect.right()
                    bottom = rect.bottom()
                    face_img = frame[top:bottom, left:right]
                    if face_img.size > 0:
                        name = self.face_recognition.recognize_face(face_img)
                        if name:
                            # 使用自定义函数绘制中文字符
                            frame = self.put_text_cn(frame, f"姓名: {name}", (left, top - 25), font_size=16, color=(0, 255, 255))

                    cv2.putText(frame, "ear: {:.2f}".format(ear), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "mar: {:.2f}".format(mar), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if test_time < TEST_TIMES:
                        test_time += 1
                        ear_sum += ear
                        mar_sum += mar
                        har_sum += har

                        if test_time == TEST_TIMES:
                            EAR_THRESH = ear_sum / TEST_TIMES
                            MAR_THRESH = mar_sum / TEST_TIMES
                            HAR_THRESH = har_sum / TEST_TIMES
                            print('眼睛长宽比ear 100次取平均的阈值:{:.2f} '.format(EAR_THRESH))
                            print('嘴部长宽比mar 100次取平均的阈值:{:.2f} '.format(MAR_THRESH))
                            print('头部俯仰角pitch 100次取平均的阈值:{:.2f} '.format(HAR_THRESH))
                            self.msg.emit('眼睛长宽比ear 100次取平均的阈值:{:.2f}'.format(EAR_THRESH))
                            self.msg.emit('嘴部长宽比mar 100次取平均的阈值:{:.2f}'.format(MAR_THRESH))
                            self.msg.emit('头部俯仰角pitch 100次取平均的阈值:{:.2f}'.format(HAR_THRESH))
                        continue

                    if self.isShowEye:
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    if self.isShowMouth:
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

                    if Detected_TIME_LIMIT > 0:
                        Detected_TIME_LIMIT -= 1
                        if ear < 0.75 * EAR_THRESH:
                            closed_times += 1
                        if mar > 1.6 * MAR_THRESH:
                            yawning_times += 1
                        if abs(har - HAR_THRESH) > PITCH_THRESH:
                            pitch_times += 1

                    else:
                        Detected_TIME_LIMIT = 60
                        isEyeTired = False
                        isYawnTired = False
                        isHeadTired = False

                        if closed_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                            self.msg.emit("闭眼时长较长")
                            isEyeTired = True

                        if yawning_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                            self.msg.emit("张嘴时长较长")
                            isYawnTired = True

                        if pitch_times / Detected_TIME_LIMIT > FATIGUE_THRESH:
                            self.msg.emit("低头时长较长")
                            isHeadTired = True

                        closed_times = 0
                        yawning_times = 0
                        pitch_times = 0

                        isWarning = False
                        if isEyeTired and isYawnTired:
                            warning_time += 2
                            isWarning = True
                        elif isHeadTired and isEyeTired:
                            warning_time += 2
                            isWarning = True
                        elif isEyeTired:
                            warning_time += 1
                            isWarning = True
                        elif isYawnTired:
                            warning_time += 1
                            isWarning = True
                        elif isHeadTired:
                            warning_time += 1
                            isWarning = True
                        else:
                            warning_time = 0

                        if warning_time >= 3:
                            warning_time = 0
                            self.msg.emit("您已经疲劳，请注意休息!")
                            self.window.emit("您已经疲劳，请注意休息!")
                            self.playMusic()
                        else:
                            if isWarning:
                                winsound.Beep(440, 1000)

                self.picture.emit(frame)

                if self.isClose:
                    break
        except Exception as e:
            error_msg = f"程序运行失败: {str(e)}"
            print(error_msg)
            self.msg.emit(error_msg)
        finally:
            if self.cap is not None:
                self.cap.release()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.adjust_camera_Thread = AdjustCamera_Thread()
        self.start_Thread = Start_Thread()
        self.setupUi(self)
        self.init_camera_list()

        self.Cam_Select.currentIndexChanged.connect(self.change_Cam_Select)
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

    def init_camera_list(self):
        self.Cam_Select.clear()
        self.Cam_Select.addItem("选择摄像头")
        
        max_cameras = 5
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.Cam_Select.addItem(f"摄像头 {i}")
                cap.release()

        if self.Cam_Select.count() > 1:
            self.Cam_Select.setCurrentIndex(1)

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
        self.start_Thread.start()

    def adjust_camera_location(self):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height = image.shape[0]
        width = image.shape[1]
        frame = QImage(image, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def pop_window(self, info):
        QMessageBox.warning(self, "提示", info, QMessageBox.Yes)

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

        conn = sqlite3.connect('mrsoft.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()

        if user:
            QMessageBox.information(self, '成功', '登录成功！')
            self.new_window = MainWindow()
            self.new_window.show()
            self.close()  # 关闭登录窗口
        else:
            QMessageBox.warning(self, '错误', '用户名或密码错误！')

        cursor.close()
        conn.close()
        
    def face_login(self):
        """人脸识别登录"""
        self.face_login_running = True
        self.stop_face_login_button.setEnabled(True)
        self.face_status_label.setText('正在启动摄像头...')
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.face_status_label.setText('摄像头打开失败')
            self.face_login_running = False
            self.stop_face_login_button.setEnabled(False)
            return
        
        self.face_status_label.setText('请面对摄像头...')
        
        while self.face_login_running:
            ret, frame = cap.read()
            if not ret:
                QMessageBox.warning(self, '错误', '无法读取摄像头帧！')
                break
            
            # 显示摄像头画面
            frame = cv2.resize(frame, (400, 300))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.camera_label.setPixmap(pixmap)
            
            # 检测人脸
            det_bboxes, det_conf, det_classid, landmarks = self.face_recognition.fatigue_detector.yolo_face.detect(frame)
            if len(det_bboxes) > 0 and len(det_conf) > 0:
                for i, box in enumerate(det_bboxes):
                    if det_conf[i] > 0.5:
                        x, y, w, h = box.astype('int')
                        if y >= 0 and y + h <= frame.shape[0] and x >= 0 and x + w <= frame.shape[1]:
                            face_img = frame[y:y + h, x:x + w]
                            # 识别人脸
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
                                self.close()  # 关闭登录窗口
                                return
                            else:
                                self.face_status_label.setText('人脸识别失败，请重试')
            
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

        conn = sqlite3.connect('mrsoft.db')
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            QMessageBox.information(self, '成功', '注册成功！')

            self.new_window = FaceRegisterWindow()
            self.new_window.show()

            self.close()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, '错误', '用户名已存在！')
            self.close()
        finally:
            cursor.close()
            conn.close()

    def Open(self):
        self.show()

class FaceRegisterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_recognition = FaceRecognition('mrsoft.db')
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
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.warning(self, '错误', '无法打开摄像头！')
            self.status_label.setText('摄像头打开失败')
            self.progress_bar.setVisible(False)
            return

        self.hint_label.setText('请面对摄像头，保持表情自然...')
        
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
            
            # 检测人脸
            det_bboxes, det_conf, det_classid, landmarks = self.face_recognition.fatigue_detector.yolo_face.detect(frame)
            if len(det_bboxes) > 0 and len(det_conf) > 0:
                for i, box in enumerate(det_bboxes):
                    if det_conf[i] > 0.5:
                        x, y, w, h = box.astype('int')
                        if y >= 0 and y + h <= frame.shape[0] and x >= 0 and x + w <= frame.shape[1]:
                            # 绘制人脸框
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, '正在注册...', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # 提取人脸并注册
                            face_img = frame[y:y + h, x:x + w]
                            success = self.face_recognition.register_face(1, name, face_img)
                            if success:
                                registered = True
                                progress = 100
                            else:
                                progress += 10
            else:
                # 没有检测到人脸
                cv2.putText(frame, '请将人脸置于画面中央', (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    login = LoginWindow()
    register = RegistrationWindow()
    facereg = FaceRegisterWindow()
    login.register_button.clicked.connect(register.Open)
    login.show()
    sys.exit(app.exec_())