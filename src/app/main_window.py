"""主检测窗口（PyQt + Ui_MainWindow）。"""
import os

import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QMessageBox,
    QFileDialog,
)

from src.ui.UI import Ui_MainWindow
from src.app.worker_threads import AdjustCamera_Thread, Start_Thread
from src.utils.cv_helpers import open_video_capture_by_index


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
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

    def init_camera_list(self):
        self.Cam_Select.blockSignals(True)
        self.Cam_Select.clear()
        self.Cam_Select.addItem("选择摄像头")

        max_cameras = 5
        for i in range(max_cameras):
            cap = open_video_capture_by_index(i)
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
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        frame = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)
        self.graphicsView.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def pop_window(self, info):
        QMessageBox.warning(self, "提示", info, QMessageBox.Yes)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        sc = self.graphicsView.scene()
        if sc is not None and sc.items():
            self.graphicsView.fitInView(sc.itemsBoundingRect(), Qt.KeepAspectRatio)
