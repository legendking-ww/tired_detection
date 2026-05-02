"""疲劳检测与摄像头调整线程。"""
import time
import cv2
import winsound
from pygame import mixer
from PyQt5.QtCore import QThread, pyqtSignal

from src.core.fatigue_detection import FatigueDetector
from src.core.face_recognition import FaceRecognition
from src.utils.cv_helpers import draw_text_cn_on_bgr, open_video_capture_by_index


class BaseThread(QThread):
    picture = pyqtSignal(object)
    msg = pyqtSignal(str)
    window = pyqtSignal(str)

    def __init__(self):
        super(BaseThread, self).__init__()
        self.fatigue_detector = FatigueDetector()
        # 复用同一套 FatigueDetector，避免重复加载 MediaPipe / PyTorch
        self.face_recognition = FaceRecognition(db_path="mrsoft.db", fatigue_detector=self.fatigue_detector)
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
            self.cap = open_video_capture_by_index(self.camSelect)
            if self.cap is None:
                self.window.emit("摄像头打开失败，请检查摄像头是否连接")
                return
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame, gray, rects = self.process_frame(frame)

                analysis = self.fatigue_detector.analyze_face(frame)
                if analysis is not None:
                    reprojectdst = analysis["reprojectdst"]
                    pitch = analysis["pitch"]
                    yaw = analysis["yaw"]
                    roll = analysis["roll"]
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
        # 与 UI 默认一致；正式开跑前 start() 会再次从界面同步
        self.offDutyTime = 5
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
        return draw_text_cn_on_bgr(img, text, position, font_size=font_size, color=color)

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
                self.cap = open_video_capture_by_index(self.camSelect)
                if self.cap is None:
                    self.window.emit("摄像头打开失败，请检查摄像头是否连接")
                    return
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
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
            off_duty_absent_since = None
            off_duty_last_alert = 0.0
            calib_miss_streak = 0
            CALIB_MISS_ABORT = 300

            self.msg.emit("程序正在加载中，请您耐心等待")
            self.window.emit("程序正在加载中，请您耐心等待")

            lb = getattr(self.fatigue_detector, "_lm_backend", None)
            if lb is None or (
                lb == "legacy_mesh" and getattr(self.fatigue_detector, "_legacy_face_mesh", None) is None
            ) or (lb == "tasks" and getattr(self.fatigue_detector, "_face_landmarker", None) is None):
                err = getattr(self.fatigue_detector, "last_model_error", None) or "未知错误"
                self.window.emit(f"人脸关键点引擎未加载（face_mesh / FaceLandmarker），疲劳检测不可用：{err}")
                self.msg.emit(f"关键点引擎未加载：{err}")
                return

            if not self.isOpenVideo:
                for _ in range(45):
                    ret, _ = self.cap.read()
                    if not ret:
                        self.msg.emit("摄像头预热中断：读取帧失败，请检查设备占用或权限")
                        break
                else:
                    self.msg.emit("摄像头预热完成，请正对镜头、面部光线均匀（勿强逆光）")

            while True:
                ret, raw = self.cap.read()
                if not ret:
                    if self.isOpenVideo:
                        self.window.emit("视频播放结束")
                    print('视频结束')
                    break

                frame, gray, _ = self.process_frame(raw)
                analysis = self.fatigue_detector.analyze_face(frame)
                if analysis is None:
                    raw_n = self.fatigue_detector._normalize_bgr(raw)
                    ar = self.fatigue_detector.analyze_face(raw_n)
                    if ar is not None:
                        h0, w0 = raw_n.shape[:2]
                        h1, w1 = frame.shape[:2]
                        analysis = FatigueDetector.rescale_face_analysis(ar, w0, h0, w1, h1)
                analyses = []
                if analysis is not None:
                    analyses.append((analysis["bbox_rect"], analysis))
                face_present = analysis is not None

                if self.isOffDutyCheck:
                    need_sec = max(1, int(self.offDutyTime))
                    now = time.monotonic()
                    if face_present:
                        off_duty_absent_since = None
                    else:
                        if off_duty_absent_since is None:
                            off_duty_absent_since = now
                        elif now - off_duty_absent_since >= need_sec and (now - off_duty_last_alert) >= 1.0:
                            self.msg.emit("您已经脱岗，请立刻回到岗位")
                            self.window.emit("您已经脱岗，请立刻回到岗位")
                            off_duty_absent_since = now
                            off_duty_last_alert = now

                primary_rect = None
                if analysis is not None:
                    primary_rect = analysis["bbox_rect"]

                if analysis is None or primary_rect is None:
                    calib_miss_streak += 1
                    if test_time < TEST_TIMES and calib_miss_streak >= CALIB_MISS_ABORT:
                        test_time = TEST_TIMES
                        self.msg.emit(
                            "连续多帧未检测到人脸关键点（MediaPipe 无输出）。已改用默认 EAR/MAR/姿态阈值。"
                            "请正对摄像头、凑近一些、改善逆光/过暗，并确认未戴口罩墨镜。"
                            "Tasks 后端为每帧 IMAGE+detect。请检查 models/ 或 resources/models/ 下 face_landmarker.task "
                            "是否有效、摄像头分辨率与光线是否充足。"
                        )
                        self.window.emit("未稳定检出人脸：请调整姿势与光线（详见日志区说明）")
                    self.picture.emit(frame)
                    if self.isClose:
                        break
                    continue

                calib_miss_streak = 0
                lm_px = analysis["landmarks"]
                rect = primary_rect
                ear = analysis["ear"]
                mar = analysis["mar"]
                reprojectdst = analysis["reprojectdst"]
                pitch = analysis["pitch"]
                yaw = analysis["yaw"]
                roll = analysis["roll"]
                har = pitch

                if self.isShowKeyPoint:
                    for i in range(0, lm_px.shape[0], 8):
                        x, y = int(lm_px[i, 0]), int(lm_px[i, 1])
                        cv2.circle(frame, (x, y), 1, (255, 8, 0), -1)

                if self.isShowHead:
                    for start, end in self.fatigue_detector.line_pairs:
                        start_point = (int(reprojectdst[start][0]), int(reprojectdst[start][1]))
                        end_point = (int(reprojectdst[end][0]), int(reprojectdst[end][1]))
                        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    name = self.face_recognition.recognize_face(face_img)
                    if name:
                        frame = draw_text_cn_on_bgr(frame, f"姓名: {name}", (left, top - 25), font_size=16, color=(0, 255, 255))

                cv2.putText(frame, "ear: {:.2f}".format(ear), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "mar: {:.2f}".format(mar), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "pitch: {:5.2f}".format(pitch), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "yaw: {:5.2f}".format(yaw), (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "roll: {:5.2f}".format(roll), (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 动态阈值：每帧最多累计一次（整图单脸标定）
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
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                    self.picture.emit(frame)
                    if self.isClose:
                        break
                    continue

                if self.isShowEye:
                    eye_pts_idx = [33, 133, 159, 145, 263, 362, 386, 374]
                    for idx in eye_pts_idx:
                        x, y = int(lm_px[idx, 0]), int(lm_px[idx, 1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if self.isShowMouth:
                    mouth_pts_idx = [61, 291, 13, 14]
                    for idx in mouth_pts_idx:
                        x, y = int(lm_px[idx, 0]), int(lm_px[idx, 1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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
                    period_len = 60
                    Detected_TIME_LIMIT = period_len
                    isEyeTired = False
                    isYawnTired = False
                    isHeadTired = False

                    if closed_times / period_len > FATIGUE_THRESH:
                        self.msg.emit("闭眼时长较长")
                        isEyeTired = True

                    if yawning_times / period_len > FATIGUE_THRESH:
                        self.msg.emit("张嘴时长较长")
                        isYawnTired = True

                    if pitch_times / period_len > FATIGUE_THRESH:
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
