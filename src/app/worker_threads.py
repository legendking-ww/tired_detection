"""疲劳检测与摄像头调整线程。"""
from __future__ import annotations

import os
import time
from collections import Counter
import cv2
import winsound
from pygame import mixer
from PyQt5.QtCore import QThread, pyqtSignal

from src.core.fatigue_detection import FatigueDetector
from src.core.face_recognition import FaceRecognition
from src.multimodal.audio_loop import start_audio_loop, stop_audio_loop
from src.multimodal.agent_runner import schedule_llm_fatigue_agent
from src.multimodal import config as mm_cfg_static
from src.multimodal.config import (
    groq_api_key,
    is_agent_local_tts_enabled,
    is_llm_agent_enabled,
    is_multimodal_enabled,
    is_multimodal_mic,
    is_multimodal_video_audio,
    llm_agent_cooldown_sec,
)
from src.multimodal import video_audio_ctx
from src.multimodal.fusion import alert_level, fuse_visual_audio, fuse_visual_audio_dynamic
from src.multimodal.state import (
    clear_fusion_display,
    get_last_audio_score,
    get_last_fusion,
    get_last_transcript,
    set_last_fusion,
)
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
        # UI 预览节流：主线程跨线程收图过密会排队卡顿。0 表示每帧都发；默认约 30fps。
        try:
            _ms = os.environ.get("TIRED_UI_PREVIEW_MIN_MS", "33").strip()
            self._ui_preview_interval = max(0.0, float(_ms or "0") / 1000.0)
        except ValueError:
            self._ui_preview_interval = 0.033
        self._last_picture_emit_time = 0.0

    def emit_picture_if_due(self, frame) -> None:
        """按 TIRED_UI_PREVIEW_MIN_MS 节流送预览帧；检测线程仍每帧跑算法。"""
        if frame is None:
            return
        if self._ui_preview_interval <= 0:
            self.picture.emit(frame)
            return
        t = time.monotonic()
        if t - self._last_picture_emit_time >= self._ui_preview_interval:
            self._last_picture_emit_time = t
            self.picture.emit(frame)

    def change_cam_select(self, camSelect):
        self.camSelect = camSelect

    def close(self):
        """仅请求线程退出；不要在 UI 线程里 release VideoCapture，避免与 worker 的 read() 竞态。"""
        self.isClose = True

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

                self.emit_picture_if_due(frame)
                
                if self.isClose:
                    break
        except Exception as e:
            error_msg = f"摄像头调整失败: {str(e)}"
            print(error_msg)
            self.msg.emit(error_msg)
        finally:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
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
        self._mm_last_audio_err = None
        self._log_throttle = {}
        self._last_warn_beep = 0.0
        self._last_strong_fatigue_alert = 0.0
        self._last_llm_agent_run = 0.0

    def _emit_msg_throttled(self, key, message, min_interval_sec=22.0):
        """同一 key 的提示在 min_interval_sec 内最多发一次，减轻日志与弹窗骚扰。"""
        now = time.monotonic()
        if now - self._log_throttle.get(key, 0.0) >= min_interval_sec:
            self._log_throttle[key] = now
            self.msg.emit(message)

    def _beep_warn_throttled(self, min_interval_sec=14.0, duration_ms=380, freq=440):
        now = time.monotonic()
        if now - self._last_warn_beep >= min_interval_sec:
            self._last_warn_beep = now
            winsound.Beep(freq, duration_ms)

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
        try:
            mixer.init()
            mixer.music.load("resources/sounds/warning.mp3")
            mixer.music.play()
        except Exception as e:
            print(f"提示音播放失败（可检查 resources/sounds/warning.mp3 与音频设备）: {e}")
    
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

            # 统计窗口为「帧数」不是秒：原 60 帧在 30fps 下仅约 2s，告警会非常密。默认改为 150 帧（约 5s@30fps），可用环境变量调节。
            try:
                stats_period_frames = max(45, int(os.environ.get("TIRED_STATS_PERIOD_FRAMES", "150").strip() or "150"))
            except ValueError:
                stats_period_frames = 150
            try:
                strong_alert_cd = max(10.0, float(os.environ.get("TIRED_STRONG_ALERT_COOLDOWN_SEC", "45").strip() or "45"))
            except ValueError:
                strong_alert_cd = 45.0
            Detected_TIME_LIMIT = stats_period_frames
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

            # 姓名：默认前若干秒抽样做人脸比对，多数表决后锁定，之后不再调用 recognize_face（省算力）
            # TIRED_FACE_NAME_PROBE_SEC：>0 为探测秒数；0 关闭叠字姓名；<0（如 -1）每帧比对（旧行为，开销大）
            _probe_raw = os.environ.get("TIRED_FACE_NAME_PROBE_SEC", "12").strip()
            try:
                face_name_probe_sec = float(_probe_raw) if _probe_raw else 12.0
            except ValueError:
                face_name_probe_sec = 12.0
            face_name_sample_every = max(1, int(os.environ.get("TIRED_FACE_NAME_SAMPLE_FRAMES", "4")))
            face_session_t0 = time.monotonic()
            face_name_hits: list[str] = []
            face_name_locked: str | None = None
            face_name_finalized = False
            face_name_frame_tick = 0
            face_name_last_hit: str | None = None

            smooth_visual = 0.0
            danger_streak_mm = 0

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

            if is_multimodal_enabled():
                clear_fusion_display()
                start_audio_loop()
                has_key = bool(groq_api_key())
                if is_multimodal_video_audio() and has_key:
                    self.msg.emit(
                        "多模态：已启用「视频伴音」模式（播放视频文件时从音轨抽音频，需本机安装 ffmpeg 并在 PATH 中）。"
                        "未播放视频时将回退麦克风或 WAV。"
                    )
                if is_multimodal_mic():
                    if has_key:
                        self.msg.emit(
                            "多模态已开启（麦克风）：按 TIRED_MULTIMODAL_RECORD_SEC 录制，"
                            "TIRED_MULTIMODAL_INTERVAL 控制分析周期；已读取 API 密钥，语音疲劳分析将自动进行。"
                        )
                    else:
                        self.msg.emit(
                            "多模态已开启（麦克风）：未检测到 API 密钥。请在项目根目录 .env 中设置 "
                            "SILICONFLOW_API_KEY（或 MULTIMODAL_API_KEY / GROQ_API_KEY）与 GROQ_API_BASE 后重启程序。"
                        )
                else:
                    if has_key:
                        self.msg.emit(
                            "多模态已开启（文件）：按 TIRED_MULTIMODAL_INTERVAL 轮询 WAV；已读取 API 密钥。"
                            "请确认 TIRED_MULTIMODAL_WAV 或 resources/samples/driver_demo.wav 存在。"
                        )
                    else:
                        self.msg.emit(
                            "多模态已开启（文件）：未检测到 API 密钥。请在 .env 中配置密钥与 GROQ_API_BASE；"
                            "并设置 TIRED_MULTIMODAL_WAV 或放置 resources/samples/driver_demo.wav。"
                        )

            while True:
                ret, raw = self.cap.read()
                if is_multimodal_enabled():
                    if ret and self.isOpenVideo and self.filePath:
                        try:
                            pos = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                        except Exception:
                            pos = 0.0
                        video_audio_ctx.set_playback(self.filePath, pos)
                    else:
                        video_audio_ctx.set_playback("", 0.0)
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
                    self.emit_picture_if_due(frame)
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

                left = int(rect.left())
                top = int(rect.top())
                right = int(rect.right())
                bottom = int(rect.bottom())
                fh, fw = frame.shape[:2]
                left_c = max(0, min(left, fw - 1))
                right_c = max(left_c + 1, min(right, fw))
                top_c = max(0, min(top, fh - 1))
                bottom_c = max(top_c + 1, min(bottom, fh))
                face_img = frame[top_c:bottom_c, left_c:right_c]
                if face_img.size > 0:
                    display_name = None
                    if face_name_probe_sec < 0:
                        display_name = self.face_recognition.recognize_face(face_img)
                    elif face_name_probe_sec == 0:
                        display_name = None
                    else:
                        elapsed = time.monotonic() - face_session_t0
                        if not face_name_finalized:
                            if elapsed <= face_name_probe_sec:
                                face_name_frame_tick += 1
                                if face_name_frame_tick % face_name_sample_every == 0:
                                    got = self.face_recognition.recognize_face(face_img)
                                    if got:
                                        face_name_hits.append(got)
                                        face_name_last_hit = got
                                display_name = face_name_last_hit
                            else:
                                face_name_finalized = True
                                if face_name_hits:
                                    face_name_locked = Counter(face_name_hits).most_common(1)[0][0]
                                    self.msg.emit(
                                        f"已锁定姓名：{face_name_locked}（前 {face_name_probe_sec:.0f} 秒内多数表决），"
                                        f"后续已关闭人脸比对以减轻负载。"
                                    )
                                else:
                                    self.msg.emit(
                                        f"前 {face_name_probe_sec:.0f} 秒未匹配到注册人脸，本段检测不叠示姓名。"
                                    )
                                display_name = face_name_locked
                        else:
                            display_name = face_name_locked

                    if display_name:
                        frame = draw_text_cn_on_bgr(
                            frame,
                            f"姓名: {display_name}",
                            (left, top - 25),
                            font_size=16,
                            color=(0, 255, 255),
                        )

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
                    self.emit_picture_if_due(frame)
                    if self.isClose:
                        break
                    continue

                n_lm = int(lm_px.shape[0])
                if self.isShowEye:
                    eye_pts_idx = [33, 133, 159, 145, 263, 362, 386, 374]
                    for idx in eye_pts_idx:
                        if idx >= n_lm:
                            break
                        x, y = int(lm_px[idx, 0]), int(lm_px[idx, 1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if self.isShowMouth:
                    mouth_pts_idx = [61, 291, 13, 14]
                    for idx in mouth_pts_idx:
                        if idx >= n_lm:
                            break
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
                    period_len = stats_period_frames
                    Detected_TIME_LIMIT = period_len
                    isEyeTired = False
                    isYawnTired = False
                    isHeadTired = False

                    if closed_times / period_len > FATIGUE_THRESH:
                        isEyeTired = True

                    if yawning_times / period_len > FATIGUE_THRESH:
                        isYawnTired = True

                    if pitch_times / period_len > FATIGUE_THRESH:
                        isHeadTired = True

                    weighted_mm_msgs = False
                    if is_multimodal_enabled():
                        from src.multimodal import config as mm_cfg

                        logic = mm_cfg.fatigue_logic_mode()
                        weighted_mm_msgs = logic == "weighted"
                        if logic == "legacy":
                            thr = max(float(FATIGUE_THRESH), 1e-6)
                            v_eye = min(1.0, (closed_times / float(period_len)) / (2.0 * thr))
                            v_yawn = min(1.0, (yawning_times / float(period_len)) / (2.0 * thr))
                            v_head = min(1.0, (pitch_times / float(period_len)) / (2.0 * thr))
                            visual_score = max(v_eye, v_yawn, v_head)
                            visual_for_fusion = visual_score
                        else:
                            re = closed_times / float(period_len)
                            ry = yawning_times / float(period_len)
                            rh = pitch_times / float(period_len)
                            sat_e = mm_cfg.visual_sat_eye()
                            sat_y = mm_cfg.visual_sat_yawn()
                            sat_h = mm_cfg.visual_sat_head()
                            v_eye = min(1.0, re / sat_e)
                            v_yawn = min(1.0, ry / sat_y)
                            v_head = min(1.0, rh / sat_h)
                            we, wy, wh = (
                                mm_cfg.visual_w_eye(),
                                mm_cfg.visual_w_yawn(),
                                mm_cfg.visual_w_head(),
                            )
                            sw = we + wy + wh
                            if sw <= 1e-6:
                                visual_score = 0.0
                            else:
                                visual_score = (we * v_eye + wy * v_yawn + wh * v_head) / sw
                            alpha = mm_cfg.visual_smooth_alpha()
                            smooth_visual = alpha * visual_score + (1.0 - alpha) * smooth_visual
                            smooth_visual = max(0.0, min(1.0, smooth_visual))
                            visual_for_fusion = smooth_visual

                        audio_s, audio_err = get_last_audio_score()
                        if (
                            audio_err
                            and (audio_s is None or audio_s < 0)
                            and self._mm_last_audio_err != audio_err
                        ):
                            self._mm_last_audio_err = audio_err
                            self.msg.emit(f"语音疲劳分析：{audio_err}")
                        if audio_s is not None and audio_s >= 0 and audio_err is None:
                            self._mm_last_audio_err = None
                        if audio_s is None or audio_s < 0:
                            fused = visual_for_fusion
                        elif logic == "legacy":
                            fused = fuse_visual_audio(
                                visual_for_fusion,
                                float(audio_s),
                                mm_cfg.visual_weight(),
                                mm_cfg.audio_weight(),
                            )
                        else:
                            fused = fuse_visual_audio_dynamic(visual_for_fusion, float(audio_s))

                        lvl = alert_level(fused)
                        set_last_fusion(visual_for_fusion, fused, lvl)
                        lvl_cn = {"danger": "危险", "watch": "注意", "normal": "正常"}.get(lvl, lvl)
                        if lvl == "danger":
                            mm_color = (40, 40, 255)
                        elif lvl == "watch":
                            mm_color = (0, 180, 255)
                        else:
                            mm_color = (60, 200, 80)
                        aud_txt = f"{float(audio_s):.2f}" if audio_s is not None and audio_s >= 0 else "--"
                        vis_label = "视觉" if logic == "legacy" else "视觉(平滑)"
                        frame = self.put_text_cn(
                            frame,
                            f"【多模态】融合 {fused:.2f}  {vis_label}{visual_for_fusion:.2f}  语音{aud_txt}  [{lvl_cn}]",
                            (10, 95),
                            font_size=20,
                            color=mm_color,
                        )
                        tx = get_last_transcript()
                        if tx:
                            preview = tx[:56] + ("…" if len(tx) > 56 else "")
                            frame = self.put_text_cn(
                                frame,
                                f"转写：{preview}",
                                (10, frame.shape[0] - 48),
                                font_size=16,
                                color=(0, 255, 255),
                            )

                        w_thr = mm_cfg.alert_watch_threshold()
                        d_thr = mm_cfg.alert_danger_threshold()
                        mid_thr = mm_cfg.alert_watch_mid()

                        if logic == "weighted":
                            if fused >= d_thr:
                                self._emit_msg_throttled(
                                    "fused_danger",
                                    "【融合】极度疲劳，请立即休息！",
                                    22.0,
                                )
                            elif fused >= mid_thr:
                                self._emit_msg_throttled(
                                    "fused_watch_hi",
                                    "【融合】中度疲劳，建议休息。",
                                    26.0,
                                )
                            elif fused >= w_thr:
                                self._emit_msg_throttled(
                                    "fused_watch_lo",
                                    "【融合】轻微疲劳，请注意。",
                                    30.0,
                                )
                            streak_need = mm_cfg.danger_streak_windows()
                            if lvl == "danger":
                                danger_streak_mm += 1
                            else:
                                danger_streak_mm = 0
                            if lvl == "danger" and danger_streak_mm >= streak_need:
                                warning_time += 2
                            elif lvl == "watch":
                                self._beep_warn_throttled(min_interval_sec=35.0, duration_ms=220, freq=880)
                        else:
                            if lvl == "danger":
                                warning_time += 2
                            elif lvl == "watch":
                                self._emit_msg_throttled(
                                    "mm_watch",
                                    "【注意级】多模态综合分数偏高，建议稍作休息（仅日志与画面提示，无强弹窗）。",
                                    40.0,
                                )
                                self._beep_warn_throttled(min_interval_sec=35.0, duration_ms=220, freq=880)

                    if not weighted_mm_msgs:
                        if isEyeTired:
                            self._emit_msg_throttled("eye_long", "闭眼时长较长", 22.0)
                        if isYawnTired:
                            self._emit_msg_throttled("yawn_long", "张嘴时长较长", 22.0)
                        if isHeadTired:
                            self._emit_msg_throttled("head_low", "低头时长较长", 22.0)

                    closed_times = 0
                    yawning_times = 0
                    pitch_times = 0

                    isWarning = False
                    if weighted_mm_msgs:
                        pass
                    elif isEyeTired and isYawnTired:
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
                        now_alert = time.monotonic()
                        if now_alert - self._last_strong_fatigue_alert >= strong_alert_cd:
                            self._last_strong_fatigue_alert = now_alert
                            self.msg.emit("【危险级】您已经疲劳，请注意休息!")
                            self.window.emit("您已经疲劳，请注意休息!")
                            self.playMusic()
                            if is_llm_agent_enabled() and groq_api_key():
                                cd_agent = llm_agent_cooldown_sec()
                                if now_alert - self._last_llm_agent_run >= cd_agent:
                                    self._last_llm_agent_run = now_alert
                                    vis_g, fus_g, lvl_g = get_last_fusion()
                                    aud_g, aerr_g = get_last_audio_score()
                                    ctx = {
                                        "trigger": "strong_fatigue_popup",
                                        "fused_score": fus_g,
                                        "visual_score": vis_g,
                                        "alert_level": lvl_g or "",
                                        "alert_watch": mm_cfg_static.alert_watch_threshold(),
                                        "alert_danger": mm_cfg_static.alert_danger_threshold(),
                                        "fatigue_logic": mm_cfg_static.fatigue_logic_mode(),
                                        "danger_streak_windows": mm_cfg_static.danger_streak_windows(),
                                        "audio_score": aud_g,
                                        "audio_analysis_error": aerr_g,
                                        "transcript_snippet": (get_last_transcript() or "")[:600],
                                        "visual_flags": {
                                            "eye_tired_window": isEyeTired,
                                            "yawn_tired_window": isYawnTired,
                                            "head_tired_window": isHeadTired,
                                        },
                                        "multimodal_enabled": is_multimodal_enabled(),
                                        "multimodal_mic": is_multimodal_mic(),
                                        "agent_local_tts": is_agent_local_tts_enabled(),
                                    }
                                    self.msg.emit(
                                        "[Agent] 已排队：强疲劳告警后将异步请求 LLM；"
                                        "侧栏多模态区橙色「主动安全 Agent」条与右侧流水会更新。"
                                    )
                                    schedule_llm_fatigue_agent(ctx, self.msg.emit)
                        else:
                            self.msg.emit(
                                "【危险级】疲劳信号持续偏高（强弹窗与提示音冷却中，请主动休息）。"
                            )
                    else:
                        if isWarning:
                            self._beep_warn_throttled()

                self.emit_picture_if_due(frame)

                if self.isClose:
                    break
        except Exception as e:
            error_msg = f"程序运行失败: {str(e)}"
            print(error_msg)
            self.msg.emit(error_msg)
        finally:
            if is_multimodal_enabled():
                try:
                    video_audio_ctx.clear_playback()
                except Exception:
                    pass
                try:
                    stop_audio_loop()
                except Exception:
                    pass
                try:
                    clear_fusion_display()
                except Exception:
                    pass
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
