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
from src.utils.history import DetectionHistory
from src.utils.logger import get_logger

_log = get_logger(__name__)


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
            except Exception:
                pass

    def stop(self, timeout_ms: int = 3000):
        """优雅停止：置关闭标志 → 释放摄像头 → 等待线程结束（超时强制终止）。"""
        self.isClose = True
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.isRunning():
            if not self.wait(timeout_ms):
                _log.warning("Thread %s did not finish within %dms, terminating", self.__class__.__name__, timeout_ms)
                self.terminate()
                self.wait(1000)

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
            _log.error(error_msg)
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
        self._mm_last_audio_err = None
        self._log_throttle = {}
        self._last_warn_beep = 0.0
        self._last_strong_fatigue_alert = 0.0
        self._last_llm_agent_run = 0.0
        self._cam_consec_fail = 0
        self._frame_idx_global = 0
        # 推理分级：根据疲劳等级动态跳帧
        self._inference_skip = 0  # 0=每帧, 1=隔1帧, 2=隔2帧
        self._inference_skip_counter = 0
        self._approx_fps = 30.0
        self._fps_update_time = time.monotonic()
        self._fps_frame_count = 0
        # 历史数据记录句柄
        self._history_writer: DetectionHistory | None = None
        self._last_analysis = None

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
        mixer.init()
        mixer.music.load('resources/sounds/warning.mp3')
        mixer.music.play()

    def _update_inference_tier(self, fused_score: float | None, alert_lvl: str) -> None:
        """根据疲劳等级动态调整推理跳帧策略。"""
        min_fps = max(5, min(30, int(os.environ.get("TIRED_INFERENCE_MIN_FPS", "10"))))
        if alert_lvl == "danger":
            self._inference_skip = 0  # 危险：每帧推理
        elif alert_lvl == "watch":
            self._inference_skip = max(0, int(30.0 / max(1, min_fps * 1.5)) - 1)  # 注意：适度跳帧
        elif fused_score is not None and fused_score >= 0.15:
            self._inference_skip = max(0, int(30.0 / max(1, min_fps)) - 1)  # 轻微：跳一些帧
        else:
            self._inference_skip = max(0, int(30.0 / max(1, min_fps * 0.75)) - 1)  # 正常：最多跳帧
        # 人脸丢失时强制全帧
        if self._last_analysis is None:
            self._inference_skip = 0

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
            stats_period_frames = max(45, int(os.environ.get("TIRED_STATS_PERIOD_FRAMES", "150")))
            strong_alert_cd = max(10.0, float(os.environ.get("TIRED_STRONG_ALERT_COOLDOWN_SEC", "45")))
            Detected_TIME_LIMIT = stats_period_frames
            closed_times = 0
            yawning_times = 0
            pitch_times = 0
            perclos_high_frames = 0
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

            # ---- 历史数据记录 ----
            try:
                self._history_writer = DetectionHistory("mrsoft.db")
                self._history_writer.init_tables()
                self._history_writer.start_session()
                _log.info("检测历史记录已启动")
            except Exception as e:
                _log.warning("历史记录初始化失败（不影响检测）: %s", e)
                self._history_writer = None

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
                        _log.info("视频播放结束")
                        break
                    # 摄像头断连：尝试指数退避重连
                    self._cam_consec_fail += 1
                    max_retry = max(1, min(10, int(os.environ.get("TIRED_CAM_RECONNECT_MAX", "5"))))
                    if self._cam_consec_fail > max_retry:
                        _log.error("摄像头连续 %d 次读取失败，已达最大重试次数，退出检测", max_retry)
                        self.window.emit(f"摄像头连续{max_retry}次断连，已放弃重连。请检查设备后重新开始。")
                        break
                    backoff = min(8.0, 2 ** (self._cam_consec_fail - 1))
                    self.msg.emit(f"摄像头断开，正在重连 ({self._cam_consec_fail}/{max_retry})…")
                    _log.warning("摄像头读帧失败 (连续%d次)，%0.1fs 后重连…", self._cam_consec_fail, backoff)
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    # 退避等待中分段检查 isClose，避免阻塞退出
                    for _t in range(int(backoff * 5)):
                        time.sleep(0.2)
                        if self.isClose:
                            break
                    if self.isClose:
                        break
                    self.cap = open_video_capture_by_index(self.camSelect)
                    if self.cap is not None:
                        try:
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        self._cam_consec_fail = 0
                        _log.info("摄像头已重连成功")
                        self.msg.emit("摄像头已重连，继续检测。")
                    else:
                        _log.warning("摄像头重连失败，将再试…")
                    continue

                # ---- 帧率估算 ----
                self._fps_frame_count += 1
                now_fps = time.monotonic()
                if now_fps - self._fps_update_time >= 2.0:
                    self._approx_fps = self._fps_frame_count / max(0.1, now_fps - self._fps_update_time)
                    self._fps_frame_count = 0
                    self._fps_update_time = now_fps

                frame, gray, _ = self.process_frame(raw)

                # ---- 推理分级：根据疲劳等级动态跳帧（跳过 MediaPipe / FaceNet 等重推理）----
                self._frame_idx_global += 1
                if test_time >= TEST_TIMES and self._inference_skip > 0:
                    self._inference_skip_counter += 1
                    if self._inference_skip_counter % (self._inference_skip + 1) != 0:
                        analysis = self._last_analysis
                    else:
                        analysis = self.fatigue_detector.analyze_face(frame)
                        self._last_analysis = analysis
                else:
                    analysis = self.fatigue_detector.analyze_face(frame)
                    self._last_analysis = analysis
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

                # PERCLOS + 眨眼分析：每帧更新（使用标定后阈值或默认 0.32）
                _blink_thr = EAR_THRESH if test_time >= TEST_TIMES else 0.32
                blink_metrics = self.fatigue_detector.blink_analyzer.update(ear, _blink_thr, self._approx_fps)

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
                # PERCLOS overlay（疲劳时显示醒目颜色）
                _pclos = blink_metrics["perclos"]
                _pclos_color = (40, 40, 255) if _pclos > 0.25 else (0, 200, 255) if _pclos > 0.12 else (200, 200, 200)
                cv2.putText(frame, "pclos:{:.2f} blk:{:.1f}/m".format(
                    _pclos, blink_metrics["blink_rate"]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _pclos_color, 2)

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
                        _log.info('眼睛长宽比ear 100次取平均的阈值:%.2f', EAR_THRESH)
                        _log.info('嘴部长宽比mar 100次取平均的阈值:%.2f', MAR_THRESH)
                        _log.info('头部俯仰角pitch 100次取平均的阈值:%.2f', HAR_THRESH)
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
                    # PERCLOS：> 0.15 视为升高
                    if blink_metrics["perclos"] > 0.15:
                        perclos_high_frames += 1

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

                    isPERCLOSTired = False
                    if perclos_high_frames / period_len > FATIGUE_THRESH:
                        isPERCLOSTired = True

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
                            rp = perclos_high_frames / float(period_len)
                            sat_e = mm_cfg.visual_sat_eye()
                            sat_y = mm_cfg.visual_sat_yawn()
                            sat_h = mm_cfg.visual_sat_head()
                            # PERCLOS 饱和比与权重（可用环境变量覆写）
                            sat_p = float(os.environ.get("TIRED_VISUAL_SAT_PERCLOS", "0.15"))
                            wp = float(os.environ.get("TIRED_VISUAL_W_PERCLOS", "0.10"))
                            v_eye = min(1.0, re / sat_e)
                            v_yawn = min(1.0, ry / sat_y)
                            v_head = min(1.0, rh / sat_h)
                            v_perclos = min(1.0, rp / sat_p)
                            we, wy, wh = (
                                mm_cfg.visual_w_eye(),
                                mm_cfg.visual_w_yawn(),
                                mm_cfg.visual_w_head(),
                            )
                            # 原有三通道 + PERCLOS
                            sw = we + wy + wh + wp
                            if sw <= 1e-6:
                                visual_score = 0.0
                            else:
                                visual_score = (we * v_eye + wy * v_yawn + wh * v_head + wp * v_perclos) / sw
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
                        self._update_inference_tier(fused, lvl)
                        set_last_fusion(visual_for_fusion, fused, lvl)
                        # ---- 记录窗口数据到历史 ----
                        if self._history_writer is not None:
                            try:
                                bp = blink_metrics["perclos"] if "blink_metrics" in dir() else 0.0
                                br = blink_metrics["blink_rate"] if "blink_metrics" in dir() else 0.0
                                self._history_writer.record_window(
                                    ts=time.time(),
                                    ear=ear, mar=mar, pitch=pitch,
                                    perclos=bp, blink_rate=br,
                                    visual_score=visual_for_fusion,
                                    audio_score=float(audio_s) if audio_s is not None and audio_s >= 0 else None,
                                    fused_score=fused,
                                    alert_level=lvl,
                                )
                            except Exception:
                                pass
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
                    perclos_high_frames = 0

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
                            # ---- 记录告警事件 ----
                            if self._history_writer is not None:
                                try:
                                    _vis, _fus, _lvl = get_last_fusion()
                                    self._history_writer.record_alert(
                                        ts=time.time(),
                                        alert_type="strong_fatigue",
                                        fused_score=_fus,
                                    )
                                except Exception:
                                    pass
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

                self.picture.emit(frame)

                if self.isClose:
                    break
        except Exception as e:
            error_msg = f"程序运行失败: {str(e)}"
            _log.error(error_msg)
            self.msg.emit(error_msg)
        finally:
            # ---- 结束历史会话 ----
            if self._history_writer is not None:
                try:
                    self._history_writer.end_session()
                except Exception:
                    pass
            if is_multimodal_enabled():
                video_audio_ctx.clear_playback()
                stop_audio_loop()
                clear_fusion_display()
            if self.cap is not None:
                self.cap.release()
