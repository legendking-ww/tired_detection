import math
from collections import deque

import cv2
import os
import importlib
import urllib.request

import numpy as np

from ..utils.utils import line_pairs, reprojectsrc
from ..utils.logger import get_logger

_log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 官方 float16 版约 3MB；过小视为损坏或占位文件
_MIN_LANDMARKER_TASK_BYTES = 512 * 1024


def _abs_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


# MediaPipe / TFLite 在 CPU 上运行时会打印大量 INFO/WARNING（与 CUDA 无关）。
# 在 import mediapipe 之前设置日志级别，避免终端“像报错一样”刷屏。
# GLOG: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL（默认用 3，尽量屏蔽 W0000 类提示）
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import mediapipe as mp  # noqa: F401
except Exception:
    mp = None

# MediaPipe Tasks FaceLandmarker 与旧版 mp.solutions.face_mesh 差异（维护时勿混用）：
# - Tasks 一般为 **478** 个 3D 点，(x,y) 归一化到 [0,1]，须乘宽高得像素；RGB 数组须 uint8 且连续。
# - face_mesh 常见 **468** 点（refine 后 478）。**FACEMESH_*** 连接表与 Tasks 的
#   **FaceLandmarksConnections** 不是同一套，不能把 solutions 的绘制代码直接套在 Tasks 输出上。
# - 运行模式：IMAGE→detect()；VIDEO→detect_for_video(帧, 单调时间戳)；LIVE_STREAM→detect_async。
#   本模块 Tasks 路径使用 **IMAGE + detect**（与官方单帧示例一致），避免 VIDEO 偶发空帧。
# - 排障：有 478 点仍画不出 → 查坐标/颜色空间/模型路径；长期无点 → 调置信度或光线。
# - EAR/MAR 所用索引在 0–467 内与常见脸模拓扑一致；更换模型须自行核对索引。


class SimpleRect:
    """人脸框矩形（left/top/right/bottom），与 OpenCV 裁剪一致。"""

    def __init__(self, left, top, right, bottom):
        self._l = int(left)
        self._t = int(top)
        self._r = int(right)
        self._b = int(bottom)

    def left(self):
        return self._l

    def top(self):
        return self._t

    def right(self):
        return self._r

    def bottom(self):
        return self._b


class BlinkAnalyzer:
    """PERCLOS + 眨眼分析器。

    PERCLOS (Percentage of Eyelid Closure) 是国际上公认最有效的疲劳指标。
    本实现使用 P80 标准：眼睑遮挡超过 80% 的时间占比。

    同时统计：
    - 眨眼频率（次/分钟）
    - 平均眨眼时长（毫秒）
    - 最长眨眼时长（毫秒）

    疲劳特征：
    - PERCLOS 升高（>0.15 需注意，>0.30 危险）
    - 眨眼频率降低（正常约 15-20 次/分钟，疲劳时下降到 <10）
    - 单次眨眼时长变长（正常 100-400ms，疲劳时可达 500-800ms）
    """

    def __init__(self, window_frames: int = 150):
        self._window_frames = max(30, window_frames)
        self._ear_history: deque[float] = deque(maxlen=self._window_frames)
        self._closed_history: deque[bool] = deque(maxlen=self._window_frames)
        # 眨眼状态机
        self._blink_active = False
        self._blink_start_idx = 0
        self._frame_idx = 0
        # 存储最近 N 次眨眼 (duration_frames, duration_ms)
        self._blink_durations: deque[tuple[int, float]] = deque(maxlen=50)
        # 当前帧的结果缓存
        self._last_perclos = 0.0
        self._last_blink_rate = 0.0
        self._last_avg_blink_ms = 0.0
        self._last_max_blink_ms = 0.0
        self._is_blinking = False

    def update(self, ear: float, ear_threshold: float, fps: float = 30.0) -> dict:
        """每帧调用一次。返回 PERCLOS、眨眼频率、平均/最长眨眼时长等指标。

        Args:
            ear: 当前帧的 Eye Aspect Ratio
            ear_threshold: 标定后的个人 EAR 阈值（正常睁眼均值）
            fps: 近似帧率，用于将帧数换算为时间

        Returns:
            dict with keys: perclos, blink_rate, avg_blink_ms, max_blink_ms, is_blinking
        """
        fps = max(1.0, min(120.0, fps))
        self._frame_idx += 1
        self._ear_history.append(ear)

        # PERCLOS P80：EAR 低于阈值 80% 视为闭眼
        closure_threshold = ear_threshold * 0.20  # 80% 遮挡
        is_closed = ear < closure_threshold
        self._closed_history.append(is_closed)

        # 眨眼检测：闭眼 → 睁眼的转换（一次完整眨眼）
        if is_closed and not self._blink_active:
            self._blink_active = True
            self._blink_start_idx = self._frame_idx
        elif not is_closed and self._blink_active:
            self._blink_active = False
            duration_frames = self._frame_idx - self._blink_start_idx
            # 只计入合理范围（2-30 帧 ≈ 67ms-1000ms @30fps）
            if 2 <= duration_frames <= 30:
                duration_ms = (duration_frames / fps) * 1000.0
                self._blink_durations.append((duration_frames, duration_ms))
        self._is_blinking = self._blink_active

        # 计算指标
        total = len(self._closed_history)
        self._last_perclos = sum(self._closed_history) / total if total > 0 else 0.0

        if self._blink_durations:
            durations_ms = [d[1] for d in self._blink_durations]
            self._last_avg_blink_ms = sum(durations_ms) / len(durations_ms)
            self._last_max_blink_ms = max(durations_ms)
        else:
            self._last_avg_blink_ms = 0.0
            self._last_max_blink_ms = 0.0

        # 眨眼频率：最近 window 内的眨眼次数 → 每分钟
        recent_window_sec = total / fps
        if recent_window_sec > 0 and self._blink_durations:
            recent_blinks = sum(1 for d in self._blink_durations if d[0] > self._frame_idx - total)
            self._last_blink_rate = (recent_blinks / recent_window_sec) * 60.0
        else:
            self._last_blink_rate = 0.0

        return {
            "perclos": self._last_perclos,
            "blink_rate": self._last_blink_rate,
            "avg_blink_ms": self._last_avg_blink_ms,
            "max_blink_ms": self._last_max_blink_ms,
            "is_blinking": self._is_blinking,
        }

    @property
    def perclos(self) -> float:
        return self._last_perclos

    @property
    def blink_rate(self) -> float:
        return self._last_blink_rate

    @property
    def avg_blink_ms(self) -> float:
        return self._last_avg_blink_ms

    @property
    def max_blink_ms(self) -> float:
        return self._last_max_blink_ms

    def reset(self) -> None:
        """重置所有状态（新会话或标定后调用）。"""
        self._ear_history.clear()
        self._closed_history.clear()
        self._blink_durations.clear()
        self._blink_active = False
        self._frame_idx = 0
        self._last_perclos = 0.0
        self._last_blink_rate = 0.0
        self._last_avg_blink_ms = 0.0
        self._last_max_blink_ms = 0.0
        self._is_blinking = False


class FatigueDetector:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.face_net = None
        self.device = None
        self.yolo_face = None
        self.last_model_error = None
        self._analyze_face_error_logged = False

        self.line_pairs = line_pairs

        self._legacy_face_mesh = None
        self._lm_backend = None
        self._face_landmarker = None
        self._mp_image_cls = None
        self._mp_image_format = None

        # PERCLOS + 眨眼分析
        self.blink_analyzer = BlinkAnalyzer(window_frames=150)

        self.load_models()

    def load_models(self):
        try:
            self.last_model_error = None
            self.detector = None
            self.predictor = None
            self._legacy_face_mesh = None
            self._face_landmarker = None
            self._mp_image_cls = None
            self._mp_image_format = None
            self._lm_backend = None

            try:
                import torch
                from ..models.facenet import InceptionResnetV1

                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.face_net = InceptionResnetV1().to(self.device)
                facenet_path = _abs_path("resources", "weights", "facenet_best_server.pt")
                if not os.path.isfile(facenet_path):
                    raise FileNotFoundError(f"FaceNet weights not found: {facenet_path}")
                self.face_net.load_state_dict(torch.load(facenet_path, map_location="cpu"))
                self.face_net.eval()
            except Exception as e:
                self.device = None
                self.face_net = None
                _log.warning("FaceNet load skipped: %s", e)

            try:
                from ..models.yolo_face_detect import YOLO_face

                yolo_path = _abs_path("resources", "weights", "yolo_face.onnx")
                if not os.path.isfile(yolo_path):
                    raise FileNotFoundError(f"YOLO onnx not found: {yolo_path}")
                self.yolo_face = YOLO_face(yolo_path)
            except Exception as e:
                self.yolo_face = None
                _log.warning("YOLO face load skipped: %s", e)

            if mp is None:
                raise ImportError("mediapipe is not installed. Run: pip install mediapipe")

            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                try:
                    self._legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=2,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._lm_backend = "legacy_mesh"
                    _log.info("Models loaded OK (MediaPipe solutions.face_mesh)")
                except Exception as e:
                    _log.warning("face_mesh load skipped: %s", e)
                    self._legacy_face_mesh = None

            if self._lm_backend is None:
                BaseOptions = importlib.import_module("mediapipe.tasks.python.core.base_options").BaseOptions
                vision = importlib.import_module("mediapipe.tasks.python.vision")
                FaceLandmarker = vision.FaceLandmarker
                FaceLandmarkerOptions = vision.FaceLandmarkerOptions
                RunningMode = vision.RunningMode

                mp_image_mod = importlib.import_module("mediapipe.tasks.python.vision.core.image")
                self._mp_image_cls = mp_image_mod.Image
                self._mp_image_format = mp_image_mod.ImageFormat

                model_path = self._ensure_face_landmarker_model()
                # 每帧 IMAGE + detect：与官方单帧管线一致，避免 VIDEO
                # detect_for_video 在跟踪过程中偶发整帧无 landmark。
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.1,
                    min_face_presence_confidence=0.1,
                    min_tracking_confidence=0.1,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                )
                self._face_landmarker = FaceLandmarker.create_from_options(options)
                self._lm_backend = "tasks"
                _log.info("Models loaded OK (MediaPipe Tasks FaceLandmarker; IMAGE+detect)")

            if self._lm_backend is None:
                raise RuntimeError("无法加载任何人脸关键点后端（face_mesh 与 FaceLandmarker 均失败）")
        except Exception as e:
            self.last_model_error = str(e)
            _log.error("Model load failed: %s", e)

    def _ensure_face_landmarker_model(self) -> str:
        """
        解析顺序（本地缓存，不重复下载）：
        1) 项目根 models/face_landmarker.task（推荐）
        2) resources/models/face_landmarker.task（兼容旧路径）
        3) 若均不存在则下载到 models/
        """
        primary = _abs_path("models", "face_landmarker.task")
        legacy = _abs_path("resources", "models", "face_landmarker.task")

        for path in (primary, legacy):
            if os.path.isfile(path) and os.path.getsize(path) >= _MIN_LANDMARKER_TASK_BYTES:
                return path

        os.makedirs(os.path.dirname(primary), exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        _log.info("Downloading FaceLandmarker model to: %s", primary)
        tmp_path = primary + ".download"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, primary)
        except Exception:
            if os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise FileNotFoundError(
                "未找到 face_landmarker.task，且自动下载失败。\n"
                "请将官方 float16 模型复制到以下任一位置后重试：\n"
                f"  - {primary}\n"
                f"  - {legacy}"
            ) from None

        if (not os.path.isfile(primary)) or os.path.getsize(primary) < _MIN_LANDMARKER_TASK_BYTES:
            raise RuntimeError(f"FaceLandmarker 模型无效或过小: {primary}")
        return primary

    @staticmethod
    def _normalize_bgr(frame):
        if frame is None or frame.size == 0:
            raise ValueError("empty frame")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise ValueError("unsupported channel count")
        return np.ascontiguousarray(frame, dtype=np.uint8)

    @staticmethod
    def _enhance_for_landmarks(bgr):
        """偏暗/逆光时提升亮度对比，利于 MediaPipe 出点（不改变几何尺寸）。"""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        merged = cv2.merge((l_ch, a_ch, b_ch))
        return np.ascontiguousarray(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), dtype=np.uint8)

    def get_face_feat(self, face_img):
        try:
            if self.face_net is None or self.device is None:
                return None
            import torch

            if face_img is None or face_img.size == 0 or face_img.ndim != 3 or face_img.shape[2] != 3:
                return None
            h, w = face_img.shape[:2]
            if min(h, w) < 32:
                return None

            face_img = cv2.resize(face_img, dsize=(112, 112))
            face_img = (face_img - 127.5) / 127.5
            face_img = np.transpose(face_img, (2, 0, 1))
            face_img = np.expand_dims(face_img, axis=0)
            face_img_tensor = torch.Tensor(face_img).to(self.device)
            face_feat_tensor = self.face_net(face_img_tensor)
            face_feat = face_feat_tensor.detach().cpu().numpy()
            return face_feat
        except Exception as e:
            _log.warning("Feature extract failed: %s", e)
            return None

    def process_frame(self, frame):
        frame = self._normalize_bgr(frame)
        h, w = frame.shape[:2]
        # 仅缩小过宽画面，避免把小分辨率摄像头强行放大导致与「读图测试」效果不一致
        max_w = 1280
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        h2, w2 = frame.shape[:2]
        rects = [SimpleRect(0, 0, w2 - 1, h2 - 1)]

        return frame, gray, rects

    @staticmethod
    def _rect_to_bbox(rect):
        return int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())

    @staticmethod
    def _clamp_bbox(bbox, w, h):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _dist2(a, b):
        return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))

    @staticmethod
    def _bbox_from_landmarks(lm_px, w, h, margin_ratio=0.08):
        xs = lm_px[:, 0]
        ys = lm_px[:, 1]
        span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
        m = margin_ratio * span
        left = int(max(0, np.floor(xs.min() - m)))
        top = int(max(0, np.floor(ys.min() - m)))
        right = int(min(w - 1, np.ceil(xs.max() + m)))
        bottom = int(min(h - 1, np.ceil(ys.max() + m)))
        if right <= left:
            right = min(w - 1, left + 1)
        if bottom <= top:
            bottom = min(h - 1, top + 1)
        return SimpleRect(left, top, right, bottom)

    def _landmarks_legacy_face_mesh(self, frame_bgr):
        if self._legacy_face_mesh is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        res = self._legacy_face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None
        lm_proto = res.multi_face_landmarks[0]
        n = len(lm_proto.landmark)
        lm_px = np.zeros((n, 3), dtype=np.float32)
        for i, p in enumerate(lm_proto.landmark):
            lm_px[i, 0] = p.x * w
            lm_px[i, 1] = p.y * h
            lm_px[i, 2] = p.z * w
        return lm_px, None

    def _landmarks_on_roi_tasks(self, frame_bgr, rect):
        if self._lm_backend != "tasks" or self._face_landmarker is None:
            return None
        if self._mp_image_cls is None or self._mp_image_format is None:
            return None

        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(self._rect_to_bbox(rect), w, h)

        pad = int(0.15 * max(x2 - x1, y2 - y1))
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w - 1, x2 + pad)
        y2p = min(h - 1, y2 + pad)

        roi = frame_bgr[y1p:y2p, x1p:x2p]
        if roi.size == 0:
            return None

        roi_h0, roi_w0 = roi.shape[:2]

        def _detect_on_bgr(bgr_patch: np.ndarray):
            rgb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            mp_image = self._mp_image_cls(image_format=self._mp_image_format.SRGB, data=rgb)
            return self._face_landmarker.detect(mp_image)

        # 摄像头整图较小时，人脸在 ROI 里像素过少，Tasks 常检不出；先原尺寸再放大重试
        infer_bgr = roi
        infer_scale = 1.0
        infer_h, infer_w = roi_h0, roi_w0
        result = _detect_on_bgr(infer_bgr)
        if not result.face_landmarks:
            mns = min(roi_h0, roi_w0)
            if mns > 0 and mns < 480:
                infer_scale = min(480.0 / float(mns), 3.0)
                infer_w = max(1, int(round(roi_w0 * infer_scale)))
                infer_h = max(1, int(round(roi_h0 * infer_scale)))
                infer_bgr = cv2.resize(roi, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                infer_bgr = np.ascontiguousarray(infer_bgr, dtype=np.uint8)
                result = _detect_on_bgr(infer_bgr)

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        lm_px = np.zeros((len(face_landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks):
            lm_px[i, 0] = (lm.x * infer_w) / infer_scale + x1p
            lm_px[i, 1] = (lm.y * infer_h) / infer_scale + y1p
            lm_px[i, 2] = float(lm.z) * infer_w / infer_scale

        fm = None
        mats = getattr(result, "facial_transformation_matrixes", None) or []
        if len(mats) > 0:
            fm = np.asarray(mats[0], dtype=np.float64)
        return lm_px, fm

    def calc_ear_mar_from_landmarks(self, lm_px):
        # FaceLandmarker 通常为 478 点；EAR/MAR 所用索引需至少覆盖到 386
        if lm_px.shape[0] < 387:
            raise ValueError("landmarks count too small")
        eye_a_outer = lm_px[33, :2]
        eye_a_inner = lm_px[133, :2]
        eye_a_up = lm_px[159, :2]
        eye_a_down = lm_px[145, :2]

        eye_b_outer = lm_px[263, :2]
        eye_b_inner = lm_px[362, :2]
        eye_b_up = lm_px[386, :2]
        eye_b_down = lm_px[374, :2]

        eye_a_w = self._dist2(eye_a_outer, eye_a_inner)
        eye_b_w = self._dist2(eye_b_outer, eye_b_inner)
        eye_a_h = self._dist2(eye_a_up, eye_a_down)
        eye_b_h = self._dist2(eye_b_up, eye_b_down)

        ear_a = (eye_a_h / eye_a_w) if eye_a_w > 1e-6 else 0.0
        ear_b = (eye_b_h / eye_b_w) if eye_b_w > 1e-6 else 0.0
        ear = (ear_a + ear_b) / 2.0

        mouth_l = lm_px[61, :2]
        mouth_r = lm_px[291, :2]
        mouth_up = lm_px[13, :2]
        mouth_down = lm_px[14, :2]
        mouth_w = self._dist2(mouth_l, mouth_r)
        mouth_h = self._dist2(mouth_up, mouth_down)
        mar = (mouth_h / mouth_w) if mouth_w > 1e-6 else 0.0
        return ear, mar

    @staticmethod
    def _pose_from_facial_transformation_matrix(T_4x4, frame_hw):
        """由 MediaPipe 输出的 4×4 面部变换矩阵得到欧拉角与头部立方体投影（不经 solvePnP）。"""
        h, w = int(frame_hw[0]), int(frame_hw[1])
        T = np.asarray(T_4x4, dtype=np.float64).reshape(4, 4)
        R = T[:3, :3].astype(np.float64)
        tvec = T[:3, 3].reshape(3, 1).astype(np.float64)
        rvec, _ = cv2.Rodrigues(R)
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        cam_matrix_local = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs_local = np.zeros((4, 1), dtype=np.float32)
        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rvec, tvec, cam_matrix_local, dist_coeffs_local)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        rotation_mat, _ = cv2.Rodrigues(rvec)
        # OpenCV 4.12+：hconcat 要求各段 dtype 与维度一致；Rodrigues 输出多为 float64，勿与 float32 混接
        Rm = np.asarray(rotation_mat, dtype=np.float64).reshape(3, 3)
        tv = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        pose_mat = np.hstack((Rm, tv)).astype(np.float64)
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch = float(euler_angle[0, 0])
        yaw = float(euler_angle[1, 0])
        roll = float(euler_angle[2, 0])
        return reprojectdst, pitch, yaw, roll

    @staticmethod
    def _head_pose_proxy_from_landmarks(lm_px, frame_hw):
        """矩阵缺失时的几何近似（仍不使用 solvePnP），保证疲劳分支里 pitch 阈值可工作。"""
        h, w = int(frame_hw[0]), int(frame_hw[1])
        le = lm_px[33, :2].astype(np.float64)
        re = lm_px[263, :2].astype(np.float64)
        mid = (le + re) * 0.5
        nose = lm_px[1, :2].astype(np.float64)
        chin = lm_px[152, :2].astype(np.float64)
        iw = float(np.linalg.norm(re - le)) + 1e-6
        roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0] + 1e-6))
        yaw = float((nose[0] - mid[0]) / iw * 45.0)
        pitch = float((chin[1] - nose[1]) / iw * 45.0 - 18.0)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        cam_matrix_local = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1.0],
            ],
            dtype=np.float32,
        )
        dist_coeffs_local = np.zeros((4, 1), dtype=np.float32)
        pr, yr, rr = math.radians(pitch), math.radians(yaw), math.radians(roll)
        Rx = np.array([[1, 0, 0], [0, math.cos(pr), -math.sin(pr)], [0, math.sin(pr), math.cos(pr)]])
        Ry = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]])
        Rz = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
        R = (Rz @ Ry @ Rx).astype(np.float64)
        rvec, _ = cv2.Rodrigues(R)
        tvec = np.array([[0.0], [0.0], [focal_length * 0.6]], dtype=np.float64)
        reprojectdst, _ = cv2.projectPoints(reprojectsrc, rvec, tvec, cam_matrix_local, dist_coeffs_local)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        return reprojectdst, pitch, yaw, roll

    @staticmethod
    def rescale_face_analysis(analysis, w_src, h_src, w_dst, h_dst):
        """将 analyze_face 结果从 (w_src,h_src) 缩放到显示用 (w_dst,h_dst)。"""
        if w_src == w_dst and h_src == h_dst:
            return analysis
        sx = w_dst / float(w_src)
        sy = h_dst / float(h_src)
        out = dict(analysis)
        lm = analysis["landmarks"].copy()
        lm[:, 0] = (lm[:, 0] * sx).astype(np.float32)
        lm[:, 1] = (lm[:, 1] * sy).astype(np.float32)
        lm[:, 2] = (lm[:, 2] * ((sx + sy) * 0.5)).astype(np.float32)
        out["landmarks"] = lm
        br = analysis["bbox_rect"]
        out["bbox_rect"] = SimpleRect(
            int(max(0, br.left() * sx)),
            int(max(0, br.top() * sy)),
            int(min(w_dst - 1, br.right() * sx)),
            int(min(h_dst - 1, br.bottom() * sy)),
        )
        rd = []
        for pt in analysis["reprojectdst"]:
            rd.append((float(pt[0]) * sx, float(pt[1]) * sy))
        out["reprojectdst"] = tuple(tuple(p) for p in rd)
        return out

    def _try_landmarks_full_image(self, frame_bgr, w, h):
        """原图 + 光照增强 + 水平镜像，尽量让 MediaPipe 出点。"""
        if self._lm_backend == "legacy_mesh":
            variants = [frame_bgr, self._enhance_for_landmarks(frame_bgr)]
            for fb in variants:
                fb = np.ascontiguousarray(fb, dtype=np.uint8)
                packed = self._landmarks_legacy_face_mesh(fb)
                if packed is not None:
                    return packed, False
                packed = self._landmarks_legacy_face_mesh(cv2.flip(fb, 1))
                if packed is not None:
                    return packed, True
            return None, False

        full_rect = SimpleRect(0, 0, w - 1, h - 1)
        variants = [frame_bgr, self._enhance_for_landmarks(frame_bgr)]
        for fb in variants:
            fb = np.ascontiguousarray(fb, dtype=np.uint8)
            packed = self._landmarks_on_roi_tasks(fb, full_rect)
            if packed is not None:
                return packed, False
            packed = self._landmarks_on_roi_tasks(cv2.flip(fb, 1), full_rect)
            if packed is not None:
                return packed, True
        return None, False

    def crop_face_for_recognition(self, frame_bgr):
        """YOLO 优先，否则用人脸关键点外接框裁剪，供登录/注册。返回 (人脸图, 框) 或 (None, None)。"""
        frame_bgr = self._normalize_bgr(frame_bgr)
        if self.yolo_face is not None:
            det_bboxes, det_conf, _, _ = self.yolo_face.detect(frame_bgr)
            if len(det_bboxes) > 0 and len(det_conf) > 0:
                best_i = int(np.argmax(det_conf))
                if det_conf[best_i] > 0.5:
                    x, y, w, h = det_bboxes[best_i].astype(int)
                    H, W = frame_bgr.shape[:2]
                    if y >= 0 and y + h <= H and x >= 0 and x + w <= W and h > 0 and w > 0:
                        br = SimpleRect(x, y, x + w - 1, y + h - 1)
                        return frame_bgr[y : y + h, x : x + w], br
        analysis = self.analyze_face(frame_bgr)
        if analysis is None:
            return None, None
        br = analysis["bbox_rect"]
        x1, y1, x2, y2 = br.left(), br.top(), br.right(), br.bottom()
        H, W = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None, None
        return frame_bgr[y1:y2, x1:x2], br

    def analyze_face(self, frame_bgr, rect=None):
        """整幅图人脸关键点 + EAR/MAR/姿态（优先 face_mesh，其次 Tasks）。"""
        if self._lm_backend is None:
            return None
        if self._lm_backend == "legacy_mesh" and self._legacy_face_mesh is None:
            return None
        if self._lm_backend == "tasks" and self._face_landmarker is None:
            return None
        frame_bgr = self._normalize_bgr(frame_bgr)
        h, w = frame_bgr.shape[:2]
        packed, used_flip = self._try_landmarks_full_image(frame_bgr, w, h)
        if packed is None:
            return None
        lm_px, fm = packed
        if used_flip:
            lm_px = lm_px.copy()
            lm_px[:, 0] = (float(w - 1) - lm_px[:, 0]).astype(np.float32)
            fm = None

        try:
            ear, mar = self.calc_ear_mar_from_landmarks(lm_px)
            if fm is not None and fm.size >= 16:
                reprojectdst, pitch, yaw, roll = self._pose_from_facial_transformation_matrix(
                    fm, frame_bgr.shape[:2]
                )
            else:
                reprojectdst, pitch, yaw, roll = self._head_pose_proxy_from_landmarks(lm_px, frame_bgr.shape[:2])
            bbox_rect = self._bbox_from_landmarks(lm_px, w, h)
        except Exception as e:
            if not self._analyze_face_error_logged:
                self._analyze_face_error_logged = True
                _log.error(
                    "[FatigueDetector] analyze_face: landmarks present but EAR/MAR/pose/bbox calc failed (logged once): %s",
                    e,
                )
            return None
        blink = self.blink_analyzer  # cached values, caller should update() with calibrated threshold

        return {
            "landmarks": lm_px,
            "ear": ear,
            "mar": mar,
            "reprojectdst": reprojectdst,
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "bbox_rect": bbox_rect,
            "perclos": blink.perclos,
            "blink_rate": blink.blink_rate,
            "avg_blink_ms": blink.avg_blink_ms,
            "max_blink_ms": blink.max_blink_ms,
            "is_blinking": blink._is_blinking,
        }

    def detect_fatigue(self, frame, gray, rects, ear_threshold=0.32, mar_threshold=0.55, har_threshold=0, fatigue_threshold=0.4, pitch_threshold=5):
        analysis = self.analyze_face(frame)
        if analysis is None:
            return []
        rect = analysis["bbox_rect"]
        ear = analysis["ear"]
        mar = analysis["mar"]
        pitch = analysis["pitch"]
        yaw = analysis["yaw"]
        roll = analysis["roll"]
        har = pitch

        is_eye_tired = ear < 0.75 * ear_threshold
        is_yawn_tired = mar > 1.6 * mar_threshold
        is_head_tired = abs(har - har_threshold) > pitch_threshold

        return [
            {
                "rect": rect,
                "shape": analysis["landmarks"],
                "ear": ear,
                "mar": mar,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "is_eye_tired": is_eye_tired,
                "is_yawn_tired": is_yawn_tired,
                "is_head_tired": is_head_tired,
                "perclos": analysis.get("perclos", 0.0),
                "blink_rate": analysis.get("blink_rate", 0.0),
                "avg_blink_ms": analysis.get("avg_blink_ms", 0.0),
            }
        ]
