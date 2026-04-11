
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
from ..utils.utils import get_head_pose, eye_aspect_ratio, mouth_aspect_ratio, line_pairs, lStart, lEnd, rStart, rEnd, mStart, mEnd
from ..models.facenet import InceptionResnetV1
from ..models.yolo_face_detect import YOLO_face
import torch
import json

class FatigueDetector:
    def __init__(self):
        self.detector = None
        self.predictor = None
        self.face_net = None
        self.yolo_face = None
        self.face_utils = face_utils
        self.line_pairs = line_pairs
        self.get_head_pose = get_head_pose
        self.eye_aspect_ratio = eye_aspect_ratio
        self.mouth_aspect_ratio = mouth_aspect_ratio
        self.lStart = lStart
        self.lEnd = lEnd
        self.rStart = rStart
        self.rEnd = rEnd
        self.mStart = mStart
        self.mEnd = mEnd
        self.load_models()
        
    def load_models(self):
        try:
            # 加载人脸检测器和关键点预测器
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            
            # 加载人脸识别模型
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.face_net = InceptionResnetV1().to(self.device)
            self.face_net.load_state_dict(torch.load('resources/weights/facenet_best_server.pt', map_location='cpu'))
            self.face_net.eval()
            
            # 加载人脸检测模型
            self.yolo_face = YOLO_face('resources/weights/yolo_face.onnx')
            
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
    
    def get_face_feat(self, face_img):
        try:
            import numpy as np
            face_img = cv2.resize(face_img, dsize=(112, 112))
            face_img = (face_img - 127.5) / 127.5
            face_img = np.transpose(face_img, (2, 0, 1))
            face_img = np.expand_dims(face_img, axis=0)
            face_img_tensor = torch.Tensor(face_img).to(self.device)
            face_feat_tensor = self.face_net(face_img_tensor)
            face_feat = face_feat_tensor.detach().cpu().numpy()
            return face_feat
        except Exception as e:
            print(f"特征提取失败: {str(e)}")
            return None
    
    def process_frame(self, frame):
        # 调整帧大小
        frame = imutils.resize(frame, width=640)
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 自适应直方图均衡
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # 检测人脸
        rects = self.detector(gray, 0)
        return frame, gray, rects
    
    def detect_fatigue(self, frame, gray, rects, ear_threshold=0.32, mar_threshold=0.55, har_threshold=0, fatigue_threshold=0.4, pitch_threshold=5):
        fatigue_info = []
        
        for rect in rects:
            # 获得脸部特征位置的信息
            shape = self.predictor(gray, rect)
            # 将脸部特征信息转换为数组格式
            shape = self.face_utils.shape_to_np(shape)
            # 提取左眼、右眼坐标、嘴巴坐标
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]
            
            # 计算眼睛和嘴巴的长宽比
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = self.mouth_aspect_ratio(mouth)
            
            # 获取头部姿态
            reprojectdst, euler_angle = self.get_head_pose(shape)
            pitch = euler_angle[0, 0]
            yaw = euler_angle[1, 0]
            roll = euler_angle[2, 0]
            har = pitch
            
            # 疲劳状态判断
            is_eye_tired = ear < 0.75 * ear_threshold
            is_yawn_tired = mar > 1.6 * mar_threshold
            is_head_tired = abs(har - har_threshold) > pitch_threshold
            
            fatigue_info.append({
                "rect": rect,
                "shape": shape,
                "ear": ear,
                "mar": mar,
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
                "is_eye_tired": is_eye_tired,
                "is_yawn_tired": is_yawn_tired,
                "is_head_tired": is_head_tired
            })
        
        return fatigue_info