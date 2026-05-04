import math
import cv2
import numpy as np

try:
    from imutils import face_utils
except Exception:
    face_utils = None

# 世界坐标系(UVW)：填写3D参考点
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 定义相机内参，用于相机坐标系（XYZ）。
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]

# 定义相机畸变参数，用于图像中心坐标系（uv）。
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系
#将相机内参转换为相机矩阵
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
#将相机畸变参数转换为畸变系数。
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 定义重新投影的3D点的世界坐标轴，用于验证结果的姿态
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# 定义绘正方体的12条轴，用于视化头部姿态
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

if face_utils is not None:
    # 获取左眼关键点的起始索引和结束索引。
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    # 获取右眼关键点的起始索引和结束索引。
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    # 获取嘴部关键点的起始索引和结束索引。
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
else:
    lStart = lEnd = rStart = rEnd = mStart = mEnd = None

#实现头部姿态估计算法，人脸的关键点信息作为输入
def get_head_pose(shape):
    # （像素坐标集合）填写2D参考点
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return reprojectdst, euler_angle  # 投影误差，欧拉角

def _euclidean_2d(p, q) -> float:
    """两点欧氏距离（仅用 numpy，避免依赖 scipy）。"""
    return float(np.linalg.norm(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)))


#用于计算眼睛的长宽比
def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = _euclidean_2d(eye[1], eye[5])
    B = _euclidean_2d(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = _euclidean_2d(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear

#用于计算嘴巴的长宽比
def mouth_aspect_ratio(mouth):  # 嘴部
    #np.linalg.norm()用于求范数，默认为二范数（欧氏距离的平方）
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 58
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 56
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


def get_head_pose_six_points(image_pts, frame_size):
    """
    使用6个2D点（鼻尖、下巴、左右眼角、左右嘴角）估计头部姿态（solvePnP）。
    image_pts: np.ndarray shape=(6,2) 像素坐标
    frame_size: (h, w)
    返回: reprojectdst(8x2), euler_angle(3x1) [pitch,yaw,roll] (degrees)
    """
    h, w = frame_size
    image_pts = np.asarray(image_pts, dtype=np.float32).reshape(-1, 2)
    if image_pts.shape[0] != 6:
        raise ValueError("image_pts 必须是6个点")

    # 通用3D人脸模型点（单位：任意，保持相对比例即可）
    model_pts = np.float32([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0),     # Right mouth corner
    ])

    # 相机内参（近似）：焦距取图像宽度
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    cam_matrix_local = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1.0],
    ], dtype=np.float32)
    dist_coeffs_local = np.zeros((4, 1), dtype=np.float32)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_pts, image_pts, cam_matrix_local, dist_coeffs_local, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP 失败")

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix_local, dist_coeffs_local)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle