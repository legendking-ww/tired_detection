# 基于深度学习的疲劳驾驶

# tireddetect 项目代码分析与总结

## 1. 项目概述

tireddetect 是一个基于深度学习的疲劳驾驶检测系统，使用计算机视觉技术实时监测驾驶员的疲劳状态，包括闭眼、打哈欠和低头等行为，并在检测到疲劳时发出警告。系统还集成了人脸识别功能，支持用户注册和身份验证。**人脸关键点**采用 **MediaPipe**（优先 `face_mesh`，否则 **Tasks FaceLandmarker**），已**移除 dlib**；**应用入口**为根目录 `main.py`，**PyQt 界面与检测线程**位于 **`src/app/`**，详见下文「第 8 节」。

## 2. 项目结构

```
tireddetect/
├── src/
│   ├── app/            # PyQt 主程序层（线程、主窗口、登录/注册）
│   │   ├── worker_threads.py
│   │   ├── main_window.py
│   │   └── auth_windows.py
│   ├── core/           # 核心功能模块
│   │   ├── fatigue_detection.py  # 疲劳检测核心实现
│   │   └── face_recognition.py   # 人脸识别核心实现
│   ├── models/         # 模型相关代码
│   │   ├── facenet.py            # 人脸识别网络
│   │   └── yolo_face_detect.py   # 人脸检测网络
│   ├── ui/             # 界面相关代码
│   │   └── UI.py                 # 主界面设计
│   └── utils/          # 工具函数
│       ├── utils.py              # 辅助函数
│       └── cv_helpers.py         # 摄像头、中文叠字（OpenCV/PIL）
├── resources/
│   ├── images/         # 图像资源
│   ├── sounds/         # 声音资源
│   └── weights/        # 模型权重文件
├── main.py             # 应用入口（启动登录界面）
└── mrsoft.db           # 数据库文件
```

## 3. 核心模块分析

### 3.1 疲劳检测模块 (fatigue_detection.py)

**功能**：实现驾驶员疲劳状态的检测，包括眼睛状态、嘴巴状态和头部姿态的分析。

**核心实现**：
- **FatigueDetector 类**：疲劳检测的核心类，负责初始化模型和执行检测
- **get_face_feat 方法**：提取人脸特征，用于人脸识别
- **process_frame 方法**：处理视频帧，调整大小和转换为灰度图
- **detect_fatigue 方法**：检测疲劳状态，计算眼睛长宽比、嘴巴长宽比和头部姿态

**技术亮点**：
- 使用 MediaPipe（优先 `solutions.face_mesh`，否则 Tasks `FaceLandmarker`）进行人脸关键点检测
- 使用 YOLO 进行人脸检测
- 使用 Inception-ResNet 进行人脸识别
- 综合分析多个指标判断疲劳状态

### 3.2 人脸识别模块 (face_recognition.py)

**功能**：实现人脸注册和识别功能，支持多用户管理。

**核心实现**：
- **FaceRecognition 类**：人脸识别的核心类，负责人脸特征的提取和比对
- **register_user 方法**：注册新用户
- **register_face 方法**：注册人脸特征
- **recognize_face 方法**：识别人脸并返回姓名
- **get_all_face_features 方法**：获取数据库中的所有人脸特征

**技术亮点**：
- 使用 SQLite 数据库存储用户信息和人脸特征
- 支持多用户的人脸注册和识别
- 使用欧氏距离计算人脸特征相似度
- 实现了线程安全的数据库操作

### 3.3 应用入口与界面层 (`main.py` + `src/app/`)

**功能**：`main.py` 仅负责启动 Qt 应用；业务界面与线程位于 `src/app/`。

**核心实现**：
- **`main.py`**：创建 `QApplication`，显示 `LoginWindow` 并连接注册窗口。
- **`src/app/worker_threads.py`**：**BaseThread**、**Start_Thread**（疲劳检测循环）、**AdjustCamera_Thread**（摄像头调整）。
- **`src/app/main_window.py`**：**MainWindow**，绑定 UI 与上述线程。
- **`src/app/auth_windows.py`**：**LoginWindow**、**RegistrationWindow**、**FaceRegisterWindow**（登录/注册/人脸采集）。
- **`src/utils/cv_helpers.py`**：摄像头多后端打开、中文叠字绘制。

**技术亮点**：
- 使用 PyQt5 实现图形界面
- 多线程设计，避免 UI 卡顿
- 支持摄像头和视频文件输入
- 实现了中文显示功能
- 提供了友好的用户界面和操作流程

### 3.4 UI 界面模块 (UI.py)

**功能**：设计应用程序的用户界面。

**核心实现**：
- **Ui_MainWindow 类**：主界面设计类，定义了所有 UI 控件
- **setupUi 方法**：设置界面布局和控件
- **retranslateUi 方法**：设置界面文本和标签

**技术亮点**：
- 使用现代的布局设计
- 美观的界面风格
- 清晰的控件组织
- 响应式的界面布局

### 3.5 工具函数模块 (utils.py)

**功能**：提供各种辅助函数，支持核心功能的实现。

**核心实现**：
- **get_head_pose 函数**：计算头部姿态
- **eye_aspect_ratio 函数**：计算眼睛长宽比
- **mouth_aspect_ratio 函数**：计算嘴巴长宽比
- **line_pairs 变量**：定义头部姿态检测的线条对
- **面部关键点索引**：定义眼睛、嘴巴等面部区域的关键点索引

**技术亮点**：
- 提供了多种实用的工具函数
- 优化了计算效率
- 支持多种面部特征的提取和分析

## 4. 技术栈

| 技术/库 | 用途 |
|--------|------|
| Python | 主要编程语言 |
| PyQt5 | GUI界面开发 |
| OpenCV | 图像处理和视频捕获 |
| MediaPipe | 人脸关键点检测（face_mesh / FaceLandmarker） |
| imutils | 图像处理辅助工具 |
| NumPy | 数值计算 |
| PyTorch | 深度学习框架 |
| SciPy | 科学计算 |
| SQLite | 数据库存储 |
| Pygame | 声音播放 |
| PIL | 图像处理，支持中文显示 |

## 5. 核心功能实现

### 5.1 疲劳检测流程

1. **视频捕获**：从摄像头或视频文件获取视频帧
2. **人脸检测**：使用 YOLO 模型检测人脸
3. **关键点检测**：使用 MediaPipe 检测人脸关键点（与摄像头分辨率一致处理）
4. **特征计算**：计算眼睛长宽比、嘴巴长宽比和头部姿态
5. **疲劳判断**：综合分析特征，判断是否疲劳
6. **警告机制**：当检测到疲劳时，发出声音警告

### 5.2 人脸识别流程

1. **人脸注册**：
   - 捕获人脸图像
   - 提取人脸特征
   - 存储到数据库

2. **人脸识别**：
   - 捕获人脸图像
   - 提取人脸特征
   - 与数据库中的特征比对
   - 返回识别结果
<img width="1116" height="779" alt="image" src="https://github.com/user-attachments/assets/f7895213-2886-4e88-92ff-1cb22a1c6934" />

<img width="1341" height="1190" alt="image" src="https://github.com/user-attachments/assets/b6fa0e5d-b8e2-44ad-a274-09a99470295b" />



### 5.3 登录流程

1. **账号密码登录**：
   - 输入用户名和密码
   - 验证用户信息
   - 登录成功后进入主界面

2. **人脸识别登录**：
   - 启动摄像头
   - 捕获人脸图像
   - 识别人脸
   - 登录成功后进入主界面

## 6. 代码优化与改进

### 6.1 性能优化

- **图像处理优化**：减小图像处理尺寸，提高处理速度
- **人脸检测优化**：使用 YOLO 模型，提高检测速度和准确性
- **线程管理优化**：使用多线程，避免 UI 卡顿
- **数据库操作优化**：实现线程安全的数据库操作

### 6.2 功能改进

- **中文显示支持**：使用 PIL 库实现中文显示
- **多用户支持**：支持多个用户的注册和识别
- **同框多人识别**：支持同时识别画面中的多个人脸
- **用户友好界面**：提供直观的可视化界面

### 6.3 可靠性改进

- **异常处理**：完善异常处理，提高系统稳定性
- **错误提示**：提供清晰的错误信息和提示
- **资源管理**：确保资源的正确释放
- **路径处理**：实现灵活的路径处理，提高兼容性

## 7. 应用场景

1. **长途驾驶**：监测驾驶员的疲劳状态，及时发出警告，减少交通事故
2. **车队管理**：对车队驾驶员进行疲劳监测，提高车队安全管理水平
3. **公共交通**：监测公交车、出租车等公共交通工具驾驶员的状态，保障乘客安全
4. **工业生产**：监测需要长时间集中注意力的工业岗位工作人员的状态，提高生产安全

## 8. 技术挑战、架构要点与排障（维护/验收用）

本节汇总**数据流、近期工程化更新、常见问题与对策**，便于部署与二次开发（原独立技术报告已并入本文档）。

### 8.1 数据流与子系统

| 层级 | 说明 |
|------|------|
| **输入** | USB 摄像头或视频文件；`Start_Thread` 读帧后经 `FatigueDetector.process_frame` 做尺寸归一与灰度辅助（关键点仍用 **BGR 彩图** 推理）。 |
| **人脸定位** | 可选 **YOLO**（`resources/weights/yolo_face.onnx`）；无 YOLO 时整图视为单 ROI，由关键点外接框支撑登录/注册裁剪。 |
| **关键点** | **路径 A**：若环境存在 `mp.solutions.face_mesh`，使用 **FaceMesh + process(RGB)**。**路径 B**：否则 **Tasks FaceLandmarker**，模型 **`face_landmarker.task`**（`models/` 或 `resources/models/`），**`IMAGE` + `detect()`**（与官方单帧一致，避免 `VIDEO` 跟踪偶发空帧）。 |
| **几何与疲劳** | 478 点归一化坐标映射像素；EAR、MAR；优先 **4×4 面部变换矩阵** 求姿态，缺失时几何代理；OpenCV 4.12 下已避免 **`cv2.hconcat` 混 dtype** 导致的断言失败（`np.hstack` + 统一 `float64`）。 |
| **输出** | PyQt5 主界面与日志；可选声音告警；人脸识别与 SQLite 联动。 |

### 8.2 近期主要变更（工程化）

- **MediaPipe**：优先 `face_mesh`，否则 Tasks；Tasks 路径含整图增强/翻转重试，ROI 短边过小时**内部放大再检测**并映射回原图；置信度适度放宽以利弱光召回。  
- **移除 dlib**：不再使用 `shape_predictor_68_face_landmarks.dat` 等；关键点与裁剪由 MediaPipe 承担。  
- **摄像头（Windows）**：`src/utils/cv_helpers.open_video_capture_by_index` — **MSMF 优先**，再 DSHOW，且 **`read()` 成功** 才算可用。  
- **中文叠字**：登录/注册预览用 **`draw_text_cn_on_bgr`（PIL）**，避免 `cv2.putText` 中文变问号。  
- **工程结构**：`main.py` 仅入口；`src/app/` 承载线程与窗口；`src/utils/cv_helpers.py` 承载摄像头与叠字。  
- **代码注释**：`fatigue_detection.py` 顶部说明 **468/478、连接表、IMAGE/detect** 等，避免与旧版 `FACEMESH_*` 混用。

### 8.3 典型问题 — 根因 — 对策

| 现象 | 根因 | 对策 |
|------|------|------|
| 主程序长期「无人脸」、控制台无其它报错 | 曾出现：姿态矩阵路径 **`cv2.hconcat` 混 float32/float64**（OpenCV 4.12），`analyze_face` 吞异常 | 已改为 **`np.hstack`** 且 **R、t 统一 float64** |
| Tasks 在别处能检出、主程序难检出 | 低分辨率下人脸像素过少 | Tasks 分支 **短边 &lt; 480 时放大重试** + 调低 `min_face_*` |
| `VIDEO` + `detect_for_video` 日志里 478 与空帧交替 | 跟踪态偶发无输出 | **主业务固定 IMAGE + `detect`** |
| DSHOW 告警多 | 部分 index 下 DSHOW 不可用 | **MSMF 优先**与多后端回退 |
| 登录/注册预览姓名乱码 | OpenCV 矢量字模不支持中文 | **PIL + 系统字体**（可放 `resources/fonts/`） |

### 8.4 环境与依赖要点

- **Python**：建议 3.9–3.11（见 `requirements.txt`）。  
- **MediaPipe**：无 `mp.solutions` 时走 Tasks；**`face_landmarker.task`** 须有效（过小会判损坏）。  
- **PyTorch / FaceNet**：人脸识别特征；未安装时识别能力受限，与疲劳关键点链路独立。

### 8.5 建议验证步骤

1. 启动主程序，**开始检测**，观察 EAR/MAR、人脸框与日志。  
2. **人脸注册**：中文姓名与预览叠字正常。  
3. **人脸登录**：绿框与「识别中…」等中文正常。  
4. 对照官方行为可参考 [Face Landmarker Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) 的 **IMAGE + `detect`** 说明。

### 8.6 遗留与可选增强

- **MediaPipe 大版本升级**：建议升级后对主程序做一次完整烟测。  
- **`cannot schedule new futures after shutdown`**：多与退出顺序有关；必要时先停子线程、释放摄像头与 landmarker，再关闭 Qt。  
- **可选**：预览仅 UI 镜像翻转、结构化日志等按产品排期。

### 8.7 代码索引

| 内容 | 路径 |
|------|------|
| 关键点双后端与注释 | `src/core/fatigue_detection.py` |
| 应用入口 | `main.py` |
| 疲劳/调参线程 | `src/app/worker_threads.py` |
| 主检测窗口 | `src/app/main_window.py` |
| 登录/注册/人脸采集 | `src/app/auth_windows.py` |
| 摄像头与中文叠字 | `src/utils/cv_helpers.py` |

### 8.8 其它历史问题与方案

1. **中文显示（通用）**：OpenCV `putText` 不支持中文 → 使用 PIL 在 BGR 与 RGB 间转换绘制（与 8.3 中登录/注册方案一致）。  
2. **人脸识别准确性**：受光线、姿态影响 → 深度学习特征 + 阈值与采集质量优化。  
3. **实时性**：多线程 + 合理缩小处理分辨率。  
4. **多用户**：SQLite 存储用户与人脸特征。

## 9. 项目亮点

1. **多模态疲劳检测**：综合考虑眼睛、嘴巴和头部姿态三个维度的信息，提高检测准确性
2. **自适应阈值**：通过前100帧的数据分析，自动计算适合当前用户的疲劳检测阈值
3. **深度学习应用**：使用 YOLO 和 Inception-ResNet 等深度学习模型，提高人脸检测和识别的准确性
4. **用户友好界面**：提供直观的可视化界面，操作简单方便
5. **模块化设计**：代码结构清晰，模块化程度高，便于维护和扩展
6. **多用户支持**：支持多个用户的注册和识别，适合团队使用
7. **中文显示**：支持中文姓名的显示，提高用户体验
8. **安全性**：实现了账号密码和人脸识别双重认证，提高系统安全性

## 10. 总结

tireddetect 是一个功能完善、技术先进的疲劳驾驶检测系统，通过实时监测驾驶员的疲劳状态，为道路交通安全提供了有效的保障。系统集成了人脸识别功能，支持多用户管理，具有良好的用户体验和可靠性。

项目使用了多种先进的计算机视觉和深度学习技术，包括 YOLO 人脸检测（可选）、MediaPipe 关键点、Inception-ResNet 人脸识别等，实现了高精度的疲劳检测和人脸识别功能。

系统的代码结构清晰，模块化程度高，便于维护和扩展。通过不断优化和改进，系统的性能和可靠性得到了显著提高，能够满足实际应用场景的需求。

tireddetect 项目展示了如何将计算机视觉和深度学习技术应用于实际安全监测场景，为驾驶员疲劳检测提供了一种有效的解决方案。通过进一步优化模型和算法，可以提高检测的准确性和实时性，为道路交通安全做出更大的贡献。
