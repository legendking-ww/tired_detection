# 基于深度学习的多模态疲劳驾驶检测（tireddetect）

## 1. 项目概述

tireddetect 是一个基于深度学习的疲劳驾驶检测系统，使用计算机视觉技术实时监测驾驶员的疲劳状态，包括闭眼、打哈欠和低头等行为，并在检测到疲劳时发出警告。系统还集成了人脸识别功能，支持用户注册和身份验证。**人脸关键点**采用 **MediaPipe**（优先 `face_mesh`，否则 **Tasks FaceLandmarker**），已**移除 dlib**；**应用入口**为根目录 `main.py`，**PyQt 界面与检测线程**位于 **`src/app/`**，详见下文「第 8 节」。

### 1.1 技术报告摘要（亮点速览）

| 方向 | 要点 |
|------|------|
| **多模态融合** | 可选 **视觉 + 语音**（Whisper 转写 + LLM 0～1 疲劳分）；默认 **加权视觉分**（闭眼/哈欠/低头不同权重）+ **EMA 平滑** + **动态语音权重**（视觉低且语音高时提高语音占比）；`TIRED_FATIGUE_LOGIC=legacy` 可切回旧版 max 融合。 |
| **预警与统计** | 融合分三级 `normal` / `watch` / `danger`；统计窗口为 **帧数**（`TIRED_STATS_PERIOD_FRAMES`）；危险级强弹窗/音乐带 **冷却**；多模态下危险可要求 **连续多窗** 才累加（`TIRED_DANGER_STREAK_WINDOWS`）。 |
| **身份与性能** | 检测过程中 **姓名识别**默认仅在前 N 秒抽样 + 多数表决后锁定（`TIRED_FACE_NAME_*`），避免每帧比对；人脸登录/注册用 **QTimer** + 识别/写库节流，主界面不卡。 |
| **界面与体验** | 主窗口加宽、**语音识别流水**区、登录/注册 **毛玻璃卡片**与高对比输入框；主预览 **复用 QGraphicsScene**、可选 **`TIRED_PREVIEW_MAX_HEIGHT`** 降采样，显著减轻卡顿。 |
| **工程与依赖** | 摄像头枚举 **静默日志 + 轻探测**（`TIRED_CAMERA_PROBE_MAX`）；EAR 距离计算 **仅用 NumPy**，**已移除 scipy**；`pygame` 仅用于疲劳 **mp3** 提示音。 |
| **排障工具** | 根目录 **`verify_env.py`** 为可选脚本：检查 `.env` 是否加载、密钥是否「像有值」（不打印完整 Secret），**不参与主程序启动**。 |

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
├── verify_env.py       # 可选：检查 .env 是否加载（手工运行，主程序不依赖）
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
- 提供了多种实用的工具函数（EAR/MAR、头部姿态、绘制用 `line_pairs` 等）
- **EAR 欧氏距离仅用 NumPy**，无 scipy 依赖
- 支持多种面部特征的提取和分析

## 4. 技术栈

| 技术/库 | 用途 |
|--------|------|
| Python | 主要编程语言 |
| PyQt5 | GUI界面开发 |
| OpenCV | 图像处理和视频捕获 |
| MediaPipe | 人脸关键点检测（face_mesh / FaceLandmarker） |
| imutils | 图像处理辅助工具 |
| NumPy | 数值计算；EAR 等欧氏距离均在 `utils.py` 内用 numpy 实现 |
| PyTorch | 深度学习框架（FaceNet 等人脸特征，可选） |
| SQLite | 数据库存储 |
| Pygame | 疲劳强告警时播放 `resources/sounds/warning.mp3` |
| PIL | 图像处理，支持中文叠字 |
| requests / python-dotenv | 多模态 HTTP 与 `.env` 加载 |

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

- **检测线程**：疲劳循环在 **`QThread`** 中运行，不阻塞 Qt 主事件循环。
- **主界面预览**：`MainWindow.show_Image` **复用** `QGraphicsScene` / `QGraphicsPixmapItem`，仅更新 `QPixmap`，避免每帧重建场景与全量 `fitInView`；窗口缩放时在 `resizeEvent` 中再适配。
- **预览分辨率**：环境变量 **`TIRED_PREVIEW_MAX_HEIGHT`**（默认 `900`，`0` 关闭）对大图降采样后再上屏，减轻 CPU/GPU 与 Qt 绘制压力。
- **摄像头枚举**：`init_camera_list` 使用 **`verify_frame=False` + `grab()`** 轻探测，并对 OpenCV 日志 **静默**，减少无效 index 导致的卡顿与刷屏（`TIRED_CAMERA_PROBE_MAX` 控制探测个数）。
- **姓名识别**：检测开始后仅在 **探测窗口**内按帧抽样做人脸比对，表决锁定后 **不再调用** `recognize_face`（见 `TIRED_FACE_NAME_*`）。
- **依赖瘦身**：已移除 **scipy**，EAR 相关欧氏距离统一为 **NumPy** 实现。

### 6.2 功能改进

- **多模态疲劳**：视觉统计窗 + 语音周期分析；融合策略可 **`weighted`（默认）** 与 **`legacy`** 切换（见第 8.9 节与 `TIRED_FATIGUE_LOGIC`）。
- **中文显示支持**：使用 PIL 库实现中文显示（登录/注册预览、主画面叠字）。
- **多用户支持**：支持多个用户的注册和识别。
- **登录/注册 UI**：毛玻璃半透明卡片、高对比输入框；人脸登录/人脸注册采用 **QTimer** 驱动，识别与写库 **节流**。
- **主界面**：加宽布局、多模态状态区、**语音识别流水**时间线、系统日志分栏；左侧无整栏滚动条，减少视觉干扰。

### 6.3 可靠性改进

- **异常处理**：完善异常处理，提高系统稳定性
- **错误提示**：提供清晰的错误信息和提示
- **资源管理**：确保资源的正确释放（如登录窗关闭时释放摄像头与定时器）
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
- **摄像头（Windows）**：`src/utils/cv_helpers.open_video_capture_by_index` — **MSMF 优先**，再 DSHOW；枚举时 **`grab()`** 轻量验证，**探测期静默 OpenCV 日志**，主线程枚举用 **`verify_frame=False`** 降低阻塞。  
- **中文叠字**：登录/注册预览用 **`draw_text_cn_on_bgr`（PIL）**，避免 `cv2.putText` 中文变问号。  
- **工程结构**：`main.py` 仅入口；`src/app/` 承载线程与窗口；`src/utils/cv_helpers.py` 承载摄像头与叠字；**`src/multimodal/`** 承载语音与融合。  
- **代码注释**：`fatigue_detection.py` 顶部说明 **468/478、连接表、IMAGE/detect** 等，避免与旧版 `FACEMESH_*` 混用。  
- **多模态融合 2.0（默认 `weighted`）**：视觉分由 **闭眼/哈欠/低头** 按饱和比例加权（非简单 max）+ **EMA 平滑**；与语音融合时若「视觉低且语音高」则 **提高语音权重**；等级阈值默认 **注意 0.30 / 危险 0.65**（均可用环境变量覆盖）。  
- **主界面流畅度**：预览 **Scene 复用**、**按需 fitInView**、**GraphicsView 优化标志**；可选预览限高 **`TIRED_PREVIEW_MAX_HEIGHT`**。  
- **认证界面**：登录/注册 **半透明玻璃卡片**、高对比输入框；人脸登录与 **人脸注册** 均为 **QTimer** 单帧 tick，避免主线程 `while` 卡死。  
- **依赖**：已去除 **scipy**（EAR 欧氏距离改 NumPy）；**`verify_env.py`** 为可选环境自检脚本。

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
| 多模态融合、Groq、麦克风录制 | `src/multimodal/`（含 `mic_qt.py`） |
| 主检测窗口 | `src/app/main_window.py` |
| 登录/注册/人脸采集 | `src/app/auth_windows.py` |
| 摄像头与中文叠字 | `src/utils/cv_helpers.py` |

### 8.8 其它历史问题与方案

1. **中文显示（通用）**：OpenCV `putText` 不支持中文 → 使用 PIL 在 BGR 与 RGB 间转换绘制（与 8.3 中登录/注册方案一致）。  
2. **人脸识别准确性**：受光线、姿态影响 → 深度学习特征 + 阈值与采集质量优化。  
3. **实时性**：多线程 + 合理缩小处理分辨率。  
4. **多用户**：SQLite 存储用户与人脸特征。

### 8.9 多模态（视觉 + 语音，可选 Groq）

在环境变量中开启后，后台线程按 `TIRED_MULTIMODAL_INTERVAL` 周期工作：**文件模式**下读取固定 WAV；**麦克风模式**（`TIRED_MULTIMODAL_MIC=1`）下用 PyQt5 `QAudioInput` 在独立 `QThread` 内录制 `TIRED_MULTIMODAL_RECORD_SEC` 秒，再调用 **Groq**（或 `GROQ_API_BASE` 指向的兼容网关）Whisper 转写 + 聊天模型得到 **0～1 语音疲劳分**。API 多次失败时语音侧记为 **-1**，融合时**仅使用视觉分**。主检测线程在 **`TIRED_STATS_PERIOD_FRAMES` 帧**（默认约 5s@30fps，非固定 60 帧）统计窗口末计算视觉分，与最近语音分融合，叠字 `fused` 与等级，并按 `danger` / `watch` 参与预警。

**融合策略（`TIRED_FATIGUE_LOGIC`，默认 `weighted`）**

| 模式 | 说明 |
|------|------|
| **`weighted`（推荐）** | 闭眼/哈欠/低头按 **`TIRED_VISUAL_SAT_*`** 饱和比例得到分量，再按 **`TIRED_VISUAL_W_*`** 加权求和（非 max）；结果经 **`TIRED_VISUAL_SMOOTH` EMA**；与语音 **`fuse_visual_audio_dynamic`**（视觉低且语音高时提高语音权重）。 |
| **`legacy`** | 原 **max(三通道)** + 固定 **`TIRED_MULTIMODAL_W_VISUAL` / `W_AUDIO`** 加权，便于与旧数据或论文对比。 |

**默认等级阈值**（均可被 `.env` 覆盖）：**注意** `TIRED_ALERT_WATCH=0.30`，**危险** `TIRED_ALERT_DANGER=0.65`。日志中文案在 `weighted` 下与融合分档对齐（轻微 / 中度 / 极度疲劳等节流提示）。

**其它常用变量（融合与视觉）**：`TIRED_FUSION_BOOST_VISUAL_LT`、`TIRED_FUSION_BOOST_AUDIO_GT`、`TIRED_FUSION_W_AUDIO_BOOST`、`TIRED_DANGER_STREAK_WINDOWS`、`TIRED_VISUAL_SAT_EYE` / `YAWN` / `HEAD`、`TIRED_VISUAL_W_EYE` / `YAWN` / `HEAD`、`TIRED_VISUAL_SMOOTH`。

| 变量 | 说明 |
|------|------|
| `TIRED_MULTIMODAL` | 设为 `1` / `true` 开启 |
| `TIRED_MULTIMODAL_MIC` | 设为 `1` 使用麦克风；不设则读 WAV 文件（向后兼容） |
| `TIRED_MULTIMODAL_VIDEO_AUDIO` | 设为 `1` 且正在**播放视频文件**时，用 **ffmpeg** 从视频当前进度附近抽音轨做语音分析（需安装 ffmpeg）；优先级高于麦克风 |
| `TIRED_MULTIMODAL_RECORD_SEC` | 单次麦克风录制秒数，默认 `10` |
| `SILICONFLOW_API_KEY` | 硅基流动等：与下两项任填其一（见 `groq_api_key()` 读取顺序） |
| `MULTIMODAL_API_KEY` | 任意兼容厂商 Bearer Token |
| `GROQ_API_KEY` | 名称保留以兼容旧说明（[Groq](https://console.groq.com/) 等） |
| `GROQ_API_BASE` | OpenAI 兼容 API 根 URL，默认 `https://api.groq.com/openai/v1`（可填代理或 SiliconFlow 等） |
| `GROQ_WHISPER_MODEL` / `GROQ_CHAT_MODEL` | 转写 / 聊天模型名，默认 `whisper-large-v3`、`llama-3.1-8b-instant` |
| `TIRED_MULTIMODAL_WAV` | 文件模式下的 wav；不设则尝试 `resources/samples/driver_demo.wav` |
| `TIRED_MULTIMODAL_INTERVAL` | 周期间隔（秒），默认 `10`；麦克风模式下实际间隔 ≈「录制 + API」后补睡到该值 |
| `TIRED_MULTIMODAL_W_VISUAL` / `TIRED_MULTIMODAL_W_AUDIO` | 融合权重，默认 `0.7` / `0.3` |
| `TIRED_STATS_PERIOD_FRAMES` | 视觉疲劳统计窗口长度（**帧数**，非秒），默认 `150`（约 5s@30fps）；过小会导致日志/蜂鸣很密，可调 `180`–`300` |
| `TIRED_STRONG_ALERT_COOLDOWN_SEC` | **危险级**弹窗+音乐的冷却（秒），默认 `45`；冷却内仅打日志，避免连续模态框轰炸 |
| `TIRED_ALERT_WATCH` / `TIRED_ALERT_DANGER` | 多模态融合分阈值：注意级 / 危险级，默认 **`0.30` / `0.65`**（可按答辩演示调敏钝） |
| `TIRED_ALERT_WATCH_MID` | 注意档内区分「轻微 / 中度」文案用，默认 `0.45` |
| `TIRED_FATIGUE_LOGIC` | `weighted`（默认）或 `legacy` / `max` / `old` / `0` |
| `TIRED_FACE_NAME_PROBE_SEC` | 检测中姓名识别：默认前 **12s** 抽样表决后锁定；`0` 关闭叠字；`-1` 每帧比对（重） |
| `TIRED_FACE_NAME_SAMPLE_FRAMES` | 探测期内每隔多少帧做一次比对，默认 `4` |
| `TIRED_FACE_LOGIN_RECOGNIZE_SEC` | 人脸登录节流识别间隔（秒），默认 `0.4` |
| `TIRED_FACE_REGISTER_DB_SEC` | 人脸注册写库节流（秒），默认 `0.5` |
| `TIRED_FACE_REGISTER_TIMEOUT_SEC` | 人脸注册最长尝试（秒），默认 `180` |
| `TIRED_CAMERA_PROBE_MAX` | 主窗口枚举摄像头最大 index 个数，默认 `3` |
| `TIRED_PREVIEW_MAX_HEIGHT` | 主界面预览最大高度（像素），超过则缩小再上屏；默认 `900`，`0` 关闭 |
| `TIRED_MIC_MIN_RMS` | 麦克风 RMS 参考下限（16bit），默认 `280`；低于此仅提示「音量偏低」，**仍会转写** |
| `TIRED_MIC_RMS_SILENT_ABORT` | RMS 低于此值视为无效静音、本段不上传，默认 `12`；环境安静仍被误杀可调为 `6`–`8` |

**分级预警（多模态 fused）**：`normal`（绿字）仅画面提示；`watch`（橙字）节流日志 + 短促高频蜂鸣（与轻视觉告警共用节流）；`danger`（红字）参与累加并尽快触发 **危险级**弹窗与 `warning.mp3`。强弹窗/音乐受 `TIRED_STRONG_ALERT_COOLDOWN_SEC` 限制，冷却内只发文字提醒。

国内网络可配置 `GROQ_API_BASE` 指向可达的兼容网关；HTTP 对 429/5xx 与网络错误有有限次退避重试（见 `src/multimodal/groq_audio.py`）。

**安全**：密钥只放在本机环境变量或已被 `.gitignore` 忽略的 `.env` 中，**不要**提交到 Git、不要发到聊天/论坛；若已泄露，请立刻在厂商控制台**作废并换新 Key**。

**方法：`.env` 文件（推荐）**：安装依赖后在项目根目录创建 `.env` 并填写密钥；`main.py` 在启动时会执行 `load_dotenv()` 加载（需 `pip install python-dotenv`，已写入 `requirements.txt`）。`.env` 已被 `.gitignore` 忽略。可选运行 **`python verify_env.py`** 检查 `.env` 是否被正确加载（不打印完整密钥）。

**SiliconFlow + Qwen 示例**（与 Groq 二选一；密钥用占位符，请换成本地变量）：

```powershell
$env:TIRED_MULTIMODAL="1"
$env:GROQ_API_BASE="https://api.siliconflow.com/v1"
$env:MULTIMODAL_API_KEY="你的_sk_密钥"
$env:GROQ_WHISPER_MODEL="FunAudioLLM/SenseVoiceSmall"
$env:GROQ_CHAT_MODEL="Qwen/Qwen3-8B"
```

若国内解析 `api.siliconflow.com` 较慢，可尝试控制台文档中的 `https://api.siliconflow.cn/v1`。转写请求对 **Groq** 会自动带 `response_format=json`，对 **其它 Base** 则不带，以兼容 SiliconFlow 等网关。

## 9. 项目亮点

1. **多模态疲劳（可选）**：视觉统计与 **Whisper + LLM 语音疲劳分** 融合；默认 **加权视觉 + EMA 平滑 + 动态语音权重**，缓解「仅低头拉高 max」「语音拉不动」等问题；**`legacy` 一键回退**便于对比实验。  
2. **自适应视觉阈值**：前 **100 帧**标定 EAR/MAR/俯仰，得到个人化阈值，再进入滑窗比例统计。  
3. **分级与节流预警**：融合 **`normal` / `watch` / `danger`**；强弹窗/音乐 **冷却**；可选 **连续多窗危险** 才叠加强告警，减少偶发抖动误报。  
4. **身份识别省算力**：检测全程 **不再每帧做人脸比对**；默认前若干秒 **抽样 + 多数表决** 锁定姓名（可关可恢复旧行为）。  
5. **界面与信息架构**：主窗口加宽、**语音流水时间线**、多模态状态与系统日志分区；登录/注册 **毛玻璃半透明** + 高对比表单；人脸登录/注册 **异步帧驱动**，避免界面假死。  
6. **运行流畅度**：预览 **Scene 复用**、按需 **`fitInView`**、可选 **预览限高**；摄像头枚举 **轻量 + 静默日志**；**移除 scipy**，依赖更轻。  
7. **深度学习与关键点**：可选 **YOLO** 人脸框；**MediaPipe** `face_mesh` 或 **Tasks FaceLandmarker**；**FaceNet** 特征与 SQLite 多用户管理。  
8. **中文与工程化**：PIL 中文叠字；`.env` + **utf-8-sig** 防 BOM；**`verify_env.py`** 可选自检；`src/multimodal` 与 **ffmpeg 视频伴音** 可扩展。

## 10. 总结

tireddetect 是一个功能完善、技术先进的疲劳驾驶检测系统，通过实时监测驾驶员的疲劳状态，为道路交通安全提供了有效的保障。系统集成了人脸识别功能，支持多用户管理，具有良好的用户体验和可靠性。

项目使用了多种先进的计算机视觉和深度学习技术，包括 YOLO 人脸检测（可选）、MediaPipe 关键点、Inception-ResNet 人脸识别等，实现了高精度的疲劳检测和人脸识别功能。

系统的代码结构清晰，模块化程度高，便于维护和扩展。通过不断优化和改进，系统的性能和可靠性得到了显著提高，能够满足实际应用场景的需求。

tireddetect 项目展示了如何将计算机视觉和深度学习技术应用于实际安全监测场景，为驾驶员疲劳检测提供了一种有效的解决方案。通过进一步优化模型和算法，可以提高检测的准确性和实时性，为道路交通安全做出更大的贡献。
