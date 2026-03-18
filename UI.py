# 修改后的代码

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QFileDialog, QComboBox, QSlider, QStyle, QStyleFactory,
                             QFrame, QSplitter, QGroupBox, QGridLayout, QMessageBox, QProgressBar)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
import qdarkstyle
from ultralytics import YOLO

# 导入原有的人脸检测函数
from yolo_face_detection import download_face_model, load_font, cv2_add_chinese_text, find_chinese_font_path


def is_lfs_pointer_file(file_path):
    """判断文件是否为 Git LFS 指针文件。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.read(128)
        return header.startswith("version https://git-lfs.github.com/spec/v1")
    except (OSError, UnicodeDecodeError):
        return False


def is_classification_model_path(model_path):
    """根据文件名快速判断是否是分类模型权重。"""
    model_name = os.path.basename(model_path).lower()
    return "-cls" in model_name or "_cls" in model_name


def iter_detection_boxes(results):
    """安全遍历检测框，兼容误传入分类模型时的结果。"""
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            yield box


def load_face_detector():
    """优先加载专用人脸检测模型，失败时回退到 OpenCV Haar。"""
    candidates = []
    env_path = os.environ.get("FACE_MODEL_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "yolov12l-face.pt",
        "yolov11n-face.pt",
        download_face_model(),
    ])

    seen = set()
    for model_path in candidates:
        if not model_path or model_path in seen:
            continue
        seen.add(model_path)
        if not os.path.exists(model_path):
            continue
        if is_lfs_pointer_file(model_path):
            print(f"人脸检测模型是 Git LFS 占位文件，跳过: {model_path}")
            continue
        if is_classification_model_path(model_path):
            print(f"人脸检测模型路径指向分类模型，跳过: {model_path}")
            continue
        if "face" not in os.path.basename(model_path).lower():
            print(f"检测到通用目标检测模型而非专用人脸模型，跳过: {model_path}")
            continue
        try:
            print(f"加载人脸检测模型: {model_path}")
            return YOLO(model_path), "yolo"
        except Exception as e:
            print(f"加载人脸检测模型失败 {model_path}: {e}")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("无法加载 OpenCV Haar 人脸检测器")
    print(f"未找到可用的人脸 YOLO 权重，已回退到 Haar 检测器: {cascade_path}")
    return cascade, "haar"


def detect_faces_in_frame(face_model, face_model_type, frame, conf_threshold):
    """统一返回检测到的人脸框坐标。"""
    if face_model is None:
        return []

    if face_model_type == "yolo":
        results = face_model(frame, conf=conf_threshold, verbose=False)
        boxes = []
        for box in iter_detection_boxes(results):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append((x1, y1, x2, y2))
        return boxes

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [(x, y, x + w, y + h) for (x, y, w, h) in faces]


def classify_emotion(emotion_model, face_roi_gray_3ch):
    """获取表情 top1 类别及置信度。"""
    emotion_results = emotion_model(face_roi_gray_3ch, verbose=False)
    probs = emotion_results[0].probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    emotion_name = emotion_results[0].names[class_id]
    emotion_map = {
        "angry": "愤怒",
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "高兴",
        "sad": "悲伤",
        "surprise": "惊讶",
        "neutral": "中性",
    }
    return emotion_map.get(str(emotion_name).lower(), str(emotion_name)), confidence


def configure_application_font(app):
    """为 Qt 应用配置中文字体，避免 Linux 下出现方框。"""
    font_path = find_chinese_font_path()
    preferred_families = [
        "Noto Sans SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "PingFang SC",
    ]

    family = None
    if font_path:
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                family = families[0]

    app_font = QFont()
    if family:
        app_font.setFamily(family)
    else:
        app_font.setFamilies(preferred_families)
    app_font.setPointSize(11)
    app.setFont(app_font)


class VideoThread(QThread):
    """视频处理线程，避免UI卡顿"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, mode='camera', file_path=None):
        super().__init__()
        self.mode = mode
        self.file_path = file_path
        self.running = True
        self.face_model = None
        self.face_model_type = None
        self.emotion_model = None
        self.conf_threshold = 0.5

    def set_models(self, face_model, emotion_model, face_model_type=None):
        self.face_model = face_model
        self.face_model_type = face_model_type
        self.emotion_model = emotion_model

    def set_conf_threshold(self, value):
        self.conf_threshold = value / 100.0

    def run(self):
        # 加载字体
        font = load_font()

        # 初始化视频源
        if self.mode == 'camera':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(self.file_path)

        if not cap.isOpened():
            if self.mode == 'camera':
                self.error_signal.emit(
                    "无法打开摄像头。\n\n"
                    "当前系统里没有检测到可用的视频设备，或当前运行环境无法访问宿主机摄像头。\n"
                    "请确认系统存在 /dev/video0 之类的设备节点，并检查摄像头权限、容器/远程环境映射。"
                )
            else:
                self.error_signal.emit(f"无法打开输入源：\n{self.file_path}")
            return

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.mode == 'video' else 0

        # 图片模式特殊处理
        if self.mode == 'image':
            # 读取单张图片
            ret, frame = cap.read()
            if ret:
                # 人脸检测
                if self.face_model:
                    face_boxes = detect_faces_in_frame(self.face_model, self.face_model_type, frame, self.conf_threshold)

                    # 处理检测结果
                    for x1, y1, x2, y2 in face_boxes:

                        # 扩大边界框
                        frame_height, frame_width = frame.shape[:2]
                        expand_x = int((x2 - x1) * 0.2)
                        expand_y = int((y2 - y1) * 0.2)

                        x1_expanded = max(0, x1 - expand_x)
                        y1_expanded = max(0, y1 - expand_y)
                        x2_expanded = min(frame_width, x2 + expand_x)
                        y2_expanded = min(frame_height, y2 + expand_y)

                        # 绘制美观的人脸框
                        cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                        # 如果有表情识别模型，进行表情识别
                        if self.emotion_model:
                            try:
                                face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                                if face_roi.size == 0:
                                    continue

                                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                                face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                                emotion, confidence = classify_emotion(self.emotion_model, face_roi_gray_3ch)

                                text = f"{emotion}: {confidence:.2f}"

                                if y1_expanded > 40:
                                    text_position = (x1_expanded, y1_expanded - 35)
                                else:
                                    text_position = (x1_expanded, y2_expanded + 55)

                                frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font)
                            except Exception as e:
                                print(f"处理人脸时出错: {e}")

                # 发送处理后的帧
                self.change_pixmap_signal.emit(frame)

            # 图片处理完成后释放资源
            cap.release()
            return

        # 视频和摄像头模式处理
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if self.mode == 'video' and total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                self.progress_signal.emit(progress)

            # 人脸检测
            if self.face_model:
                face_boxes = detect_faces_in_frame(self.face_model, self.face_model_type, frame, self.conf_threshold)

                # 处理检测结果
                for x1, y1, x2, y2 in face_boxes:

                    frame_height, frame_width = frame.shape[:2]
                    expand_x = int((x2 - x1) * 0.2)
                    expand_y = int((y2 - y1) * 0.2)

                    x1_expanded = max(0, x1 - expand_x)
                    y1_expanded = max(0, y1 - expand_y)
                    x2_expanded = min(frame_width, x2 + expand_x)
                    y2_expanded = min(frame_height, y2 + expand_y)

                    cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                    if self.emotion_model:
                        try:
                            face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                            if face_roi.size == 0:
                                continue

                            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                            emotion, confidence = classify_emotion(self.emotion_model, face_roi_gray_3ch)

                            text = f"{emotion}: {confidence:.2f}"

                            if y1_expanded > 40:
                                text_position = (x1_expanded, y1_expanded - 35)
                            else:
                                text_position = (x1_expanded, y2_expanded + 55)

                            frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font)
                        except Exception as e:
                            print(f"处理人脸时出错: {e}")

            # 发送处理后的帧
            self.change_pixmap_signal.emit(frame)

            # 控制帧率
            if self.mode == 'camera':
                cv2.waitKey(1)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


class FaceDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置应用标题和图标
        self.setWindowTitle("Face & Emotion Studio")

        # 设置窗口大小
        self.setMinimumSize(1700, 1100)

        # 定义可用的模型路径和名称
        self.model_paths = {
            "best_yolo26x_cls":"best_yolo26x_cls.pt",
            "best_yolo11n_cls":"best_yolo11n_cls.pt",
            # "综合数据集模型": "runs/classify/datasets_plus_optimized/weights/best.pt",
            # "FER2013增强模型": "runs/classify/fer2013_plus_optimized/weights/best.pt",
            # "AffectNet模型": "runs/classify/affectnet_optimized/weights/best.pt",
            # "我的自定义数据集模型": "runs/classify/my_datasets_optimized/weights/best.pt"
        }

        # 初始化UI
        self.init_ui()

        # 初始化模型
        self.init_models()

        # 初始化视频线程
        self.video_thread = None

    def init_ui(self):
        # 设置中央窗口部件
        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        # 创建主垂直布局
        full_layout = QVBoxLayout()
        full_layout.setContentsMargins(20, 20, 20, 20)
        full_layout.setSpacing(18)
        central_widget.setLayout(full_layout)

        # 创建内容水平布局
        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)

        # 创建左侧控制面板
        control_panel = QFrame()
        control_panel.setObjectName("controlPanel")
        control_panel.setFrameShape(QFrame.StyledPanel)
        control_panel.setMaximumWidth(540)
        control_panel_layout = QVBoxLayout(control_panel)
        control_panel_layout.setContentsMargins(26, 26, 26, 30)
        control_panel_layout.setSpacing(16)

        # 应用标题卡片
        title_card = QFrame()
        title_card.setObjectName("titleCard")
        title_layout = QVBoxLayout(title_card)
        title_layout.setContentsMargins(24, 22, 24, 22)
        title_layout.setSpacing(8)

        title_label = QLabel("Face & Emotion Studio")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("人脸检测与表情识别工作台")
        subtitle_label.setObjectName("subtitleLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(subtitle_label)

        self.model_status_label = QLabel("正在初始化模型...")
        self.model_status_label.setObjectName("modelStatusLabel")
        self.model_status_label.setWordWrap(True)
        self.model_status_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(self.model_status_label)

        control_panel_layout.addWidget(title_card)

        # 添加分割线
        line = QFrame()
        line.setObjectName("dividerLine")
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_panel_layout.addWidget(line)

        # 模式选择组
        mode_group = QGroupBox("检测模式")
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        mode_layout.setSpacing(20)  # 增加控件之间的间距

        # 模式下拉框
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("inputCombo")
        self.mode_combo.addItem("摄像头实时检测")
        self.mode_combo.addItem("图片文件检测")
        self.mode_combo.addItem("视频文件检测")
        self.mode_combo.currentIndexChanged.connect(self.mode_changed)
        self.mode_combo.setMinimumHeight(44)
        mode_layout.addWidget(self.mode_combo)

        # 文件选择按钮
        self.file_button = QPushButton("选择文件")
        self.file_button.setObjectName("secondaryButton")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setEnabled(False)
        self.file_button.setMinimumHeight(44)
        mode_layout.addWidget(self.file_button)

        # 文件路径显示
        self.file_label = QLabel("未选择文件")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)
        self.file_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.file_label.setMinimumHeight(48)
        mode_layout.addWidget(self.file_label)

        mode_group.setLayout(mode_layout)
        control_panel_layout.addWidget(mode_group)

        # 检测参数组
        param_group = QGroupBox("检测参数")
        param_layout = QVBoxLayout()
        param_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        param_layout.setSpacing(20)  # 增加控件之间的间距

        # 添加模型选择下拉框
        model_layout = QVBoxLayout()
        model_label = QLabel("选择表情识别模型:")
        model_label.setObjectName("sectionHintLabel")
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("inputCombo")
        for model_name in self.model_paths.keys():
            self.model_combo.addItem(model_name)
        self.model_combo.currentIndexChanged.connect(self.change_emotion_model)
        self.model_combo.setMinimumHeight(44)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        param_layout.addLayout(model_layout)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        conf_label.setObjectName("sectionHintLabel")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickPosition(QSlider.TicksBelow)
        self.conf_slider.setTickInterval(10)
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setObjectName("confValueLabel")

        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value_label)
        param_layout.addLayout(conf_layout)

        param_group.setLayout(param_layout)
        control_panel_layout.addWidget(param_group)

        # 操作按钮组
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout()
        action_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        action_layout.setSpacing(25)  # 增加控件之间的间距

        # 开始/停止按钮
        self.start_button = QPushButton("开始检测")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self.toggle_detection)
        self.start_button.setMinimumHeight(52)
        action_layout.addWidget(self.start_button)

        # 保存结果按钮
        self.save_button = QPushButton("保存结果")
        self.save_button.setObjectName("ghostButton")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        self.save_button.setMinimumHeight(52)
        action_layout.addWidget(self.save_button)

        action_group.setLayout(action_layout)
        control_panel_layout.addWidget(action_group)

        # 进度条
        self.progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(25, 35, 25, 25)  # 增加内边距
        progress_layout.setSpacing(20)  # 增加控件之间的间距
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_group.setLayout(progress_layout)
        self.progress_group.setVisible(False)
        control_panel_layout.addWidget(self.progress_group)

        # 添加弹性空间，但不要占用太多空间
        control_panel_layout.addStretch(1)

        # 将左侧控制面板添加到内容布局
        content_layout.addWidget(control_panel)

        # 创建右侧显示区域
        display_panel = QFrame()
        display_panel.setObjectName("displayPanel")
        display_panel.setFrameShape(QFrame.StyledPanel)
        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(22, 22, 22, 22)
        display_layout.setSpacing(14)

        preview_title = QLabel("实时预览")
        preview_title.setObjectName("previewTitle")
        display_layout.addWidget(preview_title)

        preview_subtitle = QLabel("检测结果会在这里展示，支持摄像头、图片与视频三种模式。")
        preview_subtitle.setObjectName("previewSubtitle")
        preview_subtitle.setWordWrap(True)
        display_layout.addWidget(preview_subtitle)

        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setObjectName("imageViewport")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(900, 650)
        self.image_label.setText("等待开始检测...")
        display_layout.addWidget(self.image_label)

        # 添加显示区域到内容布局
        content_layout.addWidget(display_panel, 1)

        # 将内容布局添加到主布局
        full_layout.addLayout(content_layout, 1)

        # 设置样式
        self.apply_styles()

    def apply_styles(self):
        # 设置应用整体样式
        self.setStyleSheet("""
            #centralWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F6F0E8, stop:0.45 #E8F0F3, stop:1 #DCE8EC);
            }
            QMainWindow {
                background: transparent;
            }
            QWidget {
                color: #17313A;
            }
            QLabel {
                color: #17313A;
                font-size: 14px;
                background: transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #126782, stop:1 #1D86A3);
                color: #FFFFFF;
                border: 1px solid rgba(18, 103, 130, 0.20);
                padding: 13px 16px;
                border-radius: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #177896, stop:1 #2897B8);
            }
            QPushButton:pressed {
                background: #115B72;
            }
            QPushButton:disabled {
                background: #C5D4DA;
                color: #5F747C;
                border: 1px solid rgba(95, 116, 124, 0.12);
            }
            QComboBox {
                background: rgba(255, 255, 255, 0.92);
                color: #18343D;
                border: 1px solid #BED0D6;
                padding: 8px 14px;
                border-radius: 12px;
                font-size: 15px;
                min-height: 28px;
            }
            QComboBox:on {
                color: #18343D;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QComboBox QAbstractItemView {
                background: #FFFFFF;
                color: #18343D;
                selection-background-color: #DCECF2;
                selection-color: #0F2830;
            }
            QGroupBox {
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid rgba(163, 188, 198, 0.34);
                border-radius: 20px;
                margin-top: 22px;
                padding-top: 32px;
                font-weight: 700;
                font-size: 16px;
                color: #17313A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 6px 14px 6px 14px;
                margin-top: 0px;
                color: #FFFFFF;
                background: #2F7284;
                border: 1px solid rgba(47, 114, 132, 0.18);
                border-radius: 12px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #C8D7DD;
                height: 10px;
                background: #E4EDF1;
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #F4A63A;
                border: 1px solid #F6C57A;
                width: 24px;
                height: 24px;
                margin: -8px 0;
                border-radius: 12px;
            }
            QSlider::handle:horizontal:hover {
                background: #F7B24F;
            }
            QProgressBar {
                background: #EEF3F5;
                border: 1px solid #D2DEE3;
                border-radius: 11px;
                text-align: center;
                color: #17313A;
                font-size: 14px;
                min-height: 24px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F4A63A, stop:1 #EE7C3D);
                border-radius: 11px;
            }
            QFrame {
                background: rgba(255, 255, 255, 0.58);
                border-radius: 20px;
                border: 1px solid rgba(164, 188, 198, 0.24);
            }
            #controlPanel {
                background: rgba(255, 255, 255, 0.54);
            }
            #displayPanel {
                background: rgba(255, 255, 255, 0.60);
            }
            #titleCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1A6B81, stop:0.54 #4FA8AA, stop:1 #F1B050);
                border: 0;
                border-radius: 28px;
            }
            #titleLabel {
                color: #FFFFFF;
                font-size: 30px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            #subtitleLabel {
                color: #F8FFFE;
                font-size: 16px;
                font-weight: 600;
                background: rgba(12, 53, 66, 0.20);
                border-radius: 10px;
                padding: 6px 10px;
            }
            #modelStatusLabel {
                background: rgba(255, 255, 255, 0.26);
                color: #FFFFFF;
                border-radius: 16px;
                padding: 12px 14px;
                font-size: 14px;
                font-weight: 600;
            }
            #dividerLine {
                background: rgba(95, 125, 135, 0.20);
                border: 0;
                min-height: 1px;
                max-height: 1px;
            }
            #sectionHintLabel {
                color: #264854;
                font-size: 14px;
                font-weight: 700;
            }
            #confValueLabel {
                color: #9A5B14;
                font-size: 19px;
                font-weight: 700;
                min-width: 62px;
                background: rgba(244, 166, 58, 0.12);
                border-radius: 10px;
                padding: 6px 10px;
                qproperty-alignment: 'AlignCenter';
            }
            #fileLabel {
                background: rgba(250, 252, 252, 0.96);
                border: 1px solid #CEDCE1;
                border-radius: 14px;
                padding: 12px 14px;
                color: #16313A;
            }
            #inputCombo {
                font-size: 15px;
                padding-top: 8px;
                padding-bottom: 8px;
            }
            #previewTitle {
                color: #17313A;
                font-size: 25px;
                font-weight: 800;
            }
            #previewSubtitle {
                color: #48636C;
                font-size: 14px;
                margin-bottom: 8px;
            }
            #imageViewport {
                background: qradialgradient(cx:0.5, cy:0.35, radius:1.0,
                    fx:0.5, fy:0.35, stop:0 #F5F8F9, stop:1 #E3ECEF);
                border-radius: 26px;
                border: 1px solid #D0DDE2;
                padding: 14px;
                color: #4C666E;
                font-size: 18px;
            }
            #primaryButton {
                font-size: 20px;
                padding: 12px 18px;
            }
            #secondaryButton {
                background: rgba(255, 255, 255, 0.92);
                color: #17313A;
                border: 1px solid #C7D7DD;
                font-size: 16px;
                padding: 10px 16px;
            }
            #secondaryButton:hover {
                background: #F6FBFC;
            }
            #ghostButton {
                background: rgba(244, 166, 58, 0.18);
                color: #8A4F12;
                border: 1px solid rgba(232, 157, 58, 0.24);
                font-size: 18px;
                padding: 12px 16px;
            }
            #ghostButton:hover {
                background: rgba(244, 166, 58, 0.28);
            }
        """)

    def init_models(self):
        try:
            # 优先加载专用人脸检测模型，失败时回退到 Haar
            self.face_model, self.face_model_type = load_face_detector()

            # 加载表情识别模型
            self.emotion_models = {}
            model_loaded = False

            # 尝试加载所有模型
            for model_name, model_path in self.model_paths.items():
                if os.path.exists(model_path):
                    try:
                        if is_lfs_pointer_file(model_path):
                            print(f"模型 {model_name} 仍是 Git LFS 占位文件: {model_path}")
                            continue
                        self.emotion_models[model_name] = YOLO(model_path)
                        model_loaded = True
                    except Exception as e:
                        print(f"加载模型 {model_name} 失败: {e}")

            # 设置当前使用的模型
            if model_loaded:
                current_model_name = self.model_combo.currentText()
                self.emotion_model = self.emotion_models.get(current_model_name)
                if not self.emotion_model and self.emotion_models:
                    # 如果当前选择的模型不可用，使用第一个可用的模型
                    first_available = list(self.emotion_models.keys())[0]
                    self.emotion_model = self.emotion_models[first_available]
                    # 更新下拉框选择
                    self.model_combo.setCurrentText(first_available)
            else:
                self.emotion_model = None
                QMessageBox.warning(
                    self,
                    "警告",
                    "未找到任何可用的表情识别模型。\n"
                    "如果文件存在但仍无法加载，很可能当前仓库里的 .pt 还是 Git LFS 占位文件。",
                )
            self.refresh_model_status()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.face_model = None
            self.face_model_type = None
            self.emotion_model = None
            self.emotion_models = {}
            self.refresh_model_status()

    def refresh_model_status(self):
        """刷新顶部模型状态摘要。"""
        if not hasattr(self, "model_status_label"):
            return

        face_text = "未加载"
        if self.face_model_type == "yolo":
            face_text = "YOLO Face"
        elif self.face_model_type == "haar":
            face_text = "OpenCV Haar"

        emotion_text = self.model_combo.currentText() if self.emotion_model else "未加载"
        self.model_status_label.setText(
            f"人脸检测: {face_text}  |  表情识别: {emotion_text}"
        )

    def change_emotion_model(self):
        """切换表情识别模型"""
        model_name = self.model_combo.currentText()

        # 如果模型已加载，直接使用
        if model_name in self.emotion_models:
            self.emotion_model = self.emotion_models[model_name]
            # 如果视频线程正在运行，更新其模型
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.set_models(self.face_model, self.emotion_model, self.face_model_type)
            self.refresh_model_status()
            return

        # 如果模型未加载，尝试加载
        model_path = self.model_paths.get(model_name)
        if model_path and os.path.exists(model_path):
            try:
                if is_lfs_pointer_file(model_path):
                    QMessageBox.warning(
                        self,
                        "警告",
                        f"模型文件仍是 Git LFS 占位文件，无法加载：\n{model_path}",
                    )
                    return
                self.emotion_models[model_name] = YOLO(model_path)
                self.emotion_model = self.emotion_models[model_name]
                # 如果视频线程正在运行，更新其模型
                if self.video_thread and self.video_thread.isRunning():
                    self.video_thread.set_models(self.face_model, self.emotion_model, self.face_model_type)
                self.refresh_model_status()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型 {model_name} 失败: {str(e)}")
        else:
            QMessageBox.warning(self, "警告", f"模型文件 {model_path} 不存在")

    def mode_changed(self):
        """处理模式变更"""
        mode = self.mode_combo.currentIndex()

        # 启用/禁用文件选择按钮
        self.file_button.setEnabled(mode > 0)

        # 重置文件选择
        if mode == 0:  # 摄像头模式
            self.file_label.setText("未选择文件")
            self.selected_file = None

    def select_file(self):
        """选择文件对话框"""
        mode = self.mode_combo.currentIndex()

        if mode == 1:  # 图片模式
            file_filter = "图片文件 (*.jpg *.jpeg *.png)"
        else:  # 视频模式
            file_filter = "视频文件 (*.mp4 *.avi *.mov *.mkv)"

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", file_filter
        )

        if file_path:
            self.selected_file = file_path
            self.file_label.setText(os.path.basename(file_path))

    def update_conf_threshold(self):
        """更新置信度阈值"""
        value = self.conf_slider.value() / 100.0
        self.conf_value_label.setText(f"{value:.2f}")

        # 如果视频线程正在运行，更新其阈值
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_conf_threshold(self.conf_slider.value())

    def toggle_detection(self):
        """开始/停止检测"""
        if self.video_thread and self.video_thread.isRunning():
            # 停止检测
            self.video_thread.stop()
            self.start_button.setText("开始检测")
            self.save_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.file_button.setEnabled(self.mode_combo.currentIndex() > 0)
            self.model_combo.setEnabled(True)  # 启用模型选择
            self.progress_group.setVisible(False)
        elif self.video_thread and not self.video_thread.isRunning() and self.mode_combo.currentIndex() == 1:
            # 图片模式特殊处理：线程可能已经结束但UI状态未更新
            self.start_button.setText("开始检测")
            self.save_button.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.file_button.setEnabled(True)
            self.model_combo.setEnabled(True)  # 启用模型选择
            self.progress_group.setVisible(False)

            # 清除视频线程引用，允许再次检测同一图片
            self.video_thread = None
        else:
            # 开始检测
            mode = self.mode_combo.currentIndex()

            if mode > 0 and not self.selected_file:
                QMessageBox.warning(self, "警告", "请先选择文件")
                return

            if not self.face_model:
                QMessageBox.critical(self, "错误", "人脸检测模型未加载")
                return

            if not self.emotion_model:
                QMessageBox.warning(self, "警告", "表情识别模型未加载，将只进行人脸检测")

            # 设置UI状态
            self.start_button.setText("停止检测")
            self.save_button.setEnabled(False)
            self.mode_combo.setEnabled(False)
            self.file_button.setEnabled(False)
            self.model_combo.setEnabled(False)  # 禁用模型选择

            # 显示进度条（仅视频模式）
            self.progress_group.setVisible(mode == 2)
            self.progress_bar.setValue(0)

            # 创建并启动视频线程
            if mode == 0:  # 摄像头模式
                self.video_thread = VideoThread(mode='camera')
            elif mode == 1:  # 图片模式
                self.video_thread = VideoThread(mode='image', file_path=self.selected_file)
            else:  # 视频模式
                self.video_thread = VideoThread(mode='video', file_path=self.selected_file)

            # 设置模型
            self.video_thread.set_models(self.face_model, self.emotion_model, self.face_model_type)
            self.video_thread.set_conf_threshold(self.conf_slider.value())

            # 连接信号
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.progress_signal.connect(self.update_progress)
            self.video_thread.error_signal.connect(self.handle_video_error)

            # 启动线程
            self.video_thread.start()

            # 对于图片模式，需要在线程结束后自动更新UI状态
            if mode == 1:  # 图片模式
                self.video_thread.finished.connect(lambda: self.image_processed_callback())

    def update_image(self, cv_img):
        """更新图像显示"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.current_image = cv_img.copy()

    def handle_video_error(self, message):
        """处理视频线程错误并恢复 UI 状态。"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        self.start_button.setText("开始检测")
        self.save_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.file_button.setEnabled(self.mode_combo.currentIndex() > 0)
        self.model_combo.setEnabled(True)
        self.progress_group.setVisible(False)
        QMessageBox.warning(self, "输入源错误", message)

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def save_result(self):
        """保存结果"""
        if not hasattr(self, 'current_image'):
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "图片文件 (*.jpg *.png)"
        )

        if file_path:
            try:
                # 创建目录
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                # 保存图像
                cv2.imwrite(file_path, self.current_image)
                QMessageBox.information(self, "成功", f"结果已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        event.accept()

    def image_processed_callback(self):
        """图片处理完成后的回调函数"""
        # 更新UI状态
        self.start_button.setText("开始检测")
        self.save_button.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.file_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_group.setVisible(False)

        # 清除视频线程引用，允许再次检测同一图片
        self.video_thread = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    configure_application_font(app)

    # 设置应用样式
    app.setStyle(QStyleFactory.create("Fusion"))

    # 创建并显示主窗口
    window = FaceDetectionApp()
    window.show()

    sys.exit(app.exec_())
