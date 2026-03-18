"""
YOLOv11人脸检测与表情识别示例代码
- 使用YOLOv11模型进行人脸检测
- 使用训练好的表情识别模型进行表情分类
- 支持摄像头实时检测、图片检测和视频文件检测
- 自动调整文本位置以避免边缘遮挡
- 需要安装 ultralytics、opencv-python、Pillow 等依赖库
使用方法:
1. 摄像头实时检测:
python yolo_face_detection.py
2. 图片检测:
python yolo_face_detection.py --input path/to/image.jpg
3. 视频文件检测:
python yolo_face_detection.py --input path/to/video.mp4
4. 显示窗口:
python yolo_face_detection.py --input path/to/image.jpg --show
注意事项:
- 确保已训练好表情识别模型并将路径正确设置在 emotion_model_path 变量中
- 确保已下载 YOLOv11 人脸检测模型并将路径正确设置在 download_face_model 函数中
- 在某些环境中可能无法显示 OpenCV 窗口，使用 --show 参数时会尝试显示窗口，如果失败会自动跳过显示并继续处理
- 处理视频文件时会将结果保存为新的视频文件，文件名以 yolo_result_ 开头，保存在 results 目录下
"""


import cv2
import numpy as np
from ultralytics import YOLO
import time
from PIL import Image, ImageDraw, ImageFont
import os
import argparse
import torch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_FONT_DIR = PROJECT_ROOT / "fonts"
SYSTEM_FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
    Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/google-noto-cjk/NotoSerifCJK-Regular.ttc"),
]
PROJECT_FONT_PATTERNS = (
    "*NotoSansSC*.otf",
    "*NotoSansSC*.ttf",
    "*NotoSansCJK*.ttc",
    "*SourceHanSans*.otf",
    "*SourceHanSans*.ttf",
    "*wqy*.ttf",
    "*wqy*.ttc",
    "*.ttf",
    "*.ttc",
    "*.otf",
)


def download_face_model():
    """返回人脸检测模型路径。"""
    env_path = os.environ.get("FACE_MODEL_PATH")
    if env_path:
        return env_path
    return "yolov12l-face.pt"


def is_lfs_pointer_file(file_path):
    """判断文件是否为 Git LFS 占位文件。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            header = f.read(128)
        return header.startswith("version https://git-lfs.github.com/spec/v1")
    except (OSError, UnicodeDecodeError):
        return False


def load_emotion_model():
    """加载表情识别模型，自动跳过 LFS 占位文件。"""
    candidates = [
        "best_yolo26x_cls.pt",
    ]

    for model_path in candidates:
        if not os.path.exists(model_path):
            continue
        if is_lfs_pointer_file(model_path):
            print(f"警告：模型文件是 Git LFS 占位文件，跳过: {model_path}")
            continue
        try:
            print(f"加载表情识别模型: {model_path}")
            return YOLO(model_path)
        except Exception as e:
            print(f"警告：加载表情识别模型失败 {model_path}: {e}")

    print("警告：未找到可用的表情识别模型，将仅进行人脸检测")
    return None


def find_chinese_font_path():
    """查找可用的中文字体文件路径。"""
    windows_candidates = [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
    ]
    for candidate in windows_candidates + SYSTEM_FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    PROJECT_FONT_DIR.mkdir(exist_ok=True)
    for pattern in PROJECT_FONT_PATTERNS:
        matches = sorted(PROJECT_FONT_DIR.rglob(pattern))
        if matches:
            return str(matches[0])

    return None


def load_font(size=24):
    """加载中文字体。"""
    font_path = find_chinese_font_path()
    try:
        if font_path is None:
            raise FileNotFoundError("未找到可用的中文字体文件")
        font = ImageFont.truetype(font_path, size)
        print(f"已加载字体: {font_path}")
        return font
    except Exception as e:
        print(f"加载字体失败: {e}")
        print("将使用默认字体，中文可能显示为方框或乱码")
        return ImageFont.load_default()


def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), font=None, adjust_position=True):
    """在OpenCV图像上添加中文文本，可自动调整位置避免边缘遮挡"""
    if font is None:
        font = load_font()

    # 获取图像尺寸
    img_height, img_width = img.shape[:2]
    x_pos, y_pos = position

    # 判断是否需要使用PIL绘制
    if font is not None and font != ImageFont.load_default():
        # 转换图像从OpenCV格式(BGR)到PIL格式(RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建绘图对象
        draw = ImageDraw.Draw(img_pil)

        # 估算文本尺寸
        text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]

        # 如果需要调整位置以避免边缘遮挡
        if adjust_position:
            # 检查是否会超出左边界
            if x_pos < 0:
                x_pos = 5

            # 检查是否会超出右边界
            if x_pos + text_width > img_width:
                x_pos = img_width - text_width - 5

            # 检查是否会超出上边界
            if y_pos < text_height:
                # 如果文本在人脸框上方且会超出上边界，将其移到人脸框下方
                y_pos = position[1] + 30  # 移到人脸框下方

        # 绘制文本
        draw.text((x_pos, y_pos), text, font=font, fill=text_color[::-1])  # PIL颜色顺序是RGB
        # 转换回OpenCV格式
        img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_with_text
    else:
        # 如果没有合适的字体，使用OpenCV默认方法（可能显示乱码）
        # 估算文本尺寸
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

        # 如果需要调整位置以避免边缘遮挡
        if adjust_position:
            # 检查是否会超出左边界
            if x_pos < 0:
                x_pos = 5

            # 检查是否会超出右边界
            if x_pos + text_width > img_width:
                x_pos = img_width - text_width - 5

            # 检查是否会超出上边界
            if y_pos < text_height:
                # 如果文本在人脸框上方且会超出上边界，将其移到人脸框下方
                y_pos = position[1] + 30  # 移到人脸框下方

        cv2.putText(img, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        return img


def translate_emotion_label(label):
    """将模型类别名转换为中文显示。"""
    mapping = {
        "angry": "愤怒",
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "高兴",
        "sad": "悲伤",
        "surprise": "惊讶",
        "neutral": "中性",
    }
    return mapping.get(str(label).lower(), str(label))


def get_top1_emotion(result):
    """从分类结果中提取 top1 类别及置信度。"""
    probs = result.probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    emotion_name = result.names[class_id]
    emotion_label = translate_emotion_label(emotion_name)
    return emotion_name, emotion_label, confidence


def show_frame(window_name, frame, wait_key=1):
    """尝试显示窗口，失败时返回 False。"""
    try:
        cv2.imshow(window_name, frame)
        return (cv2.waitKey(wait_key) & 0xFF) != ord('q')
    except cv2.error as e:
        print(f"当前 OpenCV 环境不支持窗口显示，已跳过 imshow: {e}")
        return False


def detect_faces_video(show_window=True):
    """使用YOLOv11人脸检测模型进行视频人脸检测（摄像头）"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model = load_emotion_model()

    # 加载字体
    font = load_font()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("按 'q' 键退出程序")

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            print("无法获取视频帧")
            break

        # 使用YOLOv11检测人脸

        results = face_model(frame, conf=0.8)
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 扩大边界框（调整人脸框大小）
                frame_height, frame_width = frame.shape[:2]
                # 计算边界框的扩展量（框的20%）
                expand_x = int((x2 - x1) * 0.2)
                expand_y = int((y2 - y1) * 0.2)

                # 应用扩展，但确保不超出图像边界
                x1_expanded = max(0, x1 - expand_x)
                y1_expanded = max(0, y1 - expand_y)
                x2_expanded = min(frame_width, x2 + expand_x)
                y2_expanded = min(frame_height, y2 + expand_y)

                # 绘制扩大后的人脸框
                cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                # 提取扩大后的人脸区域
                face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                # 如果没有加载表情识别模型，则跳过表情识别
                if emotion_model is None or face_roi.size == 0:
                    continue

                try:
                    # 将人脸区域转换为灰度图像，与训练数据保持一致
                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                    # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                    face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                    # 使用YOLO模型进行表情识别
                    emotion_results = emotion_model(face_roi_gray_3ch, verbose=False)
                    emotion_name, emotion, confidence = get_top1_emotion(emotion_results[0])

                    # 在图像上显示预测结果
                    text = f"{emotion}: {confidence:.2f}"

                    # 确定文本位置，智能调整以避免边缘遮挡
                    text_position = (x1_expanded, y1_expanded - 30)
                    # 如果文本位置太靠近上边界，则将其移到人脸框下方
                    if text_position[1] < 30:
                        text_position = (x1_expanded, y2_expanded + 30)

                    # 使用自定义函数添加中文文本
                    frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font, adjust_position=True)
                    print(f"检测结果: {emotion} / {emotion_name} ({confidence:.2f})")
                except Exception as e:
                    print(f"处理人脸时出错: {e}")

        # 显示结果
        if show_window and not show_frame("YOLO人脸检测与表情识别", frame, wait_key=1):
            break

    # 释放资源
    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def detect_faces_image(image_path, show_window=False):
    """使用YOLOv11人脸检测模型进行图片人脸检测"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model = load_emotion_model()

    # 加载字体
    font = load_font()

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 使用YOLOv11检测人脸

    results = face_model(image, conf=0.8)
    # 检查是否检测到人脸
    if len(results[0].boxes) == 0:
        print("未检测到人脸")
        image = cv2_add_chinese_text(image, "未检测到人脸", (20, 60), (0, 0, 255), font)
    else:
        print(f"检测到 {len(results[0].boxes)} 个人脸")

    # 处理检测结果
    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 扩大边界框（调整人脸框大小）
            image_height, image_width = image.shape[:2]
            # 计算边界框的扩展量（框的20%）
            expand_x = int((x2 - x1) * 0.2)
            expand_y = int((y2 - y1) * 0.2)

            # 应用扩展，但确保不超出图像边界
            x1_expanded = max(0, x1 - expand_x)
            y1_expanded = max(0, y1 - expand_y)
            x2_expanded = min(image_width, x2 + expand_x)
            y2_expanded = min(image_height, y2 + expand_y)

            # 绘制扩大后的人脸框
            cv2.rectangle(image, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

            # 如果没有加载表情识别模型，则跳过表情识别
            if emotion_model is None:
                continue

            # 提取扩大后的人脸区域
            face_roi = image[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

            if face_roi.size == 0:
                continue

            try:
                # 将人脸区域转换为灰度图像，与训练数据保持一致
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                # 使用YOLO模型进行表情识别
                emotion_results = emotion_model(face_roi_gray_3ch, verbose=False)
                emotion_name, emotion, confidence = get_top1_emotion(emotion_results[0])

                # 在图像上显示预测结果
                text = f"人脸 {j + 1}: {emotion} ({confidence:.2f})"

                # 确定文本位置，智能调整以避免边缘遮挡
                text_position = (x1_expanded, y1_expanded - 30)
                # 如果文本位置太靠近上边界，则将其移到人脸框下方
                if text_position[1] < 30:
                    text_position = (x1_expanded, y2_expanded + 30)

                image = cv2_add_chinese_text(image, text, text_position, (36, 255, 12), font, adjust_position=True)
                print(f"人脸 {j + 1}: {emotion} / {emotion_name} (置信度: {confidence:.2f})")
            except Exception as e:
                print(f"处理人脸 {j + 1} 时出错: {e}")

    # 保存结果
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"yolo_result_{filename}")
    cv2.imwrite(output_path, image)
    print(f"结果已保存至: {output_path}")

    if show_window:
        show_frame("YOLO人脸检测与表情识别结果", image, wait_key=0)
        cv2.destroyAllWindows()


def detect_faces_video_file(video_path, show_window=True):
    """使用YOLOv11人脸检测模型进行视频文件人脸检测"""
    # 下载并加载YOLOv11人脸检测模型
    face_model_path = download_face_model()
    face_model = YOLO(face_model_path)  # 使用专门的人脸检测模型

    # 加载表情识别模型
    emotion_model = load_emotion_model()

    # 加载字体
    font = load_font()

    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}")
        return

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出目录
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置输出视频
    video_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"yolo_result_{video_filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"正在处理视频: {video_path}")
    print("按 'q' 键退出程序")

    frame_count = 0

    while True:
        # 读取视频帧
        ret, frame = cap.read()

        if not ret:
            print("视频处理完成")
            break

        frame_count += 1

        # 使用YOLOv11检测人脸
        # results = face_model(frame)
        results = face_model(frame, conf=0.8)
        # 处理检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # 扩大边界框（调整人脸框大小）
                # 计算边界框的扩展量（框的20%）
                expand_x = int((x2 - x1) * 0.2)
                expand_y = int((y2 - y1) * 0.2)

                # 应用扩展，但确保不超出图像边界
                x1_expanded = max(0, x1 - expand_x)
                y1_expanded = max(0, y1 - expand_y)
                x2_expanded = min(frame_width, x2 + expand_x)
                y2_expanded = min(frame_height, y2 + expand_y)

                # 绘制扩大后的人脸框
                cv2.rectangle(frame, (x1_expanded, y1_expanded), (x2_expanded, y2_expanded), (0, 255, 0), 2)

                # 如果没有加载表情识别模型，则跳过表情识别
                if emotion_model is None or (y2_expanded - y1_expanded) <= 0 or (x2_expanded - x1_expanded) <= 0:
                    continue

                try:
                    # 提取扩大后的人脸区域
                    face_roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                    if face_roi.size == 0:
                        continue

                    # 将人脸区域转换为灰度图像，与训练数据保持一致
                    face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                    # 将灰度图像转换为3通道，因为YOLO模型需要3通道输入
                    face_roi_gray_3ch = cv2.cvtColor(face_roi_gray, cv2.COLOR_GRAY2BGR)

                    # 使用YOLO模型进行表情识别
                    emotion_results = emotion_model(face_roi_gray_3ch, verbose=False)
                    emotion_name, emotion, confidence = get_top1_emotion(emotion_results[0])

                    # 在图像上显示预测结果
                    text = f"{emotion}: {confidence:.2f}"

                    # 确定文本位置，智能调整以避免边缘遮挡
                    text_position = (x1_expanded, y1_expanded - 30)
                    # 如果文本位置太靠近上边界，则将其移到人脸框下方
                    if text_position[1] < 30:
                        text_position = (x1_expanded, y2_expanded + 30)

                    # 使用自定义函数添加中文文本
                    frame = cv2_add_chinese_text(frame, text, text_position, (36, 255, 12), font, adjust_position=True)
                    print(f"检测结果: {emotion} / {emotion_name} ({confidence:.2f})")
                except Exception as e:
                    print(f"处理人脸时出错: {e}")

        # 显示进度
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")

        # 写入输出视频
        out.write(frame)

        # 显示结果
        if show_window and not show_frame("YOLO人脸检测与表情识别", frame, wait_key=1):
            break

    # 释放资源
    cap.release()
    out.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"处理完成，结果已保存至: {output_path}")


def main():
    """主函数，根据输入类型选择相应的处理方式"""
    parser = argparse.ArgumentParser(description="YOLO 人脸检测与表情识别")
    parser.add_argument("--input", "-i", dest="input_path", default=None, help="图片或视频路径；留空则使用摄像头")
    parser.add_argument("--show", action="store_true", help="尝试显示 OpenCV 窗口")
    args = parser.parse_args()

    # 获取用户输入
    input_path = args.input_path.strip() if args.input_path else input("请输入图片/视频文件路径（直接回车使用摄像头）：").strip()

    # 根据输入类型选择相应的处理方式
    if not input_path:
        # 没有提供输入路径，使用摄像头
        print("启动视频流人脸检测模式...")
        detect_faces_video(show_window=args.show)
    else:
        # 检查文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：文件不存在: {input_path}")
            return

        # 获取文件扩展名
        _, ext = os.path.splitext(input_path.lower())

        if ext in ['.jpg', '.jpeg', '.png']:
            # 图片处理
            print(f"正在处理图片: {input_path}")
            detect_faces_image(input_path, show_window=args.show)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            # 视频处理
            print(f"正在处理视频: {input_path}")
            detect_faces_video_file(input_path, show_window=args.show)
        else:
            print(f"不支持的文件类型: {ext}")
            print("支持的文件类型: .jpg, .jpeg, .png, .mp4, .avi, .mov, .mkv")


if __name__ == "__main__":
    main()
