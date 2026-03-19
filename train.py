# from ultralytics import YOLO
# import time

# # 加载模型
# model = YOLO("yolo26x-cls.pt")  # 从YAML构建新模型
# # model = YOLO("yolo11l-cls.pt")

# results = model.train(
#     # fer2013plus
#     # affectnet
#     # my_yolo_emotion_dataset
#     # yolo_emotion_dataset_plus
#     data="datasets/archive",
#     epochs=200,
#     batch=128,
#     imgsz=224,
#     workers=6,

#     # 优化器设置
#     optimizer="AdamW",  # 使用具有自适应动量的现代优化器
#     lr0=0.001,  # 初始学习率
#     lrf=0.001,  # 最终学习率因子
#     warmup_epochs=5,  # 逐渐预热以防止早期不稳定
#     cos_lr=True,  # 余弦退火学习率调度

#     # 正则化
#     weight_decay=0.0005,  # L2正则化
#     dropout=0.2,  # 添加dropout以提高泛化能力

#     # 数据增强
#     augment=True,  # 启用内置增强
#     mixup=0.1,  # 应用mixup增强

#     # 训练管理
#     patience=20,  # 早停耐心值
#     save_period=10,  # 每10个epoch保存一次检查点

#     # 项目设置
#     project="runs/classify",
#     name="archive_yolo26x_cls",
#     exist_ok=True,
# )



from ultralytics import YOLO
import swanlab

# 初始化 SwanLab
run = swanlab.init(
    project="emotion-classification",
    experiment_name="archive_yolo26x_cls",
    config={
        "model": "yolo26x-cls",
        "epochs": 200,
        "batch": 128,
        "imgsz": 224,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "dropout": 0.2,
        "mixup": 0.1
    }
)

# 加载模型
model = YOLO("yolo26x-cls.pt")


# 记录每个epoch指标
def log_metrics(trainer):
    metrics = trainer.metrics

    swanlab.log({
        "train_loss": metrics.get("train/loss", 0),
        "val_loss": metrics.get("val/loss", 0),
        "top1_acc": metrics.get("metrics/top1", 0),
        "top5_acc": metrics.get("metrics/top5", 0),
        "lr": trainer.optimizer.param_groups[0]["lr"]
    })


# 注册 callback（不会改变训练参数）
model.add_callback("on_fit_epoch_end", log_metrics)


# 你的原始训练代码（完全不变）
results = model.train(
    data="datasets/archive",
    epochs=200,
    batch=128,
    imgsz=224,
    workers=6,

    # 优化器设置
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.001,
    warmup_epochs=5,
    cos_lr=True,

    # 正则化
    weight_decay=0.0005,
    dropout=0.2,

    # 数据增强
    augment=True,
    mixup=0.1,

    # 训练管理
    patience=20,
    save_period=10,

    # 项目设置
    project="runs/classify",
    name="archive_yolo26x_cls",
    exist_ok=True,
)

swanlab.finish()