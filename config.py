"""配置文件 - 跌倒检测训练数据生成"""
import os
from pathlib import Path

# 路径配置
VIDEO_DIR = Path(r"D:\IPC\IPC_video_data_mp4\跌倒\data_positive_0512")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒")
FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒")
MODEL_PATH = Path(r"D:\pycharm\工作脚本\跌倒\跌倒训练数据生成\yolo26l-pose.pt")

# 视频信息 (实际FPS=20, 总帧数=662)
VIDEO_NAME = "2026-03-03_10-18-25"  # 默认视频名
VIDEO_FPS = 20.0
VIDEO_TOTAL_FRAMES = 662

# 帧率配置
FRAME_INTERVALS_MS = [250, 300, 350, 400, 450, 500]
SAMPLE_FRAMES = 11

# 关键点配置 (COCO 17点)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
NUM_KEYPOINTS = 17
HIP_CENTER_IDX = (11 + 12) // 2  # 11=left_hip, 12=right_hip


# 特征维度
FEATURE_DIM = 123
POS_DIM = 34  # 17 * 2
INTERVAL_DIM = 10
BBOX_WH_DIM = 22  # 11帧 * 2
BBOX_RATIO_AREA_DIM = 20  # 10帧 * (w/h, w*h)
REL_POS_DIM = 34
ANGLE_SPINE_LEG = 102
HIP_HEIGHT_CHANGE = 103
BODY_ORIENTATION = 104
LABEL_DIM = 105