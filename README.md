# 跌倒训练数据生成

用于生成跌倒检测模型训练数据的流水线工具。

## 项目结构

```
.
├── steps/              # 数据处理流水线步骤
│   ├── step1_video_split.py    # 视频切分
│   ├── step2_pose_detection.py # 姿态检测
│   ├── step3_data_cleaning.py   # 数据清洗
│   ├── step4_fall_split.py      # 跌倒片段切分
│   ├── step5_sample_extract.py # 样本提取
│   └── step6_feature_calc.py   # 特征计算
├── utils/              # 工具函数
│   ├── annotation_parser.py    # 标注解析
│   ├── file_utils.py           # 文件操作
│   └── pose_utils.py           # 姿态相关工具
├── main_pipeline.py    # 主流水线
├── config.py           # 配置文件
└── analyze_accel.py    # 加速度分析
```

## 主要功能

1. 视频切分与预处理
2. 基于 YOLO-Pose 的姿态检测
3. 跌倒动作识别与样本生成
4. 特征计算与数据导出

## 使用方法

```bash
python main_pipeline.py
```

## 依赖

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- ultralytics (YOLO)