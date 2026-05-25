# Fall Detection Training Data Generation - AGENTS.md

## Project Overview

跌倒检测训练数据生成项目，处理视频提取姿态关键点，生成139维特征用于模型训练。

**工作目录**: `D:\pycharm\工作脚本\跌倒检测\跌倒训练数据生成`

## Key Files

| 文件 | 说明 |
|------|------|
| `main_pipeline.py` | 主流程：视频切帧 → Pose检测 → 特征计算 → 输出NPZ |
| `config.py` | 配置文件：路径、帧率、关键点配置等 |
| `utils/pose_utils.py` | 关键点计算工具 |
| `steps/step2_pose_detection.py` | Pose检测模块 |

## Feature Dimensions (139维)

| 特征索引 | 名称 | 计算方式 |
|---------|------|----------|
| 0-33 | positions | 外接矩形相对坐标归一化，可为负值 |
| 34-67 | velocities | (positions[i] - positions[i-1]) / dt，单位秒⁻¹ |
| 68-101 | accelerations | (velocities[i] - velocities[i-1]) / dt，单位秒⁻² |
| 102-135 | relative_positions | 先归一化再减hip_center，容许负值 |
| 136 | spine_leg_angle | spine=左肩-左髋, leg=左膝-左髋, 使用arctanh允许负值 |
| 137 | height_change | 使用归一化坐标差 |
| 138 | body_orientation | spine=左肩-左髋, vertical=[0,1], 使用arctanh允许负值 |

## Important Fixes (经验记录)

### 1. 0-33 positions 不使用clip
- **原问题**: `np.clip((x-min_x)/bbox_w, 0, 1)` 将负值变成0
- **修复**: 移除clip，容许相对坐标为负值

### 2. 102-135 relative_positions 计算顺序
- **原问题**: 先减hip_center再归一化
- **修复**: 先归一化positions_norm，再减归一化后的hip_center
```python
# 正确方式
hip_center_norm = [(hip_center[0]-min_x)/bbox_w, (hip_center[1]-min_y)/bbox_h]
rel = positions_norm[i] - hip_center_norm_flat
```

### 3. 136 spine_leg_angle 向量定义
- **原问题**: spine用双肩中心-双髋中心，leg用膝-踝平均
- **修复**:
  - spine = 左肩 - 左髋
  - leg = 左膝 - 左髋

### 4. 138 body_orientation
- **原问题**: vertical=[0, -1]，spine用中心点
- **修复**:
  - vertical = [0, 1] (y轴向下)
  - spine = 左肩 - 左髋

### 5. 无人帧过滤
- 在extract_samples中，跳过包含无人帧(x全为0)的样本

### 6. 34-67 velocities 和 68-101 accelerations 正确计算
- **问题**: 原本直接置为0，导致速度和加速度维度无效
- **修复**:
  - velocities: `v = (positions_norm[i] - positions_norm[i-1]) / dt`
  - accelerations: `a = (velocities[i] - velocities[i-1]) / dt`
  - dt为时间间隔（秒），使用实际时间戳差分计算
  - 第一帧velocities为0（无前一帧），前两帧accelerations为0（无足够速度数据）

### 7. 136 spine_leg_angle 和 138 body_orientation 使用 arctanh
- **原问题**: `arccos(cos_angle)` 输出范围 [0, π]，无法表示负值
- **修复**: 改用 `arctanh(cos_angle)` 输出范围 (-∞, +∞)，允许负值
- **注意**: cos_angle 需要 clip 到 [-0.9999, 0.9999] 避免数值溢出

## Data Split

```python
# 9:1分割，保持类别均衡
# 训练集: 16500样本 (FALL=5709, NOFALL=10791)
# 验证集: 1834样本 (FALL=635, NOFALL=1199)
```

## Output Format

- **NPZ格式**: `X.shape=(N, 10, 139)`, `y.shape=(N,)`
- **标签**: 1=FALL, 0=NOFALL

## Common Commands

```bash
# 运行完整流程
python main_pipeline.py --batch --workers 1

# 解析NPZ为Excel
python -c "import numpy as np; npz=np.load('path'); ..."
```