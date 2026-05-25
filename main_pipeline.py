"""主流程脚本 - 跌倒检测训练数据生成"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import os
import shutil
import numpy as np
import pandas as pd
import json
import cv2
import random
import logging
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def _compute_features_wrapper(args):
    """包装compute_features_for_sample用于并行"""
    sample_detections, sample_timestamps, interval_ms, intervals_ms = args
    return compute_features_for_sample(
        sample_detections, sample_timestamps, interval_ms, intervals_ms
    )


# 全局标志：是否使用内部并行（当外部run_parallel.py提供进程隔离时设为False）
# 通过环境变量设置，避免进程间传递变量的复杂性
_DISABLE_INTERNAL_PARALLELISM = os.environ.get('DISABLE_INTERNAL_PARALLEL', '0') == '1'


def _should_disable_gpu_parallelism() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() <= 1
    except Exception:
        return False
def compute_features_batch(samples: list, intervals_ms: list, max_workers: int = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    使用进程池并行计算多个样本的特征
    返回: (all_sample_features, all_relationships)
    """
    if _DISABLE_INTERNAL_PARALLELISM:
        # 单进程顺序处理，避免嵌套并行
        all_sample_features = []
        all_relationships = []
        for s in samples:
            feat, fall_rel = compute_features_for_sample(
                s['detections'], s['timestamps'], s['interval_ms'], intervals_ms
            )
            all_sample_features.append(feat)
            all_relationships.append(fall_rel)
        return all_sample_features, all_relationships

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    all_sample_features = []
    all_relationships = []

    # 准备参数列表
    args_list = [
        (s['detections'], s['timestamps'], s['interval_ms'], intervals_ms)
        for s in samples
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_compute_features_wrapper, args) for args in args_list]
        for future in as_completed(futures):
            feat, fall_rel = future.result()
            all_sample_features.append(feat)
            all_relationships.append(fall_rel)

    return all_sample_features, all_relationships

# 配置日志 - 只写文件，不输出到终端
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

from utils import AnnotationParser, PoseUtils


def detect_single_round(round_dir: Path, round_idx: int, model_path: Path, skip_existing: bool) -> tuple:
    """检测单个round的pose（用于并行）"""
    pose_results_path = round_dir / "pose_results.npz"
    if skip_existing and pose_results_path.exists():
        return round_idx, True
    from steps.step2_pose_detection import PoseDetector
    detector = PoseDetector(model_path)
    detector.detect_folder(round_dir, pose_results_path, force=True, batch_size=32)
    return round_idx, False


def _detect_round_wrapper(args):
    """包装detect_single_round用于并行"""
    round_dir, round_idx, model_path, skip_existing = args
    return detect_single_round(round_dir, round_idx, model_path, skip_existing)


def load_pose_results(npz_path: Path) -> np.ndarray:
    """加载pose检测结果"""
    data = np.load(npz_path, allow_pickle=True)
    return data['results']


def load_frame_timestamps(round_dir: Path) -> list:
    """加载帧时间戳"""
    timestamps_path = round_dir / "frame_timestamps.json"
    if timestamps_path.exists():
        with open(timestamps_path, 'r') as f:
            return json.load(f)
    return None


def clean_no_person_frames(frames: list, detections: np.ndarray, gap_threshold: int = 20) -> list:
    """清洗逻辑：删除所有无人帧，以及无人区间前后的短暂有人帧"""
    person_flags = np.array([det.get('has_person', False) for det in detections])
    n = len(person_flags)

    no_person_regions = []
    in_region = False
    start = 0

    for i in range(n):
        if not person_flags[i] and not in_region:
            in_region = True
            start = i
        elif person_flags[i] and in_region:
            in_region = False
            no_person_regions.append((start, i - 1))

    if not person_flags[-1]:
        no_person_regions.append((start, n - 1))

    logger.info(f"  无人区间: {[(f'{s}-{e}', e-s+1) for s,e in no_person_regions]}")

    to_delete = set()
    for start, end in no_person_regions:
        for i in range(start, end + 1):
            to_delete.add(i)

        if start > 0 and not person_flags[start - 1]:
            pre_start = start - 1
            while pre_start > 0 and not person_flags[pre_start - 1]:
                pre_start -= 1
            if start - pre_start <= gap_threshold:
                for i in range(pre_start, start):
                    to_delete.add(i)

        if end < n - 1 and not person_flags[end + 1]:
            post_end = end + 1
            while post_end < n - 1 and not person_flags[post_end + 1]:
                post_end += 1
            if post_end - end <= gap_threshold:
                for i in range(end + 1, post_end + 1):
                    to_delete.add(i)

    kept_indices = [i for i in range(n) if i not in to_delete]
    logger.info(f"  清洗: 删除{len(to_delete)}帧, 保留{len(kept_indices)}帧")

    return kept_indices


def compute_spine_leg_angle(positions: np.ndarray) -> float:
    """计算躯干与腿夹角"""
    # spine_vector = 左肩 - 左髋
    left_shoulder = positions[10:12]
    left_hip = positions[22:24]
    spine = left_shoulder - left_hip

    # leg_vector = 左膝 - 左髋
    left_knee = positions[26:28]
    leg = left_knee - left_hip

    def angle_between(v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    return angle_between(spine, leg)


def compute_body_orientation(positions: np.ndarray) -> float:
    """计算身体朝向"""
    # spine_vector = 左肩 - 左髋
    left_shoulder = positions[10:12]
    left_hip = positions[22:24]
    spine = left_shoulder - left_hip
    vertical = np.array([0, 1])  # y轴向下

    def angle_between(v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    return angle_between(spine, vertical)

def extract_keypoints_flat(det: dict) -> np.ndarray:
    """从检测结果提取并展平关键点"""
    keypoints = det.get('keypoints', [])
    if isinstance(keypoints, np.ndarray) and keypoints.shape[0] >= 17:
        flat = keypoints[:17, :2].flatten()
        return flat.astype(np.float64)
    return np.zeros(34, dtype=np.float64)


def compute_features_for_sample(sample_detections: list, sample_timestamps: list,
                                 interval_ms: list, intervals_ms: list) -> tuple:
    """计算单个样本的139维特征"""
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    # 计算外接矩形（跳过无人帧）
    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    # 归一化positions使用外接矩形（而非固定图片尺寸）
    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = np.clip((positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1), 0, 1)
    positions_norm[:, 1::2] = np.clip((positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1), 0, 1)

    # 使用实际时间戳计算每帧的速度
    velocities = []
    for i in range(1, len(positions_seq)):
        dt = sample_timestamps[i] - sample_timestamps[i-1]
        if dt > 0:
            # 特征34-67使用归一化坐标差来计算速度
            v = (positions_norm[i] - positions_norm[i-1]) / dt
        else:
            v = np.zeros(34)
        velocities.append(v)
    velocities_seq = np.array(velocities) if velocities else np.zeros((0, 34))

    # 计算accelerations (使用velocities的差来计算)
    accelerations = []
    for i in range(1, len(velocities_seq)):
        dt_prev = sample_timestamps[i] - sample_timestamps[i-1]
        dt_curr = sample_timestamps[i+1] - sample_timestamps[i]
        dt_avg = (dt_prev + dt_curr) / 2 if (dt_prev + dt_curr) > 0 else 1
        if dt_avg > 0:
            # 特征68-101使用速度差来计算加速度
            a = (velocities_seq[i] - velocities_seq[i-1]) / dt_avg
        else:
            a = np.zeros(34)
        accelerations.append(a)
    accelerations_seq = np.array(accelerations) if accelerations else np.zeros((0, 34))

    rel_positions = []
    for positions in positions_seq:
        left_hip = positions[22:24]
        right_hip = positions[24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_flat = np.tile(hip_center, 17)
        rel = positions - hip_center_flat
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    spine_leg_angles = [compute_spine_leg_angle(p) for p in positions_seq]

    hip_height_changes = []
    for i in range(1, len(positions_seq)):
        hip_center_y_curr = (positions_seq[i][23] + positions_seq[i][25]) / 2
        hip_center_y_prev = (positions_seq[i-1][23] + positions_seq[i-1][25]) / 2
        hip_height_changes.append(hip_center_y_curr - hip_center_y_prev)
    hip_height_changes = np.array(hip_height_changes)

    body_orientations = [compute_body_orientation(p) for p in positions_seq]

    # 使用实际时间戳计算样本时间范围
    sample_start_s = sample_timestamps[0]
    sample_end_s = sample_timestamps[-1]
    sample_start_ms = sample_start_s * 1000
    sample_end_ms = sample_end_s * 1000

    label = 0
    fall_relationship = ""

    for s_ms, e_ms in intervals_ms:
        if s_ms >= sample_start_ms and e_ms <= sample_end_ms:
            label = 1
            fall_relationship = f"fall_in_sample:{s_ms:.0f}-{e_ms:.0f}"
            break
        elif s_ms <= sample_end_ms and e_ms >= sample_start_ms:
            fall_relationship = f"overlap:{s_ms:.0f}-{e_ms:.0f}"

    features = []
    features.extend(positions_norm[-1].tolist())
    if len(velocities_seq) > 0:
        features.extend(velocities_seq[-1].tolist())
    else:
        features.extend([0] * 34)
    if len(accelerations_seq) > 0:
        features.extend(accelerations_seq[-1].tolist())
    else:
        features.extend([0] * 34)
    features.extend(rel_positions_seq[-1].tolist())
    features.append(spine_leg_angles[-1])
    features.append(hip_height_changes[-1] if len(hip_height_changes) > 0 else 0)
    features.append(body_orientations[-1])
    features.append(label)

    return np.array(features), fall_relationship


def compute_bounding_box(positions_seq: np.ndarray) -> tuple:
    """
    计算11帧的最小外接矩形（跳过无人帧，即全0坐标帧）
    返回: (left, top, right, bottom, bbox_w, bbox_h)
    """
    # 过滤掉无人帧（全0坐标帧）
    valid_frames = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:  # 非无人帧
            valid_frames.append(i)

    if len(valid_frames) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    all_x = positions_seq[valid_frames][:, 0::2]  # 所有x坐标
    all_y = positions_seq[valid_frames][:, 1::2]  # 所有y坐标

    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)

    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    return min_x, min_y, max_x, max_y, bbox_w, bbox_h


def compute_sample_features_for_npz(sample_detections: list, sample_timestamps: list) -> np.ndarray:
    """
    计算单个样本所有11帧的123维特征，用于NPZ输出
    返回: shape (10, 123) - 10帧有效数据(跳过第一帧)
    """
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    # 计算11帧的最小外接矩形
    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    # 0-33: positions (外接矩形相对坐标归一化，容许负值)
    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = (positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1)
    positions_norm[:, 1::2] = (positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1)

    # 计算每帧的相对位置：先归一化positions_norm，再减hip_center（hip_center也需要归一化）
    rel_positions = []
    for i in range(len(positions_seq)):
        left_hip = positions_seq[i, 22:24]
        right_hip = positions_seq[i, 24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_norm = np.array([
            (hip_center[0] - min_x) / (bbox_w if bbox_w > 0 else 1),
            (hip_center[1] - min_y) / (bbox_h if bbox_h > 0 else 1)
        ])
        hip_center_norm_flat = np.tile(hip_center_norm, 17)
        rel = positions_norm[i] - hip_center_norm_flat
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    # 计算每帧的bbox宽高（相对于最小外接矩形归一化）
    bbox_widths = []
    bbox_heights = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:
            w = (np.max(frame_x) - np.min(frame_x)) / (bbox_w if bbox_w > 0 else 1)
            h = (np.max(frame_y) - np.min(frame_y)) / (bbox_h if bbox_h > 0 else 1)
        else:
            w, h = 0, 0
        bbox_widths.append(w)
        bbox_heights.append(h)

    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    bbox_ratios = np.divide(
        bbox_widths[1:],
        bbox_heights[1:],
        out=np.zeros_like(bbox_widths[1:]),
        where=bbox_heights[1:] != 0
    )
    bbox_areas = bbox_widths[1:] * bbox_heights[1:]

    spine_leg_angles = np.array([compute_spine_leg_angle(p) for p in positions_seq])
    hip_height_changes = np.array([(((positions_norm[i][23] + positions_norm[i][25]) / 2) -
                                     ((positions_norm[i-1][23] + positions_norm[i-1][25]) / 2))
                                    for i in range(1, len(positions_norm))])
    body_orientations = np.array([compute_body_orientation(p) for p in positions_seq])

    # 为每帧构建123维特征
    sample_features = []
    n_frames = len(positions_seq)

    for frame_idx in range(n_frames):
        features = []
        # 0-33: positions (归一化)
        features.extend(positions_norm[frame_idx].tolist())

        # 34-43: 采样间隔 (10个时间差)
        if frame_idx > 0:
            dt = sample_timestamps[frame_idx] - sample_timestamps[frame_idx - 1]
            features.extend([dt] * 10)
        else:
            features.extend([0] * 10)

        # 44-63: bbox归一化宽高 (20个值 = 10帧的宽高，去掉第0帧，按w/h交替)
        for w, h in zip(bbox_widths[1:], bbox_heights[1:]):
            features.extend([w, h])

        # 64-73: bbox宽高比 (10个值 = w / h)
        features.extend(bbox_ratios.tolist())

        # 74-83: bbox宽高积 (10个值 = w * h)
        features.extend(bbox_areas.tolist())

        # 84-117: relative_positions
        features.extend(rel_positions_seq[frame_idx].tolist())

        # 118: spine_leg_angle
        features.append(spine_leg_angles[frame_idx])

        # 119: hip_height_change
        if frame_idx > 0:
            features.append(hip_height_changes[frame_idx - 1])
        else:
            features.append(0)

        # 120: body_orientation
        features.append(body_orientations[frame_idx])

        sample_features.append(features)

    return np.array(sample_features)[1:]  # shape (10, 121) - 跳过第一帧


def compute_sample_debug_details(sample_detections: list, sample_timestamps: list,
                                  sample_idx: int, intervals_ms: list) -> str:
    """
    生成单个样本的123维详细计算过程
    """
    positions_list = [extract_keypoints_flat(d) for d in sample_detections]
    positions_seq = np.array(positions_list)

    # 计算外接矩形（跳过无人帧）
    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    # 归一化positions使用外接矩形（而非固定图片尺寸）
    positions_norm = positions_seq.copy()
    positions_norm[:, 0::2] = np.clip((positions_norm[:, 0::2] - min_x) / (bbox_w if bbox_w > 0 else 1), 0, 1)
    positions_norm[:, 1::2] = np.clip((positions_norm[:, 1::2] - min_y) / (bbox_h if bbox_h > 0 else 1), 0, 1)

    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Sample #{sample_idx}")
    lines.append(f"{'='*60}")
    lines.append(f"时间范围: {sample_timestamps[0]:.3f}s - {sample_timestamps[-1]:.3f}s")
    lines.append(f"帧间隔: {[f'{t:.0f}ms' for t in [(sample_timestamps[i]-sample_timestamps[i-1])*1000 for i in range(1, len(sample_timestamps))]]}")
    lines.append(f"总帧数: {len(sample_detections)}")

    # 计算外接矩形（跳过无人帧）
    min_x, min_y, max_x, max_y, bbox_w, bbox_h = compute_bounding_box(positions_seq)

    lines.append("")
    lines.append("-" * 60)
    lines.append("外接矩形计算 (Bounding Box)")
    lines.append("-" * 60)
    lines.append(f"  各帧bbox范围:")
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        f_min_x, f_max_x = np.min(frame_x), np.max(frame_x)
        f_min_y, f_max_y = np.min(frame_y), np.max(frame_y)
        lines.append(f"    帧{i}: x=[{f_min_x:.1f}, {f_max_x:.1f}], y=[{f_min_y:.1f}, {f_max_y:.1f}]")
    lines.append(f"  合并后外接矩形: left={min_x:.1f}, top={min_y:.1f}, right={max_x:.1f}, bottom={max_y:.1f}")
    lines.append(f"  外接矩形宽度: bbox_w = {max_x:.1f} - {min_x:.1f} = {bbox_w:.1f}")
    lines.append(f"  外接矩形高度: bbox_h = {max_y:.1f} - {min_y:.1f} = {bbox_h:.1f}")

    # 标签判断
    sample_start_ms = sample_timestamps[0] * 1000
    sample_end_ms = sample_timestamps[-1] * 1000
    label = 0
    for s_ms, e_ms in intervals_ms:
        if s_ms >= sample_start_ms and e_ms <= sample_end_ms:
            label = 1
            lines.append(f"标签: FALL (跌倒区间 {s_ms:.0f}-{e_ms:.0f}ms 完全在样本范围内)")
            break
        elif s_ms <= sample_end_ms and e_ms >= sample_start_ms:
            lines.append(f"标签: NOFALL (与跌倒区间 {s_ms:.0f}-{e_ms:.0f}ms 有重叠)")

    lines.append("")

    # 计算velocities (使用归一化后的positions计算)
    velocities = []
    for i in range(1, len(positions_seq)):
        dt = sample_timestamps[i] - sample_timestamps[i-1]
        if dt > 0:
            # 特征34-67使用特征0-33的归一化坐标差来计算速度
            v = (positions_norm[i] - positions_norm[i-1]) / dt
        else:
            v = np.zeros(34)
        velocities.append(v)
    velocities_seq = np.array(velocities) if velocities else np.zeros((0, 34))

    # 计算accelerations (使用velocities的差来计算)
    accelerations = []
    for i in range(1, len(velocities_seq)):
        dt_prev = sample_timestamps[i] - sample_timestamps[i-1]
        dt_curr = sample_timestamps[i+1] - sample_timestamps[i]
        dt_avg = (dt_prev + dt_curr) / 2 if (dt_prev + dt_curr) > 0 else 1
        if dt_avg > 0:
            # 特征68-101使用特征34-67的速度差来计算加速度
            a = (velocities_seq[i] - velocities_seq[i-1]) / dt_avg
        else:
            a = np.zeros(34)
        accelerations.append(a)
    accelerations_seq = np.array(accelerations) if accelerations else np.zeros((0, 34))

    # 对velocities和accelerations进行min-max归一化（与step6一致）
    if len(velocities_seq) > 0:
        vel_min = np.min(velocities_seq)
        vel_max = np.max(velocities_seq)
        if vel_max != vel_min:
            velocities_normed = (velocities_seq - vel_min) / (vel_max - vel_min)
        else:
            velocities_normed = np.zeros_like(velocities_seq)
    else:
        vel_min = vel_max = 0
        velocities_normed = np.zeros((1, 34))

    if len(accelerations_seq) > 0:
        acc_min = np.min(accelerations_seq)
        acc_max = np.max(accelerations_seq)
        if acc_max != acc_min:
            accelerations_normed = (accelerations_seq - acc_min) / (acc_max - acc_min)
        else:
            accelerations_normed = np.zeros_like(accelerations_seq)
    else:
        acc_min = acc_max = 0
        accelerations_normed = np.zeros((1, 34))

    # 计算relative_positions (减去hip_center后，再基于外接矩形归一化)
    rel_positions = []
    for positions in positions_seq:
        left_hip = positions[22:24]
        right_hip = positions[24:26]
        hip_center = (left_hip + right_hip) / 2
        hip_center_flat = np.tile(hip_center, 17)
        rel = positions - hip_center_flat
        # 基于外接矩形归一化
        rel[0::2] /= bbox_w if bbox_w > 0 else 1
        rel[1::2] /= bbox_h if bbox_h > 0 else 1
        rel_positions.append(rel)
    rel_positions_seq = np.array(rel_positions)

    spine_leg_angles = [compute_spine_leg_angle(p) for p in positions_seq]
    hip_height_changes = np.array([(((positions_norm[i][23] + positions_norm[i][25]) / 2) -
                                     ((positions_norm[i-1][23] + positions_norm[i-1][25]) / 2))
                                    for i in range(1, len(positions_norm))])
    body_orientations = [compute_body_orientation(p) for p in positions_seq]

    # 计算每帧的bbox宽高
    bbox_widths = []
    bbox_heights = []
    for i in range(len(positions_seq)):
        frame_x = positions_seq[i, 0::2]
        frame_y = positions_seq[i, 1::2]
        if np.sum(frame_x) > 0 or np.sum(frame_y) > 0:
            w = (np.max(frame_x) - np.min(frame_x)) / (bbox_w if bbox_w > 0 else 1)
            h = (np.max(frame_y) - np.min(frame_y)) / (bbox_h if bbox_h > 0 else 1)
        else:
            w, h = 0, 0
        bbox_widths.append(w)
        bbox_heights.append(h)

    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    bbox_ratios = np.divide(
        bbox_widths,
        bbox_heights,
        out=np.zeros_like(bbox_widths),
        where=bbox_heights != 0
    )
    bbox_areas = bbox_widths * bbox_heights

    lines.append("-" * 60)
    lines.append("特征0-33 (positions 外接矩形相对坐标归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  归一化方式: (x - min_x) / bbox_w, (y - min_y) / bbox_h")
    for frame_idx in range(11):
        lines.append(f"  --- Frame {frame_idx} ---")
        for i in range(17):
            kp = keypoint_names[i]
            x_raw = positions_seq[frame_idx][i*2]
            y_raw = positions_seq[frame_idx][i*2+1]
            x_norm = positions_norm[frame_idx][i*2]
            y_norm = positions_norm[frame_idx][i*2+1]
            lines.append(f"    feat_{i*2}: {kp}_x: raw={x_raw:.2f}, norm={x_norm:.6f}")
            lines.append(f"    feat_{i*2+1}: {kp}_y: raw={y_raw:.2f}, norm={y_norm:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征34-43 (采样间隔): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 时间差 (秒)")
    lines.append("  计算方式: sample_timestamps[i] - sample_timestamps[i-1]")
    for frame_idx in range(11):
        if frame_idx == 0:
            lines.append(f"  feat_34: frame_{frame_idx} = 0 (无前一帧)")
        else:
            dt = sample_timestamps[frame_idx] - sample_timestamps[frame_idx - 1]
            lines.append(f"  feat_34: frame_{frame_idx} = {sample_timestamps[frame_idx]:.6f} - {sample_timestamps[frame_idx-1]:.6f} = {dt:.6f}s")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征44-63 (bbox归一化宽高): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 归一化宽高")
    lines.append("  计算方式: frame_bbox_w / bbox_w, frame_bbox_h / bbox_h")
    for frame_idx in range(10):
        source_frame = frame_idx + 1
        lines.append(f"  feat_{44 + frame_idx*2}: frame_{source_frame}_w = {bbox_widths[source_frame]:.6f}")
        lines.append(f"  feat_{45 + frame_idx*2}: frame_{source_frame}_h = {bbox_heights[source_frame]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征64-73 (bbox宽高比): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 宽高比")
    lines.append("  计算方式: frame_bbox_w / frame_bbox_h")
    for frame_idx in range(10):
        source_frame = frame_idx + 1
        lines.append(f"  feat_{64 + frame_idx}: frame_{source_frame}_ratio = {bbox_widths[source_frame]:.6f} / {bbox_heights[source_frame]:.6f} = {bbox_ratios[source_frame]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征74-83 (bbox宽高积): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 宽高积")
    lines.append("  计算方式: frame_bbox_w * frame_bbox_h")
    for frame_idx in range(10):
        source_frame = frame_idx + 1
        lines.append(f"  feat_{74 + frame_idx}: frame_{source_frame}_area = {bbox_widths[source_frame]:.6f} * {bbox_heights[source_frame]:.6f} = {bbox_areas[source_frame]:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征84-117 (relative_positions 外接矩形相对坐标归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 相对坐标 (基于外接矩形)")
    lines.append("  计算方式: (point - hip_center) / bbox_w, (point - hip_center) / bbox_h")
    for frame_idx in range(1, 11):
        lines.append(f"  --- Frame {frame_idx} ---")
        left_hip = positions_seq[frame_idx][22:24]
        right_hip = positions_seq[frame_idx][24:26]
        hip_center = (left_hip + right_hip) / 2
        for i in range(17):
            kp = keypoint_names[i]
            raw_x = positions_seq[frame_idx][i*2]
            raw_y = positions_seq[frame_idx][i*2+1]
            rx = rel_positions_seq[frame_idx][i*2]
            ry = rel_positions_seq[frame_idx][i*2+1]
            lines.append(f"    feat_{66+i*2}: rel_pos_{kp}_x:")
            lines.append(f"      原始坐标: {raw_x:.2f}, hip_center_x: {hip_center[0]:.2f}")
            lines.append(f"      相对坐标: {raw_x:.2f} - {hip_center[0]:.2f} = {raw_x - hip_center[0]:.2f}")
            lines.append(f"      归一化: {raw_x - hip_center[0]:.2f} / {bbox_w:.1f} = {rx:.6f}")
            lines.append(f"    feat_{66+i*2+1}: rel_pos_{kp}_y:")
            lines.append(f"      原始坐标: {raw_y:.2f}, hip_center_y: {hip_center[1]:.2f}")
            lines.append(f"      相对坐标: {raw_y:.2f} - {hip_center[1]:.2f} = {raw_y - hip_center[1]:.2f}")
            lines.append(f"      归一化: {raw_y - hip_center[1]:.2f} / {bbox_h:.1f} = {ry:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征118 (spine_leg_angle): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 角度值 (无坐标类型)")
    for frame_idx in range(1, 11):
        lines.append(f"  feat_118: frame_{frame_idx} = {spine_leg_angles[frame_idx]:.6f} rad")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征119 (hip_height_change 外接矩形归一化): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 高度差值 (y方向)")
    lines.append("  归一化方式: (y差值) / bbox_h")
    for frame_idx in range(1, 11):
        if frame_idx == 1:
            lines.append(f"  feat_119: frame_{frame_idx} = 0 (无前一帧)")
        else:
            h_change = hip_height_changes[frame_idx-1] if frame_idx-1 < len(hip_height_changes) else 0
            curr_hip_center_y = (positions_norm[frame_idx][23] + positions_norm[frame_idx][25]) / 2
            prev_hip_center_y = (positions_norm[frame_idx-1][23] + positions_norm[frame_idx-1][25]) / 2
            lines.append(f"  feat_119: frame_{frame_idx} = ({curr_hip_center_y:.4f} - {prev_hip_center_y:.4f}) = {h_change:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("特征120 (body_orientation): 所有11帧")
    lines.append("-" * 60)
    lines.append("  坐标类型: 角度值 (无坐标类型)")
    for frame_idx in range(1, 11):
        lines.append(f"  feat_120: frame_{frame_idx} = {body_orientations[frame_idx]:.6f} rad")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def extract_samples(frame_dir: Path, detections: np.ndarray, timestamps: list,
                    intervals_ms: list, sample_frames: int = 11) -> list:
    """提取所有样本 - 每个样本的11帧间隔独立随机从配置中选取"""
    import random
    total_frames = len(timestamps)
    logger.info(f"  提取样本: 总帧数={total_frames}")

    # 间隔配置 [250, 300, 350, 400, 450, 500] ms
    interval_choices_ms = [250, 300, 350, 400, 450, 500]

    samples = []

    frame_idx = 0
    while True:
        # 为这11帧生成随机间隔
        random_intervals = [random.choice(interval_choices_ms) for _ in range(sample_frames)]

        # 计算累积时间戳
        sample_timestamps = []
        start_time = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
        sample_timestamps.append(start_time)

        current_time = start_time
        for i in range(1, sample_frames):
            current_time += random_intervals[i] / 1000.0
            sample_timestamps.append(current_time)

        # 检查最后一帧是否在视频范围内
        if sample_timestamps[-1] >= timestamps[total_frames - 1]:
            break

        # 读取每帧的关键点
        sample_detections = []
        for seq_idx in range(frame_idx, frame_idx + sample_frames):
            if seq_idx < len(detections):
                sample_detections.append(detections[seq_idx])
            else:
                sample_detections.append({'keypoints': np.zeros((17, 3)), 'has_person': False})

        # 检查11帧中是否存在无人帧（x=[0.0, 0.0], y=[0.0, 0.0]），如果有则跳过此样本
        has_no_person = False
        for det in sample_detections:
            keypoints = det.get('keypoints', np.zeros((17, 3)))
            if isinstance(keypoints, np.ndarray):
                # 检查所有17个点的x坐标是否全为0
                kp_x = keypoints[:17, 0]
                if np.all(kp_x == 0):
                    has_no_person = True
                    break

        if has_no_person:
            frame_idx += 1
            continue

        sample = {
            'interval_ms': random_intervals,
            'start_frame': frame_idx,
            'timestamps': sample_timestamps,
            'detections': sample_detections
        }
        samples.append(sample)

        frame_idx += 1

    return samples


def _extract_single_sample(args) -> dict | None:
    """从单个起始帧提取单个样本（用于并行）"""
    frame_idx, detections, timestamps, sample_frames, interval_choices_ms = args
    import random

    total_frames = len(timestamps)

    # 为这11帧生成随机间隔
    random_intervals = [random.choice(interval_choices_ms) for _ in range(sample_frames)]

    # 计算累积时间戳
    sample_timestamps = []
    start_time = timestamps[frame_idx] if frame_idx < len(timestamps) else 0
    sample_timestamps.append(start_time)

    current_time = start_time
    for i in range(1, sample_frames):
        current_time += random_intervals[i] / 1000.0
        sample_timestamps.append(current_time)

    # 检查最后一帧是否在视频范围内
    if sample_timestamps[-1] >= timestamps[total_frames - 1]:
        return None

    # 读取每帧的关键点
    sample_detections = []
    for seq_idx in range(frame_idx, frame_idx + sample_frames):
        if seq_idx < len(detections):
            sample_detections.append(detections[seq_idx])
        else:
            sample_detections.append({'keypoints': np.zeros((17, 3)), 'has_person': False})

    # 检查11帧中是否存在无人帧（x=[0.0, 0.0], y=[0.0, 0.0]），如果有则跳过此样本
    has_no_person = False
    for det in sample_detections:
        keypoints = det.get('keypoints', np.zeros((17, 3)))
        if isinstance(keypoints, np.ndarray):
            kp_x = keypoints[:17, 0]
            if np.all(kp_x == 0):
                has_no_person = True
                break

    if has_no_person:
        return None

    return {
        'interval_ms': random_intervals,
        'start_frame': frame_idx,
        'timestamps': sample_timestamps,
        'detections': sample_detections
    }


def extract_samples_parallel(frame_dir: Path, detections: np.ndarray, timestamps: list,
                            intervals_ms: list, sample_frames: int = 11,
                            num_workers: int = None) -> list:
    """使用多进程并行提取所有样本"""
    import random

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    total_frames = len(timestamps)
    logger.info(f"  [并行] 提取样本: 总帧数={total_frames}, 工作进程数={num_workers}")

    # 间隔配置 [250, 300, 350, 400, 450, 500] ms
    interval_choices_ms = [250, 300, 350, 400, 450, 500]

    # 构建每个起始帧的处理参数
    frame_args = []
    for frame_idx in range(total_frames):
        frame_args.append((
            frame_idx, detections, timestamps, sample_frames, interval_choices_ms
        ))

    # 如果禁用了内部并行，直接使用单进程顺序处理
    if _DISABLE_INTERNAL_PARALLELISM:
        samples = []
        for arg in frame_args:
            result = _extract_single_sample(arg)
            if result is not None:
                samples.append(result)
        samples.sort(key=lambda x: x['start_frame'])
        logger.info(f"  [单进程] 提取样本数: {len(samples)}")
        return samples

    # 使用进程池并行处理
    samples = []
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(_extract_single_sample, arg): arg[0] for arg in frame_args}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    samples.append(result)
            except Exception as e:
                frame_idx = futures[future]
                logger.warning(f"  帧{frame_idx}处理异常: {e}")

    # 按start_frame排序保持原有顺序
    samples.sort(key=lambda x: x['start_frame'])

    logger.info(f"  [并行] 提取样本数: {len(samples)}")
    return samples


def clip_video_from_frames(frame_paths: list, output_path: Path, fps: float = 20.0):
    """将帧序列合成为视频"""
    import cv2
    if not frame_paths:
        return False

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        return False

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            out.write(frame)

    out.release()
    return True


def create_fall_video_clips(
    round_dir: Path,
    segment_output_dir: Path,
    samples: list,
    round_idx: int,
    timestamps: list,
    video_path: Path,
    fps: float = 20.0
):
    """为跌倒样本创建视频剪辑"""
    clips_dir = segment_output_dir / "fall_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    fall_clips_paths = []

    # 获取round_dir中所有帧文件，按文件名排序
    frame_files = sorted(round_dir.glob('*.jpg'))
    if not frame_files:
        logger.warning(f"  警告: {round_dir} 中没有找到帧文件")
        return fall_clips_paths

    for i, sample in enumerate(samples):
        if sample.get('label', 0) != 1:
            continue

        start_frame = sample['start_frame']
        sample_timestamps = sample['timestamps']
        start_time = sample_timestamps[0]
        end_time = sample_timestamps[-1]

        # 根据start_frame直接获取连续的11帧
        clip_frame_paths = []
        for offset in range(11):
            frame_idx = start_frame + offset
            if frame_idx < len(frame_files):
                clip_frame_paths.append(frame_files[frame_idx])

        if len(clip_frame_paths) >= 2:
            output_path = clips_dir / f"round{round_idx}_fall_{i}_T{start_time:.2f}-{end_time:.2f}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            first_frame = cv2.imread(str(clip_frame_paths[0]))
            if first_frame is None:
                continue
            h, w = first_frame.shape[:2]
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            for frame_path in clip_frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    out.write(frame)
            out.release()
            fall_clips_paths.append(str(output_path))

    logger.info(f"  跌倒视频剪辑: {len(fall_clips_paths)} 个 -> {clips_dir}")
    return fall_clips_paths


def process_single_round(
    round_dir: Path,
    round_idx: int,
    pose_results_path: Path,
    annotation_path: Path,
    output_dir: Path,
    video_path: Path = None
) -> Tuple[str, List[Dict]]:
    """处理单个round的数据"""
    logger.info(f"\n{'='*50}")
    logger.info(f"处理 Round {round_idx}: {round_dir.name}")
    logger.info(f"{'='*50}")

    segment_output_dir = output_dir / f"round_{round_idx}_output"
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    detections = load_pose_results(pose_results_path)
    timestamps = load_frame_timestamps(round_dir)

    logger.info(f"  总帧数: {len(detections)}")

    # 如果没有时间戳文件，使用旧的估算方式
    if timestamps is None:
        fps = 20.0
        total_frames = len(detections)
        timestamps = [i / fps for i in range(total_frames)]
        logger.info(f"  警告: 无时间戳文件，使用估算: FPS={fps}")

    frames = sorted(round_dir.glob('*.jpg'))
    kept_indices = clean_no_person_frames(frames, detections, gap_threshold=20)

    if len(kept_indices) == len(detections):
        logger.info("  没有需要清洗的区间")
        cleaned_dir = round_dir
        segments = [round_dir]
        kept_timestamps = timestamps
    else:
        cleaned_dir = segment_output_dir / f"{round_dir.name}_cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        # 保留原始帧号命名，但记录清洗后的时间戳
        # kept_indices 基于 detections 索引，直接用于 timestamps
        kept_timestamps = [timestamps[i] for i in kept_indices if i < len(timestamps)]

        for orig_idx in kept_indices:
            shutil.copy2(frames[orig_idx], cleaned_dir / frames[orig_idx].name)

        logger.info(f"  清洗完成: {len(kept_indices)} 帧 -> {cleaned_dir.name}")
        segments = [cleaned_dir]

        dst_ann = cleaned_dir / "annotation.txt"
        if not dst_ann.exists() and annotation_path.exists():
            shutil.copy2(annotation_path, dst_ann)

    parser = AnnotationParser()
    intervals_ms = parser.parse(annotation_path)
    logger.info(f"  跌倒区间: {[(s/1000, e/1000) for s, e in intervals_ms]}")

    sample_frames = 11

    for seg_dir in sorted(segments):
        logger.info(f"\n  处理片段: {seg_dir.name}")

        seg_detections = detections  # 使用原始检测结果（索引对应）

        # 构建清洗后帧的timestamp对应关系
        cleaned_timestamps = kept_timestamps

        samples = extract_samples_parallel(
            seg_dir, seg_detections, cleaned_timestamps, intervals_ms,
            sample_frames
        )
        logger.info(f"  提取样本数: {len(samples)}")

        if not samples:
            logger.info("  跳过 - 此段无样本")
            continue

        logger.info("  开始特征计算(并行)...")

        all_sample_features, all_relationships = compute_features_batch(samples, intervals_ms)
        all_labels = [int(feat[-1]) for feat in all_sample_features]

        # 将label添加到samples中
        for i, label in enumerate(all_labels):
            samples[i]['label'] = label

        # 创建跌倒视频剪辑
        if video_path and video_path.exists():
            create_fall_video_clips(
                seg_dir, segment_output_dir, samples,
                round_idx, cleaned_timestamps, video_path, fps=20.0
            )

        columns = [f'feat_{i}' for i in range(139)]
        columns.append('label')
        df = pd.DataFrame(all_sample_features, columns=columns)
        df['start_time'] = [s['timestamps'][0] for s in samples]
        df['end_time'] = [s['timestamps'][-1] for s in samples]
        df['fall_relationship'] = all_relationships

        features_csv = segment_output_dir / f"round_{round_idx}_features.csv"
        df.to_csv(features_csv, index=False)
        logger.info(f"  特征保存到: {features_csv}")

        return str(features_csv), samples

    return "", []


def _compute_sample_features_npz_wrapper(args):
    """包装compute_sample_features_for_npz用于并行"""
    sample_detections, sample_timestamps = args
    return compute_sample_features_for_npz(sample_detections, sample_timestamps)


def compute_sample_features_batch_for_npz(
    samples: list, max_workers: int = None
) -> np.ndarray:
    """
    使用进程池并行计算多个样本的NPZ特征
    返回: shape (n_samples, 10, 139)
    """
    # 如果禁用了内部并行，直接使用单进程顺序处理
    if _DISABLE_INTERNAL_PARALLELISM:
        results = []
        for s in samples:
            result = compute_sample_features_for_npz(s['detections'], s['timestamps'])
            results.append(result)
        return np.array(results)

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    # 准备参数列表
    args_list = [(s['detections'], s['timestamps']) for s in samples]

    results = [None] * len(samples)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_sample_features_npz_wrapper, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return np.array(results)



def generate_npz_and_details(video_frame_dir: Path, all_samples: List[Dict], intervals_ms: List[Tuple[float, float]]):
    """生成NPZ文件samples.npz、详细CSV和sample_details.txt"""
    from utils import AnnotationParser

    results_dir = video_frame_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 50)
    logger.info("生成NPZ和详细文档...")
    logger.info("=" * 50)

    # 计算所有样本的特征矩阵 X 和标签 y
    # X: shape (N, 10, 103)
    # y: shape (N,)
    n_samples = len(all_samples)
    n_frames = 10
    feat_dim = 103

    X = np.zeros((n_samples, n_frames, feat_dim))
    y = np.zeros(n_samples, dtype=np.int32)

    all_sample_data = []

    logger.info(f"  计算 {n_samples} 个样本的NPZ特征(并行)...")
    X = compute_sample_features_batch_for_npz(all_samples)

    # 准备y和all_sample_data
    y = np.array([s.get('label', 0) for s in all_samples], dtype=np.int32)
    all_sample_data = [
        {
            'sample_idx': i,
            'label': s.get('label', 0),
            'start_time': s['timestamps'][0],
            'end_time': s['timestamps'][-1],
            'intervals_ms': s.get('interval_ms', [])
        }
        for i, s in enumerate(all_samples)
    ]

    # 保存NPZ
    npz_path = results_dir / "samples.npz"
    np.savez(npz_path, X=X, y=y)
    logger.info(f"  NPZ已保存: {npz_path}")
    logger.info(f"    X.shape = {X.shape}")
    logger.info(f"    y.shape = {y.shape}")
    logger.info(f"    FALL (label=1) count: {np.sum(y == 1)}")
    logger.info(f"    NOFALL (label=0) count: {np.sum(y == 0)}")

    # 随机抽取50个样本 (25个跌倒 + 25个非跌倒)
    fall_indices = [i for i, s in enumerate(all_sample_data) if s['label'] == 1]
    nofall_indices = [i for i, s in enumerate(all_sample_data) if s['label'] == 0]

    n_sample = min(25, len(fall_indices), len(nofall_indices))
    selected_fall = random.sample(fall_indices, n_sample) if fall_indices else []
    selected_nofall = random.sample(nofall_indices, n_sample) if nofall_indices else []
    selected_indices = selected_fall + selected_nofall
    random.shuffle(selected_indices)

    logger.info(f"\n  生成50个随机抽样样本的详细计算过程...")
    logger.info(f"    跌倒样本: {len(selected_fall)} 个")
    logger.info(f"    非跌倒样本: {len(selected_nofall)} 个")

    # 生成sample_details.txt
    details_path = results_dir / "sample_details.txt"
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("跌倒检测训练数据 - 样本详细计算过程\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总样本数: {n_samples}\n")
        f.write(f"跌倒样本数: {len(fall_indices)}\n")
        f.write(f"非跌倒样本数: {len(nofall_indices)}\n")
        f.write(f"抽样数: {len(selected_indices)} (25 FALL + 25 NOFALL)\n\n")

        for idx in selected_indices:
            sample = all_samples[idx]
            details = compute_sample_debug_details(
                sample['detections'],
                sample['timestamps'],
                idx,
                intervals_ms
            )
            f.write(details)

    logger.info(f"  详细计算文档已保存: {details_path}")

    # 生成samples_with_details.csv (带可视化路径)
    csv_data = []
    for i, sample in enumerate(all_samples):
        start_time = sample['timestamps'][0]
        end_time = sample['timestamps'][-1]

        csv_data.append({
            'sample_idx': i,
            'label': sample.get('label', 0),
            'start_time': start_time,
            'end_time': end_time,
            'label_name': 'FALL' if sample.get('label', 0) == 1 else 'NOFALL',
            'num_frames': len(sample['detections'])
        })

    df_details = pd.DataFrame(csv_data)
    csv_path = results_dir / "samples_with_details.csv"
    df_details.to_csv(csv_path, index=False)
    logger.info(f"  样本信息CSV已保存: {csv_path}")

    logger.info("\n" + "=" * 50)
    logger.info("NPZ和详细文档生成完成!")
    logger.info("=" * 50)



def run_pipeline(
    video_name: str,
    video_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float = 20.0,
    num_rounds: int = 6,
    skip_existing: bool = True,
    skip_pose_detection: bool = True
) -> dict:
    """运行完整流程

    Args:
        skip_pose_detection: True则跳过pose检测，直接使用现有的pose_results.npz
    """
    from steps.step2_pose_detection import PoseDetector

    video_path = video_dir / f"{video_name}.mp4"
    annotation_path = annotation_dir / video_name / "annotation.txt"
    video_frame_dir = frame_output_dir / video_name

    logger.info(f"视频路径: {video_path}")
    logger.info(f"帧输出目录: {video_frame_dir}")
    logger.info(f"标注路径: {annotation_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    parser = AnnotationParser()
    intervals_ms = parser.parse(annotation_path)
    logger.info(f"  跌倒区间: {[(s/1000, e/1000) for s, e in intervals_ms]}")

    results = {}
    all_features_dfs = []
    all_samples = []

    for round_idx in tqdm(range(1, num_rounds + 1), desc=f"  Round", leave=False):
        round_dir = video_frame_dir / f"round_{round_idx}"
        pose_results_path = round_dir / "pose_results.npz"

        if not round_dir.exists():
            logger.info(f"\n[Round {round_idx}] 目录不存在，跳过")
            continue

        if not pose_results_path.exists():
            if skip_pose_detection:
                logger.info(f"\n[Round {round_idx}] 跳过Pose检测（使用现有结果）")
                continue
            else:
                logger.info(f"\n[Round {round_idx}] 开始Pose检测...")
                detector = PoseDetector(model_path)
                detector.detect_folder(round_dir, pose_results_path, force=True)
        else:
            logger.info(f"\n[Round {round_idx}] 跳过Pose检测，已有结果")

        output_dir = video_frame_dir / "results"
        output_dir.mkdir(parents=True, exist_ok=True)

        result, samples = process_single_round(
            round_dir,
            round_idx,
            pose_results_path,
            annotation_path,
            output_dir,
            video_path=video_path
        )

        if result and Path(result).exists():
            df = pd.read_csv(result)
            if len(df) > 0:
                all_features_dfs.append(df)
            results[round_idx] = result

        if samples:
            all_samples.extend(samples)

    if all_features_dfs:
        all_features_df = pd.concat(all_features_dfs, ignore_index=True)
        final_output = video_frame_dir / "results" / "all_features.csv"
        all_features_df.to_csv(final_output, index=False)

        logger.info("\n" + "=" * 50)
        logger.info(f"流程完成! 最终特征文件: {final_output}")
        logger.info(f"总样本数: {len(all_features_df)}")
        if 'label' in all_features_df.columns:
            logger.info(f"跌倒样本数: {(all_features_df['label'] > 0).sum():.0f}")
        logger.info("=" * 50)

        results['final'] = str(final_output)

        # 生成NPZ文件、详细CSV和sample_details.txt
        if all_samples:
            generate_npz_and_details(video_frame_dir, all_samples, intervals_ms)

    return results


def _process_video_wrapper(args):
    """包装process_single_video用于并行"""
    video_path, video_name, video_frame_dir, annotation_dir, frame_output_dir, model_path, fps, num_rounds, skip_existing = args
    return process_single_video(
        video_path, video_name, video_frame_dir,
        annotation_dir, frame_output_dir, model_path,
        fps, num_rounds, skip_existing
    )


def process_single_video(
    video_path: Path,
    video_name: str,
    video_frame_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float,
    num_rounds: int,
    skip_existing: bool
) -> dict:
    """处理单个视频 - 供并行调用"""
    result = {
        'video_name': video_name,
        'features_df': None,
        'samples': [],
        'fall': 0,
        'nofall': 0,
        'error': None
    }

    try:
        annotation_path = annotation_dir / video_name / "annotation.txt"
        if not annotation_path.exists():
            result['error'] = f"标注文件不存在: {annotation_path}"
            return result

        parser = AnnotationParser()
        intervals_ms = parser.parse(annotation_path)

        # Step 1: 切帧
        from steps.step1_video_split import extract_random_intervals
        extract_random_intervals(video_path, video_frame_dir, force=not skip_existing, num_rounds=num_rounds)

        # Step 2: Pose检测 (round级并行)
        from steps.step2_pose_detection import PoseDetector
        args_list = [(video_frame_dir / f"round_{i}", i, model_path, skip_existing) for i in range(1, num_rounds + 1)]

        if _DISABLE_INTERNAL_PARALLELISM or _should_disable_gpu_parallelism():
            # 单卡场景顺序处理，避免多个进程同时抢占同一块GPU
            for args in args_list:
                _detect_round_wrapper(args)
        else:
            with ProcessPoolExecutor(max_workers=num_rounds) as executor:
                futures = [executor.submit(_detect_round_wrapper, args) for args in args_list]
                for future in as_completed(futures):
                    future.result()  # 等待完成

        # Step 3: 特征计算
        video_features_dfs = []
        video_samples = []

        for round_idx in range(1, num_rounds + 1):
            round_dir = video_frame_dir / f"round_{round_idx}"
            pose_results_path = round_dir / "pose_results.npz"
            output_dir = video_frame_dir / "results"
            output_dir.mkdir(parents=True, exist_ok=True)

            if not round_dir.exists() or not pose_results_path.exists():
                continue

            feat_result, samples = process_single_round(
                round_dir, round_idx, pose_results_path,
                annotation_path, output_dir, video_path=video_path
            )

            if feat_result and Path(feat_result).exists():
                df = pd.read_csv(feat_result)
                if len(df) > 0:
                    video_features_dfs.append(df)
                    video_samples.extend(samples)

        if video_features_dfs:
            video_df = pd.concat(video_features_dfs, ignore_index=True)
            result['features_df'] = video_df
            result['samples'] = video_samples
            if 'label' in video_df.columns:
                result['fall'] = int((video_df['label'] > 0).sum())
                result['nofall'] = int((video_df['label'] == 0).sum())

    except Exception as e:
        import traceback
        traceback.print_exc()
        result['error'] = str(e)

    return result


def run_batch_pipeline(
    video_dir: Path,
    annotation_dir: Path,
    frame_output_dir: Path,
    model_path: Path,
    fps: float = 20.0,
    num_rounds: int = 6,
    skip_existing: bool = True,
    max_videos: int = None,
    num_workers: int = 1,  # 并行视频数
    offset: int = 0  # 视频列表起始偏移量
) -> dict:
    """批量运行完整流程 - 支持视频级并行"""
    import config

    # 获取所有视频
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(video_dir.glob(f"**/{ext}"))

    if max_videos:
        video_files = video_files[offset:offset+max_videos]
    elif offset:
        video_files = video_files[offset:]

    logger.info(f"找到 {len(video_files)} 个视频，启用了 {num_workers} 个并行worker")

    all_video_features = []
    all_video_samples = []
    total_fall = 0
    total_nofall = 0
    errors = []

    # 视频级并行
    video_args = [
        (vp, vp.stem, frame_output_dir / vp.stem, annotation_dir, frame_output_dir, model_path, fps, num_rounds, skip_existing)
        for vp in video_files
    ]

    # 使用num_workers参数进行视频级并行
    with tqdm(total=len(video_files), desc="处理视频") as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_video_wrapper, va): va for va in video_args}
            for future in as_completed(futures):
                res = future.result()
                pbar.update(1)

                if res['error']:
                    errors.append(f"{res['video_name']}: {res['error']}")

                if res['features_df'] is not None:
                    all_video_features.append(res['features_df'])
                    all_video_samples.extend(res['samples'])
                    total_fall += res['fall']
                    total_nofall += res['nofall']

                logger.info(f"  {res['video_name']} 完成: {len(res['samples'])} 样本")
                # 输出进度标记，供run_parallel.py解析
                print(f"[PROGRESS] video_completed: {res['video_name']}", flush=True)

    # 汇总所有视频结果
    if all_video_features:
        final_df = pd.concat(all_video_features, ignore_index=True)
        final_output = frame_output_dir / "all_features.csv"
        final_df.to_csv(final_output, index=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"批量处理完成!")
        logger.info(f"总样本数: {len(final_df)}")
        logger.info(f"跌倒样本数: {total_fall}")
        logger.info(f"非跌倒样本数: {total_nofall}")
        logger.info(f"最终特征文件: {final_output}")
        if errors:
            logger.info(f"失败视频: {len(errors)}")
        logger.info(f"{'='*60}")

        # 生成NPZ
        if all_video_samples:
            generate_npz_and_details(frame_output_dir, all_video_samples, [])

        return {'final': str(final_output), 'total_samples': len(final_df), 'fall': total_fall, 'nofall': total_nofall}

    return {}


if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(description="跌倒检测训练数据生成流程")
    parser.add_argument("--video-name", type=str, help="视频名称（单视频模式）")
    parser.add_argument("--batch", action="store_true", help="批量处理所有视频")
    parser.add_argument("--max-videos", type=int, default=None, help="最多处理视频数")
    parser.add_argument("--fps", type=float, default=config.VIDEO_FPS, help="视频帧率")
    parser.add_argument("--rounds", type=int, default=6, help="切帧轮数")
    parser.add_argument("--recompute", action="store_true", help="强制重新计算（包括切帧和pose检测）")
    parser.add_argument("--reuse-frames", action="store_true", help="复用现有切帧和pose结果，只重新计算特征")
    parser.add_argument("--workers", type=int, default=1, help="并行视频处理数 (A100建议4-8)")
    parser.add_argument("--offset", type=int, default=0, help="视频列表起始偏移量")
    parser.add_argument("--disable-internal-parallel", action="store_true",
                        help="禁用内部并行（当外部已提供进程隔离时使用）")

    args = parser.parse_args()

    # 设置全局并行标志
    if args.disable_internal_parallel:
        logger.info("[配置] 内部并行已禁用")

    if args.batch or args.video_name is None:
        # 批量模式 - 使用config中的视频目录
        results = run_batch_pipeline(
            video_dir=config.VIDEO_DIR,
            annotation_dir=config.ANNOTATION_DIR,
            frame_output_dir=config.FRAME_OUTPUT_DIR,
            model_path=config.MODEL_PATH,
            fps=args.fps,
            num_rounds=args.rounds,
            skip_existing=not args.recompute,
            max_videos=args.max_videos,
            num_workers=args.workers,
            offset=args.offset
        )
    else:
        # 单视频模式
        results = run_pipeline(
        video_name=args.video_name,
        video_dir=config.VIDEO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        frame_output_dir=config.FRAME_OUTPUT_DIR,
        model_path=config.MODEL_PATH,
        fps=args.fps,
        num_rounds=args.rounds,
        skip_existing=not args.recompute,
        skip_pose_detection=args.reuse_frames
    )

    logger.info("\n结果汇总:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")
