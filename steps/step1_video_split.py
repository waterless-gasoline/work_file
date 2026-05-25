"""Step 1: 视频切帧模块 - 随机间隔切帧"""
import cv2
import random
from pathlib import Path
from typing import Optional, List
import sys, os, logging
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# 简单日志配置
logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """视频切帧器 - 随机间隔切帧"""

    def __init__(self, video_path: Path, output_dir: Path):
        self.video_path = video_path
        self.output_dir = output_dir
        self.source_fps = None
        self.total_frames = None
        self.cap = None
        self.frame_intervals_ms = config.FRAME_INTERVALS_MS  # [250, 300, 350, 400, 450, 500]

    def get_video_info(self) -> tuple:
        """获取视频信息"""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.release()
        return self.source_fps, self.total_frames

    def extract_one_round(self, round_idx: int, force: bool = False, first_interval_used: Optional[set] = None, enforce_unique_first_interval: bool = False) -> tuple:
        """
        执行一轮切帧：随机间隔遍历整个视频

        Returns:
            (frame_count, interval_ms_list) - 本轮提取的帧数和使用的间隔列表
        """
        round_dir = self.output_dir / f"round_{round_idx}"
        if not force and round_dir.exists() and len(list(round_dir.glob('*.jpg'))) > 0:
            logger.info(f"[Step1] 跳过round_{round_idx}，已有数据")
            return -1, []

        round_dir.mkdir(parents=True, exist_ok=True)

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_source_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration_s = total_source_frames / source_fps

        logger.info(f"[Step1] Round {round_idx}: 视频时长={video_duration_s:.2f}s, 源FPS={source_fps}")

        # 将间隔从ms转为秒
        intervals_s = [ms / 1000.0 for ms in self.frame_intervals_ms]

        extracted_idx = 0
        current_time_s = 0.0
        interval_ms_list = []

        # 保存每帧的时间戳，用于后续样本提取
        frame_timestamps = []

        while current_time_s < video_duration_s:
            # 第一帧间隔在前6轮内不能重复
            if extracted_idx == 0 and enforce_unique_first_interval and first_interval_used is not None:
                candidate_intervals = [s for s in intervals_s if int(s * 1000) not in first_interval_used]
                if candidate_intervals:
                    interval_s = random.choice(candidate_intervals)
                else:
                    interval_s = random.choice(intervals_s)
            else:
                interval_s = random.choice(intervals_s)
            interval_ms_list.append(int(interval_s * 1000))

            # 移动到目标帧
            target_frame = int(current_time_s * source_fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

            ret, frame = self.cap.read()
            if not ret:
                break

            # 保存帧
            output_path = round_dir / f"frame_{extracted_idx:06d}.jpg"
            cv2.imwrite(str(output_path), frame)

            # 记录这帧的时间戳（秒）
            frame_timestamps.append(current_time_s)

            extracted_idx += 1

            # 前进到下一个时间点
            current_time_s += interval_s

        self.cap.release()

        # 保存时间戳文件
        import json
        timestamps_path = round_dir / "frame_timestamps.json"
        with open(timestamps_path, 'w') as f:
            json.dump(frame_timestamps, f)

        logger.info(f"[Step1] Round {round_idx}: 提取 {extracted_idx} 帧 -> {round_dir}")
        if interval_ms_list:
            logger.info(f"[Step1] Round {round_idx}: 首帧间隔={interval_ms_list[0]} ms")
        logger.info(f"[Step1] 时间戳保存到: {timestamps_path}")
        return extracted_idx, interval_ms_list

    def extract_frames(self, force: bool = False, num_rounds: int = 6) -> List[Path]:
        """
        执行多轮切帧

        Args:
            force: 是否强制重新切帧
            num_rounds: 轮数，默认6次
        """
        self.get_video_info()
        logger.info(f"[Step1] 开始切帧: {self.video_path.name}")
        logger.info(f"  源视频: FPS={self.source_fps}, 总帧数={self.total_frames}")
        logger.info(f"  随机间隔: {self.frame_intervals_ms} ms")
        logger.info(f"  切帧轮数: {num_rounds}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        first_interval_used = set()
        for round_idx in range(1, num_rounds + 1):
            logger.info(f"\n--- Round {round_idx}/{num_rounds} ---")
            enforce_unique_first_interval = round_idx <= 6
            frame_count, intervals = self.extract_one_round(
                round_idx,
                force,
                first_interval_used=first_interval_used,
                enforce_unique_first_interval=enforce_unique_first_interval,
            )
            if intervals:
                first_interval_used.add(intervals[0])
            results.append(frame_count)

        logger.info(f"\n[Step1] 切帧完成: 共 {num_rounds} 轮")
        for i, count in enumerate(results):
            logger.info(f"  Round {i+1}: {count} 帧")
        return results


def extract_random_intervals(video_path: Path, output_dir: Path, force: bool = False, num_rounds: int = 6) -> List[Path]:
    """为指定视频执行随机间隔切帧"""
    extractor = VideoFrameExtractor(video_path, output_dir)
    return extractor.extract_frames(force=force, num_rounds=num_rounds)


def run(video_name: str = config.VIDEO_NAME, force: bool = False, num_rounds: int = 6) -> List[Path]:
    """执行切帧"""
    video_path = config.VIDEO_DIR / f"{video_name}.mp4"
    output_dir = config.FRAME_OUTPUT_DIR / video_name

    extractor = VideoFrameExtractor(video_path, output_dir)
    return extractor.extract_frames(force=force, num_rounds=num_rounds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='视频切帧 - 随机间隔')
    parser.add_argument('--video', type=str, default=config.VIDEO_NAME, help='视频名')
    parser.add_argument('--force', action='store_true', help='强制重新切帧')
    parser.add_argument('--rounds', type=int, default=6, help='切帧轮数')
    args = parser.parse_args()

    run(args.video, args.force, args.rounds)
