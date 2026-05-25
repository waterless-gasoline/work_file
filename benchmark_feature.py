"""基准测试脚本 - 测试特征计算性能"""
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main_pipeline import (
    load_pose_results, load_frame_timestamps, extract_samples_parallel,
    compute_features_batch, clean_no_person_frames
)
from utils import AnnotationParser

FRAME_OUTPUT_DIR = Path(r"D:\IPC\IPC_data_clip_photo\跌倒\data_positive_0512")
ANNOTATION_DIR = Path(r"D:\IPC\IPC_video_data_annotation\跌倒")

# 使用已有的帧目录（带完整round和pose数据）
VIDEO_LIST = [
    "0407add_IR_positive_0004",
    "0407add_IR_positive_0006",
]

def benchmark_video(video_name):
    """测试单个视频的特征计算时间"""
    video_frame_dir = FRAME_OUTPUT_DIR / video_name
    annotation_path = ANNOTATION_DIR / video_name / "annotation.txt"

    if not video_frame_dir.exists():
        return None, f"目录不存在: {video_frame_dir}"

    num_rounds = 6

    results = {
        'video_name': video_name,
        'round_times': [],
        'total_time': 0,
        'samples_count': 0,
        'error': None
    }

    total_start = time.time()

    try:
        parser = AnnotationParser()
        intervals_ms = parser.parse(annotation_path)

        for round_idx in range(1, num_rounds + 1):
            round_dir = video_frame_dir / f"round_{round_idx}"
            pose_results_path = round_dir / "pose_results.npz"

            if not round_dir.exists() or not pose_results_path.exists():
                continue

            round_start = time.time()
            detections = load_pose_results(pose_results_path)
            timestamps = load_frame_timestamps(round_dir)

            if timestamps is None:
                fps = 20.0
                timestamps = [i / fps for i in range(len(detections))]

            frames = sorted(round_dir.glob('*.jpg'))
            kept_indices = clean_no_person_frames(frames, detections, gap_threshold=20)
            kept_timestamps = [timestamps[i] for i in kept_indices if i < len(timestamps)]

            samples = extract_samples_parallel(
                round_dir, detections, kept_timestamps, intervals_ms, sample_frames=11
            )

            if samples:
                all_sample_features, all_relationships = compute_features_batch(samples, intervals_ms)
                results['samples_count'] += len(samples)

            round_time = time.time() - round_start
            results['round_times'].append(f"r{round_idx}:{round_time:.1f}s")

    except Exception as e:
        import traceback
        results['error'] = f"{e}\n{traceback.format_exc()}"

    results['total_time'] = time.time() - total_start
    return results

def main():
    print("=" * 60)
    print("特征计算基准测试")
    print("=" * 60)
    print()

    all_results = []

    for video_name in VIDEO_LIST:
        print(f"测试: {video_name}...", end=" ", flush=True)
        result = benchmark_video(video_name)

        if result is None:
            print("SKIP")
            continue

        if result['error']:
            print(f"ERROR")
            all_results.append({
                'video_name': video_name,
                'time': 0,
                'samples': 0,
                'error': result['error']
            })
            print(f"  错误: {result['error'][:200]}")
        else:
            print(f"{result['total_time']:.1f}s")
            all_results.append({
                'video_name': video_name,
                'time': result['total_time'],
                'samples': result['samples_count'],
                'error': None
            })
            print(f"  样本数: {result['samples_count']}")
            print(f"  各round: {', '.join(result['round_times'])}")

    print()
    print("=" * 60)
    print("汇总")
    print("=" * 60)

    if all_results:
        valid_results = [r for r in all_results if r['error'] is None]
        if valid_results:
            total_time = sum(r['time'] for r in valid_results)
            avg_time = total_time / len(valid_results)
            total_samples = sum(r['samples'] for r in valid_results)

            print(f"总视频数: {len(valid_results)}")
            print(f"总耗时: {total_time:.1f}s")
            print(f"平均每视频: {avg_time:.1f}s")
            print(f"总样本数: {total_samples}")
            print()
            print("各视频耗时:")
            for r in valid_results:
                print(f"  {r['video_name']}: {r['time']:.1f}s ({r['samples']}样本)")

        error_results = [r for r in all_results if r['error'] is not None]
        if error_results:
            print(f"\n失败视频: {len(error_results)}")
            for r in error_results:
                print(f"  {r['video_name']}: {r['error'][:100]}")

if __name__ == "__main__":
    main()