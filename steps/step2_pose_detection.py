"""Step 2: 人体关键点检测模块"""
import json
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import config
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, handlers=[logging.NullHandler()])
logger = logging.getLogger(__name__)


def _process_batch_worker(args):
    """Worker进程：处理一个batch的检测"""
    batch_file_strs, model_path, keypoint_names, device = args
    import numpy as np
    from ultralytics import YOLO
    from pathlib import Path

    # 在worker进程中重新加载模型（每个进程独立的GPU内存）
    model = YOLO(model_path)
    model.to(device)

    # 转换回Path对象
    batch_files = [Path(f) for f in batch_file_strs]

    batch_results = []
    results = model(batch_files, verbose=False)

    for result in results:
        keypoints_array = np.zeros((17, 3), dtype=np.float32)
        keypoints_list = []

        if result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints.xy[0]
            conf = result.keypoints.conf[0] if result.keypoints.conf is not None else None

            for i in range(min(17, len(kp))):
                x, y = float(kp[i][0].item()), float(kp[i][1].item())
                c = float(conf[i].item()) if conf is not None else 1.0
                keypoints_array[i] = [x, y, c]
                keypoints_list.append({'x': x, 'y': y, 'name': keypoint_names[i]})

        batch_results.append({
            'frame': result.path.split('/')[-1].split('\\')[-1] if result.path else 'unknown.jpg',
            'keypoints': keypoints_array,
            'has_person': len(keypoints_list) > 0
        })

    return batch_results


class PoseDetector:
    """人体关键点检测器"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.keypoint_names = config.KEYPOINT_NAMES

    def load_model(self):
        """加载模型"""
        logger.info(f"[Step2] 加载模型: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect_frame(self, frame_path: Path) -> dict:
        """检测单帧"""
        import numpy as np

        results = self.model(frame_path, verbose=False)
        result = results[0]

        # 提取关键点
        keypoints_list = []
        keypoints_array = np.zeros((17, 3), dtype=np.float32)  # 17点 (x, y, conf)

        if result.keypoints is not None and len(result.keypoints) > 0:
            kp = result.keypoints.xy[0]  # 第一个人
            conf = result.keypoints.conf[0] if result.keypoints.conf is not None else None

            for i in range(min(17, len(kp))):
                x, y = float(kp[i][0].item()), float(kp[i][1].item())
                c = float(conf[i].item()) if conf is not None else 1.0
                keypoints_array[i] = [x, y, c]
                keypoints_list.append({
                    'x': x,
                    'y': y,
                    'name': self.keypoint_names[i]
                })

        has_person = len(keypoints_list) > 0

        return {
            'frame': f'{frame_path.stem}.jpg',  # 保持与原格式一致
            'keypoints': keypoints_array,
            'has_person': has_person
        }

    def detect_folder(self, frame_dir: Path, output_path: Path, force: bool = False, batch_size: int = 32, num_workers: int = None) -> Path:
        """
        检测整个文件夹的帧 (多进程并行批量推理)

        Args:
            frame_dir: 帧文件目录
            output_path: 输出路径
            force: 是否强制重新检测
            batch_size: 批量大小，A100建议32-64
            num_workers: 并行worker数量，默认使用CPU核心数
        """
        if not force and output_path.exists():
            logger.info(f"[Step2] 跳过检测，已有结果: {output_path}")
            return output_path

        frame_files = sorted(frame_dir.glob('*.jpg'))
        total_frames = len(frame_files)

        # 获取GPU device (检查CUDA是否可用)
        import os, tempfile
        debug_log_path = Path(tempfile.gettempdir()) / "step2_debug.log"
        debug_log = f"[Step2 DEBUG] frame_dir={frame_dir}\n"
        debug_log += f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','NOT_SET')}\n"
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            debug_log += f"  torch.cuda.is_available()={torch.cuda.is_available()}\n"
            debug_log += f"  torch.cuda.device_count()={torch.cuda.device_count()}\n"
            debug_log += f"  device={device}\n"
            debug_log += f"  _should_disable_gpu_parallelism() result: {__import__('main_pipeline', fromlist=['_should_disable_gpu_parallelism'])._should_disable_gpu_parallelism() if 'main_pipeline' in dir() else 'N/A'}\n"
            if torch.cuda.is_available():
                debug_log += f"  torch.cuda.current_device()={torch.cuda.current_device()}\n"
                debug_log += f"  torch.cuda.get_device_name(0)={torch.cuda.get_device_name(0)}\n"
        except Exception as ex:
            device = 'cpu'
            debug_log += f"  device detect exception: {ex}\n"

        # CUDA 场景下同一张卡不适合多 worker 同时推理，容易抢占失败
        if device == 'cuda':
            num_workers = 1

        debug_log += f"  after cuda check: num_workers={num_workers}, device={device}\n"

        # 自动确定worker数量
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)

        debug_log += f"  final: num_workers={num_workers}\n"
        with open(debug_log_path, "a", encoding="utf-8") as f:
            f.write(debug_log + "\n")

        logger.info(f"[Step2] 开始检测: {total_frames} 帧 (batch_size={batch_size}, num_workers={num_workers})")

        # 划分batch
        batches = []
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batches.append(frame_files[batch_start:batch_end])

        worker_args = [
            ([str(f) for f in batch], str(self.model_path), self.keypoint_names, device)
            for batch in batches
        ]

        all_results = []

        # CUDA 单卡场景：直接在主进程跑，避免 spawn 子进程初始化 CUDA 冲突
        if device == 'cuda' and num_workers == 1:
            debug_str = f"[Step2 CUDA path] device={device}, num_workers={num_workers}, batches={len(batches)}\n"
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(debug_str + "\n")
            logger.info(f"[Step2] 使用CUDA单进程路径 (device={device})")
            model = YOLO(str(self.model_path))
            debug_str2 = f"[Step2] after YOLO(), about to model.to(device={device})\n"
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write(debug_str2 + "\n")
            logger.info(f"[Step2] 调用 model.to({device}) ...")
            model.to(device)
            for i, batch in enumerate(batches):
                batch_results = []
                results = model(list(batch), verbose=False)
                for result in results:
                    keypoints_array = np.zeros((17, 3), dtype=np.float32)
                    keypoints_list = []
                    if result.keypoints is not None and len(result.keypoints) > 0:
                        kp = result.keypoints.xy[0]
                        conf = result.keypoints.conf[0] if result.keypoints.conf is not None else None
                        for j in range(min(17, len(kp))):
                            x, y = float(kp[j][0].item()), float(kp[j][1].item())
                            c = float(conf[j].item()) if conf is not None else 1.0
                            keypoints_array[j] = [x, y, c]
                            keypoints_list.append({'x': x, 'y': y, 'name': self.keypoint_names[j]})
                    batch_results.append({
                        'frame': result.path.split('/')[-1].split('\\')[-1] if result.path else 'unknown.jpg',
                        'keypoints': keypoints_array,
                        'has_person': len(keypoints_list) > 0
                    })
                all_results.extend(batch_results)
                processed = min((i + 1) * batch_size, total_frames)
                if processed % 100 == 0 or processed == total_frames:
                    logger.info(f"  已检测 {processed}/{total_frames} 帧")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers) as pool:
                for i, batch_results in enumerate(pool.imap(_process_batch_worker, worker_args)):
                    all_results.extend(batch_results)
                    processed = min((i + 1) * batch_size, total_frames)
                    if processed % 100 == 0 or processed == total_frames:
                        logger.info(f"  已检测 {processed}/{total_frames} 帧")

        # 保存结果为NPZ格式
        np.savez_compressed(output_path, results=all_results)

        logger.info(f"[Step2] 检测完成: {len(all_results)} 帧 -> {output_path}")
        return output_path


def run(video_name: str = config.VIDEO_NAME, force: bool = False) -> Path:
    """执行检测"""
    frame_dir = config.FRAME_OUTPUT_DIR / video_name
    detection_path = Path(__file__).parent.parent / f"{video_name}_detection.json"

    detector = PoseDetector(config.MODEL_PATH)
    return detector.detect_folder(frame_dir, detection_path, force=force)


if __name__ == '__main__':
    run()