#!/usr/bin/env python3
"""RK35xx 设备端 YOLOv8-Det 批量验收，输出可导回电脑的 JSON 报告。"""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from infer_on_device import letterbox, postprocess_det

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(root):
    images = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(dirpath, filename))
    return sorted(images)


def percentile(values, quantile):
    return round(float(np.percentile(np.asarray(values), quantile)), 3) if values else None


def run(args):
    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise RuntimeError('未安装 rknn-toolkit-lite2，无法执行设备验收') from exc

    images = collect_images(args.images)
    if not images:
        raise RuntimeError(f'验证目录中未找到图片：{args.images}')

    class_names = [name.strip() for name in args.classes.split(',') if name.strip()]
    runtime = RKNNLite(verbose=False)
    failures = []
    predictions = []
    latencies = []
    warmed_up = False

    try:
        ret = runtime.load_rknn(args.model)
        if ret != 0:
            raise RuntimeError(f'load_rknn 失败，返回码 {ret}')
        ret = runtime.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError(f'init_runtime 失败，返回码 {ret}')

        for index, image_path in enumerate(images):
            image = cv2.imread(image_path)
            if image is None:
                failures.append({'image': image_path, 'error': '无法解码'})
                continue

            image_rgb, scale, pad_x, pad_y = letterbox(image, args.width, args.height)
            input_tensor = np.expand_dims(image_rgb, axis=0)
            if not warmed_up:
                for _ in range(args.warmup):
                    runtime.inference(inputs=[input_tensor], data_format='nhwc')
                warmed_up = True
            started = time.perf_counter()
            outputs = runtime.inference(inputs=[input_tensor], data_format='nhwc')
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not outputs:
                failures.append({'image': image_path, 'error': '推理返回空结果'})
                continue

            _, _, detections = postprocess_det(
                outputs, image, scale, pad_x, pad_y,
                args.conf, args.iou, class_names,
                input_wh=(args.width, args.height),
            )
            latencies.append(elapsed_ms)
            predictions.append({
                'image': os.path.relpath(image_path, args.images).replace('\\', '/'),
                'image_size': [int(image.shape[1]), int(image.shape[0])],
                'latency_ms': round(elapsed_ms, 3),
                'detections': detections,
            })
            if args.progress and (index + 1) % args.progress == 0:
                print(f'[INFO] 已处理 {index + 1}/{len(images)}')
    finally:
        runtime.release()

    report = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'device': {
            'machine': platform.machine(),
            'platform': platform.platform(),
            'hostname': platform.node(),
        },
        'model': {
            'path': os.path.basename(args.model),
            'sha256': sha256_file(args.model),
            'type': 'yolov8_det',
            'input_size': [args.width, args.height],
        },
        'settings': {
            'confidence_threshold': args.conf,
            'iou_threshold': args.iou,
            'warmup_iterations': args.warmup,
            'class_names': class_names,
        },
        'summary': {
            'requested_images': len(images),
            'successful_images': len(predictions),
            'failed_images': len(failures),
            'latency_ms': {
                'mean': round(float(np.mean(latencies)), 3) if latencies else None,
                'p50': percentile(latencies, 50),
                'p95': percentile(latencies, 95),
                'min': round(min(latencies), 3) if latencies else None,
                'max': round(max(latencies), 3) if latencies else None,
            },
        },
        'predictions': predictions,
        'failures': failures,
    }

    with open(args.output, 'w', encoding='utf-8') as file_obj:
        json.dump(report, file_obj, ensure_ascii=False, indent=2)
    print(f'[INFO] 验收报告已保存：{args.output}')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RK35xx 设备端 YOLOv8-Det 批量验收')
    parser.add_argument('--model', required=True, help='RKNN 模型路径')
    parser.add_argument('--images', required=True, help='验证图片根目录')
    parser.add_argument('--output', default='device-validation.json', help='JSON 报告路径')
    parser.add_argument('--classes', default='', help='类别名，逗号分隔')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.001, help='导出预测的最低置信度')
    parser.add_argument('--iou', type=float, default=0.65)
    parser.add_argument('--warmup', type=int, default=3, help='计时前 NPU 预热次数')
    parser.add_argument('--progress', type=int, default=50, help='每 N 张输出一次进度，0 表示关闭')
    run(parser.parse_args())
