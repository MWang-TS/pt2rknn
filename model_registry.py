"""
模型类型注册表
每个条目定义了模型的 RKNN 配置参数、接受的输入格式及校验规则
"""

MODEL_REGISTRY = {
    # -------------------------------------------------------
    # YOLOv8 系列 - 均支持 .pt / .onnx 输入
    # -------------------------------------------------------
    'yolov8_det': {
        'name': 'YOLOv8 目标检测 (Detection)',
        'short': 'YOLOv8-Det',
        'icon': '🎯',
        'description': '通用目标检测，输出边框+类别',
        'accepted_exts': ['pt', 'pth', 'onnx'],
        'source_type': 'pt_or_onnx',       # pt 会自动导出 onnx 再转 rknn
        'ultralytics_task': 'detect',       # 用于校验 PT 文件 task 字段
        'input_size_default': [640, 640],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'calibration_subdir': 'coco',
        'hint': '上传 YOLOv8/YOLOv5 等目标检测 .pt 或导出的 .onnx'
    },
    'yolov8_seg': {
        'name': 'YOLOv8 实例分割 (Segmentation)',
        'short': 'YOLOv8-Seg',
        'icon': '✂️',
        'description': '实例分割，输出边框+掩码',
        'accepted_exts': ['pt', 'pth', 'onnx'],
        'source_type': 'pt_or_onnx',
        'ultralytics_task': 'segment',
        'input_size_default': [640, 640],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'calibration_subdir': 'coco',
        'hint': '上传 yolov8n-seg.pt / yolov8m-seg.pt 或对应 .onnx'
    },
    'yolov8_pose': {
        'name': 'YOLOv8 姿态估计 (Pose)',
        'short': 'YOLOv8-Pose',
        'icon': '🧍',
        'description': '关键点检测，输出骨骼关节点坐标',
        'accepted_exts': ['pt', 'pth', 'onnx'],
        'source_type': 'pt_or_onnx',
        'ultralytics_task': 'pose',
        'input_size_default': [640, 640],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'calibration_subdir': 'coco',
        'hint': '上传 yolov8n-pose.pt / yolov8m-pose.pt 或对应 .onnx'
    },
    'yolov8_obb': {
        'name': 'YOLOv8 旋转目标检测 (OBB)',
        'short': 'YOLOv8-OBB',
        'icon': '🔄',
        'description': '旋转框目标检测，适合航拍/遥感场景',
        'accepted_exts': ['pt', 'pth', 'onnx'],
        'source_type': 'pt_or_onnx',
        'ultralytics_task': 'obb',
        'input_size_default': [1024, 1024],
        'mean_values': [[0, 0, 0]],
        'std_values': [[255, 255, 255]],
        'calibration_subdir': 'coco',
        'hint': '上传 yolov8n-obb.pt 或对应 .onnx（DOTA 数据集训练）'
    },

    # -------------------------------------------------------
    # 图像分类 / 人脸检测 - 仅接受 .onnx
    # -------------------------------------------------------
    'resnet': {
        'name': 'ResNet 图像分类 (Classification)',
        'short': 'ResNet',
        'icon': '🏷️',
        'description': 'ImageNet 图像分类，输出1000类概率',
        'accepted_exts': ['onnx'],
        'source_type': 'onnx_only',
        'ultralytics_task': None,
        'input_size_default': [224, 224],
        # 标准 ImageNet 归一化 (pixel 0-255 → RKNN 统一处理)
        'mean_values': [[123.675, 116.28, 103.53]],
        'std_values': [[58.395, 57.12, 57.375]],
        'calibration_subdir': 'imagenet',
        'hint': '上传 resnet50-v2-7.onnx 等来自 ONNX Model Zoo 或自训练的 .onnx'
    },
    'retinaface': {
        'name': 'RetinaFace 人脸检测',
        'short': 'RetinaFace',
        'icon': '😊',
        'description': '多任务人脸检测，输出人脸框+关键点',
        'accepted_exts': ['onnx'],
        'source_type': 'onnx_only',
        'ultralytics_task': None,
        'input_size_default': [640, 640],
        # CV2 BGR 格式使用的均值
        'mean_values': [[104, 117, 123]],
        'std_values': [[1, 1, 1]],
        'calibration_subdir': 'face',
        'hint': '上传 RetinaFace.onnx（需先用 tools/pytorch_retinaface 导出）'
    },
}


def get_model_types_meta():
    """返回供前端显示的简化列表（不暴露内部参数）"""
    result = []
    for key, cfg in MODEL_REGISTRY.items():
        result.append({
            'value': key,
            'name': cfg['name'],
            'short': cfg['short'],
            'icon': cfg['icon'],
            'description': cfg['description'],
            'accepted_exts': cfg['accepted_exts'],
            'source_type': cfg['source_type'],
            'input_size_default': cfg['input_size_default'],
            'hint': cfg['hint'],
            'calibration_subdir': cfg['calibration_subdir'],
        })
    return result


def validate_file_ext(model_type: str, filename: str) :
    """校验文件扩展名与模型类型是否匹配"""
    if model_type not in MODEL_REGISTRY:
        return False, f"未知模型类型：{model_type}"
    cfg = MODEL_REGISTRY[model_type]
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in cfg['accepted_exts']:
        accepted = ', '.join(f'.{e}' for e in cfg['accepted_exts'])
        return False, f"❌ {cfg['short']} 只接受 {accepted} 文件，您上传的是 .{ext}"
    return True, "ok"


def validate_pt_task(model_type: str, pt_path: str) :
    """对 .pt/.pth 文件，用 ultralytics 加载并校验 task"""
    cfg = MODEL_REGISTRY[model_type]
    expected_task = cfg.get('ultralytics_task')
    if expected_task is None:
        return True, "不需要 task 校验"
    try:
        from ultralytics import YOLO
        model = YOLO(pt_path)
        actual_task = getattr(model, 'task', None)
        if actual_task and actual_task != expected_task:
            task_map = {
                'detect': 'YOLOv8-Det 目标检测',
                'segment': 'YOLOv8-Seg 实例分割',
                'pose': 'YOLOv8-Pose 姿态估计',
                'obb': 'YOLOv8-OBB 旋转框检测',
                'classify': 'YOLOv8-Cls 分类',
            }
            actual_name = task_map.get(actual_task, actual_task)
            expected_name = task_map.get(expected_task, expected_task)
            return False, (
                f"❌ 模型类型不匹配！\n"
                f"   您选择了：{cfg['short']}（task={expected_task}）\n"
                f"   实际上传的是：{actual_name}（task={actual_task}）\n"
                f"   请重新选择正确的网络类型"
            )
        return True, f"✅ 模型校验通过（task={actual_task}）"
    except Exception as e:
        # 加载失败不阻止转换，只作提示
        return True, f"⚠️ 无法读取 task（{e}），将继续尝试转换"
