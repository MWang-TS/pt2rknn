"""
PT to RKNN 多模型转换引擎 v2
支持模型类型：yolov8_det / yolov8_seg / yolov8_pose / yolov8_obb / resnet / retinaface
YOLO系列: PT --(ultralytics.export)--> ONNX --(rknn-toolkit2)--> RKNN
ONNX系列: ONNX --(rknn-toolkit2)--> RKNN
"""
import os
import sys
import glob
import logging

from model_registry import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 校准数据集工具
# ──────────────────────────────────────────────────────────────

def _auto_dataset_txt(images_dir: str, dataset_txt: str) -> bool:
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
    imgs = []
    for pat in exts:
        imgs.extend(glob.glob(os.path.join(images_dir, pat)))
    imgs.sort()
    if not imgs:
        return False
    with open(dataset_txt, 'w') as f:
        for p in imgs:
            f.write(os.path.abspath(p) + '\n')
    logger.info(f"已生成 dataset.txt，共 {len(imgs)} 张校准图片")
    return True


def _resolve_dataset(calibration_dir: str, subdir: str):
    base = os.path.join(calibration_dir, subdir)
    txt_path = os.path.join(base, 'dataset.txt')

    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        return txt_path

    images_dir = os.path.join(base, 'images')
    if os.path.isdir(images_dir):
        if _auto_dataset_txt(images_dir, txt_path):
            return txt_path

    logger.warning(f"未找到校准数据集：{base}/images/，INT8 量化可能失败")
    return None


# ──────────────────────────────────────────────────────────────
# PT → ONNX（仅 YOLO 系列，使用 ultralytics 导出）
# ──────────────────────────────────────────────────────────────

def pt_to_onnx(pt_path: str, input_size: tuple, tmp_dir: str):
    try:
        from ultralytics import YOLO
        logger.info(f"[PT→ONNX] 加载模型：{pt_path}")
        model = YOLO(pt_path)

        logger.info(f"[PT→ONNX] 导出 ONNX，输入尺寸：{input_size}")
        result = model.export(
            format='onnx',
            imgsz=list(input_size),
            simplify=True,
            opset=12,
            dynamic=False,
        )
        onnx_path = str(result)

        if not os.path.exists(onnx_path):
            fallback = os.path.splitext(pt_path)[0] + '.onnx'
            if os.path.exists(fallback):
                onnx_path = fallback
            else:
                return False, "ONNX 导出成功但找不到输出文件", ''

        logger.info(f"[PT→ONNX] 导出完成：{onnx_path}")
        return True, "PT → ONNX 导出成功", onnx_path

    except Exception as e:
        return False, f"PT → ONNX 导出失败：{e}", ''


# ──────────────────────────────────────────────────────────────
# ONNX → RKNN
# ──────────────────────────────────────────────────────────────

def onnx_to_rknn(onnx_path, output_path, platform, do_quant,
                  dataset_path, mean_values, std_values, input_size,
                  verbose=False):
    try:
        from rknn.api import RKNN
    except ImportError:
        return False, "未安装 rknn-toolkit2，请先安装"

    rknn = RKNN(verbose=verbose)
    try:
        logger.info(f"[ONNX→RKNN] 配置：platform={platform}, quant={do_quant}, "
                    f"mean={mean_values}, std={std_values}")
        ret = rknn.config(
            mean_values=mean_values,
            std_values=std_values,
            target_platform=platform,
        )
        if ret != 0:
            return False, f"RKNN config 失败，ret={ret}"

        logger.info(f"[ONNX→RKNN] 加载 ONNX：{onnx_path}")
        ret = rknn.load_onnx(
            model=onnx_path,
            input_size_list=[[1, 3, input_size[0], input_size[1]]],
        )
        if ret != 0:
            return False, f"加载 ONNX 失败，ret={ret}"

        logger.info(f"[ONNX→RKNN] 构建 RKNN 模型 (do_quant={do_quant}) ...")
        if do_quant and dataset_path:
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            if do_quant and not dataset_path:
                logger.warning("缺少校准数据集，将回退到 FP16 模式")
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            return False, f"RKNN build 失败，ret={ret}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"[ONNX→RKNN] 导出：{output_path}")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            return False, f"导出 RKNN 失败，ret={ret}"

        logger.info("[ONNX→RKNN] 完成 ✓")
        return True, "RKNN 导出成功"

    finally:
        rknn.release()


# ──────────────────────────────────────────────────────────────
# 统一入口
# ──────────────────────────────────────────────────────────────

class UniversalConverter:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def convert(self, model_type, input_path, platform, do_quant,
                calibration_dir, output_path, input_size=(640, 640)):
        if model_type not in MODEL_REGISTRY:
            return False, f"未知模型类型：{model_type}", ''

        cfg = MODEL_REGISTRY[model_type]
        ext = os.path.splitext(input_path)[1].lower()
        onnx_path = None
        tmp_onnx = None
        steps = []

        try:
            if ext in ('.pt', '.pth'):
                if cfg['source_type'] == 'onnx_only':
                    return False, f"{cfg['short']} 只支持 .onnx 输入，不支持 .pt", ''

                logger.info("检测到 PT 文件，开始 PT → ONNX 导出...")
                ok, msg, onnx_path = pt_to_onnx(
                    pt_path=input_path,
                    input_size=input_size,
                    tmp_dir=os.path.dirname(input_path),
                )
                if not ok:
                    return False, msg, ''
                steps.append(f"PT → ONNX：{msg}")
                tmp_onnx = onnx_path

            elif ext == '.onnx':
                onnx_path = input_path
                steps.append("输入为 ONNX，跳过导出步骤")
            else:
                return False, f"不支持的文件格式：{ext}", ''

            dataset_path = None
            if do_quant:
                dataset_path = _resolve_dataset(calibration_dir, cfg['calibration_subdir'])
                if not dataset_path:
                    do_quant = False
                    steps.append(
                        f"⚠️ 未找到 {cfg['calibration_subdir']} 校准数据集，已回退为 FP16"
                    )

            logger.info("开始 ONNX → RKNN 转换...")
            ok, msg = onnx_to_rknn(
                onnx_path=onnx_path,
                output_path=output_path,
                platform=platform,
                do_quant=do_quant,
                dataset_path=dataset_path,
                mean_values=cfg['mean_values'],
                std_values=cfg['std_values'],
                input_size=input_size,
                verbose=self.verbose,
            )
            steps.append(f"ONNX → RKNN：{msg}")

            if ok:
                return True, '\n'.join(steps), output_path
            else:
                return False, msg, ''

        finally:
            if tmp_onnx and os.path.exists(tmp_onnx):
                try:
                    os.remove(tmp_onnx)
                    logger.info(f"已清理临时 ONNX：{tmp_onnx}")
                except Exception:
                    pass


# 保持向后兼容
PT2RKNNConverter = UniversalConverter
