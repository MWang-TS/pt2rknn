"""
PT to RKNN 多模型转换引擎 v2
支持模型类型：yolov8_det / yolov8_seg / yolov8_pose / yolov8_obb / resnet / retinaface
YOLO系列: PT --(ultralytics.export)--> ONNX --(rknn-toolkit2)--> RKNN
ONNX系列: ONNX --(rknn-toolkit2)--> RKNN
"""
import os
import sys
import glob
import json
import logging
import subprocess
import tempfile
import hashlib
import shutil

from model_registry import MODEL_REGISTRY

# yolov8-gpu 环境 Python 路径（WSL 下用 /mnt/ 路径直接执行 Windows exe）
_WIN_YOLO_PYTHON = '/mnt/e/Anaconda3/envs/yolov8-gpu/python.exe'

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

QUANT_CONFIG = {
    'quantized_algorithm': 'normal',
    'quantized_method': 'channel',
    'optimization_level': 3,
}


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_manifest(dataset_path: str):
    if not dataset_path:
        return None
    with open(dataset_path, 'r', encoding='utf-8') as file_obj:
        images = [line.strip() for line in file_obj if line.strip()]
    digest = hashlib.sha256('\n'.join(images).encode('utf-8')).hexdigest()
    return {
        'path': os.path.abspath(dataset_path),
        'image_count': len(images),
        'manifest_sha256': digest,
    }


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


def _wsl_to_win(path: str) -> str:
    """将 WSL /mnt/x/... 路径转换为 Windows X:\\... 路径"""
    if path.startswith('/mnt/') and len(path) > 5:
        parts = path[5:].split('/', 1)
        drive = parts[0].upper()
        rest = parts[1].replace('/', '\\') if len(parts) > 1 else ''
        return f'{drive}:\\{rest}'
    return path


def _win_to_wsl(path: str) -> str:
    """将 Windows X:\\... 路径转换为 WSL /mnt/x/... 路径"""
    if len(path) >= 3 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    return path


def _pt_export_subprocess(pt_path: str, input_size: tuple, fmt: str) -> tuple:
    """
    通过 subprocess 调用 yolov8-gpu Python 导出 ONNX 或 rknn torchscript。
    用于处理当前环境 ultralytics 版本过旧、无法加载新模型的情况。
    fmt: 'onnx' 或 'rknn'
    """
    # WSL 路径转 Windows 路径传给 Windows Python
    win_pt = _wsl_to_win(pt_path)

    script_content = (
        "import sys, os\n"
        "sys.path = [p for p in sys.path if 'ultralytics_yolov8' not in p]\n"
        "from ultralytics import YOLO\n"
        f"model = YOLO({json.dumps(win_pt)})\n"
        f"result = model.export(format={json.dumps(fmt)}, imgsz={list(input_size)}, "
        "simplify=True, opset=12, dynamic=False)\n"
        'print("__OUTPUT__:" + str(result))\n'
    )

    tmp_script_wsl = None
    tmp_script_win = None
    try:
        # 写到 Windows 可访问的目录（/mnt/e 下），Windows Python 才能读取
        win_tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        os.makedirs(win_tmp_dir, exist_ok=True)
        import tempfile
        fd, tmp_script_wsl = tempfile.mkstemp(suffix='.py', prefix='rknn_export_', dir=win_tmp_dir)
        with os.fdopen(fd, 'w') as f:
            f.write(script_content)
        tmp_script_win = _wsl_to_win(tmp_script_wsl)

        # WSL 可以直接执行 Windows .exe（通过 /mnt/ 路径）
        proc = subprocess.run(
            [_WIN_YOLO_PYTHON, tmp_script_win],
            capture_output=True, text=True, timeout=300
        )
        output = proc.stdout + proc.stderr
        logger.debug(f'[subprocess] stdout: {proc.stdout[-400:]}')
        if proc.returncode != 0:
            logger.debug(f'[subprocess] stderr: {proc.stderr[-400:]}')

        for line in (proc.stdout + '\n' + proc.stderr).splitlines():
            line = line.strip()
            if line.startswith('__OUTPUT__:'):
                out_path = line[len('__OUTPUT__:'):].strip()
                # Windows 路径转回 WSL 路径
                out_path = _win_to_wsl(out_path)
                return True, out_path

        err = (proc.stderr.strip() or proc.stdout.strip())[-800:]
        return False, f'subprocess 导出失败（exit {proc.returncode}）：{err}'
    except subprocess.TimeoutExpired:
        return False, 'subprocess 导出超时'
    except Exception as e:
        return False, f'subprocess 调用异常：{e}'
    finally:
        if tmp_script_wsl and os.path.exists(tmp_script_wsl):
            os.unlink(tmp_script_wsl)


def pt_to_rknnopt(pt_path: str, input_size: tuple, tmp_dir: str):
    """
    使用 Rockchip 修改版 ultralytics 导出 rknnopt torchscript。
    输出为多头分离格式（3 scale × bbox_dfl + class_scores），
    可被 load_pytorch 正确量化，避免 bbox/class 共用 INT8 scale 的问题。
    """
    import sys
    # 优先使用 lib/ultralytics_yolov8（Rockchip 版）
    rk_lib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib', 'ultralytics_yolov8'))
    inserted = False
    try:
        if os.path.isdir(rk_lib):
            sys.path.insert(0, rk_lib)
            inserted = True
        from ultralytics import YOLO
        model = YOLO(pt_path)
        result = model.export(
            format='rknn',
            imgsz=list(input_size),
            simplify=True,
            opset=12,
            dynamic=False,
        )
        ts_path = str(result)
        logger.info(f'[PT→rknnopt] ultralytics 返回路径：{ts_path}')
        
        # 处理返回路径可能是目录或文件的情况
        if os.path.isdir(ts_path):
            # 如果是目录，在目录中查找 .torchscript 文件
            logger.warning(f'[PT→rknnopt] 返回的是目录，尝试在目录中查找 .torchscript 文件')
            candidates = glob.glob(os.path.join(ts_path, '*.torchscript'))
            if candidates:
                ts_path = candidates[0]
                logger.info(f'[PT→rknnopt] 在目录中找到：{ts_path}')
            else:
                return False, f'导出目录中未找到 .torchscript 文件：{ts_path}', ''
        elif os.path.isfile(ts_path):
            # 如果是文件但没有扩展名，添加 .torchscript
            if not ts_path.endswith('.torchscript') and not ts_path.endswith('.pt'):
                new_path = ts_path + '.torchscript'
                os.rename(ts_path, new_path)
                ts_path = new_path
                logger.info(f'[PT→rknnopt] 文件已重命名为：{ts_path}')
        else:
            # 文件不存在，尝试在 PT 文件目录查找
            logger.warning(f'[PT→rknnopt] 路径不存在：{ts_path}，尝试查找备用路径')
            base = os.path.splitext(pt_path)[0]
            for suffix in ('_rknnopt.torchscript', '_rknn_model.torchscript', '.torchscript'):
                candidate = base + suffix
                if os.path.isfile(candidate):
                    ts_path = candidate
                    logger.info(f'[PT→rknnopt] 找到备用文件：{ts_path}')
                    break
            else:
                return False, 'rknnopt 导出后找不到输出文件', ''
        
        if not os.path.isfile(ts_path):
            return False, f'最终路径不是有效文件：{ts_path}', ''
            
        logger.info(f'[PT→rknnopt] 导出完成：{ts_path}')
        return True, 'PT → rknnopt torchscript 导出成功', ts_path
    except Exception as e:
        err_msg = str(e)
        # 版本不兼容（如 DFLoss 缺失）时尝试 subprocess 回退
        if any(kw in err_msg for kw in ("Can't get attribute", 'DFLoss', 'attribute')):
            logger.warning(f'[PT→rknnopt] 检测到版本不兼容（{err_msg[:120]}），尝试 subprocess（yolov8-gpu）...')
            ok, result = _pt_export_subprocess(pt_path, input_size, 'rknn')
            if ok:
                ts_path = result
                # 同样处理路径
                if os.path.isdir(ts_path):
                    candidates = glob.glob(os.path.join(ts_path, '*.torchscript'))
                    ts_path = candidates[0] if candidates else ''
                if ts_path and os.path.isfile(ts_path):
                    logger.info(f'[PT→rknnopt] subprocess 导出完成：{ts_path}')
                    return True, 'PT → rknnopt torchscript 导出成功（via subprocess yolov8-gpu）', ts_path
            logger.warning(f'[PT→rknnopt] subprocess rknn 导出也失败：{result}，rknnopt 路径放弃')
        return False, f'PT → rknnopt 导出失败：{e}', ''
    finally:
        if inserted and rk_lib in sys.path:
            sys.path.remove(rk_lib)

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
        err_msg = str(e)
        # 版本不兼容时（如 DFLoss 缺失）回退到 subprocess yolov8-gpu
        if any(kw in err_msg for kw in ("Can't get attribute", 'DFLoss', 'attribute')):
            logger.warning(f"[PT→ONNX] 检测到版本不兼容（{err_msg[:120]}），尝试 subprocess（yolov8-gpu）...")
            ok, result = _pt_export_subprocess(pt_path, input_size, 'onnx')
            if ok:
                onnx_path = result
                if not os.path.exists(onnx_path):
                    fallback = os.path.splitext(pt_path)[0] + '.onnx'
                    onnx_path = fallback if os.path.exists(fallback) else onnx_path
                if os.path.exists(onnx_path):
                    logger.info(f"[PT→ONNX] subprocess 导出完成：{onnx_path}")
                    return True, "PT → ONNX 导出成功（via subprocess yolov8-gpu）", onnx_path
            return False, f"PT → ONNX 导出失败（subprocess 也失败）：{result}", ''
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
            **QUANT_CONFIG,
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
        if do_quant:
            if not dataset_path:
                return False, "INT8 转换缺少校准数据集"
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
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

def torchscript_to_rknn(ts_path, output_path, platform, do_quant,
                        dataset_path, mean_values, std_values, input_size,
                        verbose=False):
    """使用 load_pytorch 将 rknnopt torchscript 转换为 RKNN（分头量化，INT8 更准确）"""
    from rknn.api import RKNN
    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(mean_values=mean_values, std_values=std_values,
                  target_platform=platform, **QUANT_CONFIG)
        if ret != 0:
            return False, f'RKNN config 失败，ret={ret}'

        logger.info(f'[TS→RKNN] 加载 torchscript：{ts_path}')
        ret = rknn.load_pytorch(model=ts_path,
                                input_size_list=[[1, 3, input_size[0], input_size[1]]])
        if ret != 0:
            return False, f'load_pytorch 失败，ret={ret}'

        logger.info(f'[TS→RKNN] 构建 RKNN 模型 (do_quant={do_quant}) ...')
        if do_quant:
            if not dataset_path:
                return False, 'INT8 转换缺少校准数据集'
            ret = rknn.build(do_quantization=True, dataset=dataset_path)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            return False, f'RKNN build 失败，ret={ret}'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            return False, f'export_rknn 失败，ret={ret}'
        return True, 'rknnopt torchscript → RKNN 成功（分头量化）'
    except Exception as e:
        return False, f'转换异常：{e}'
    finally:
        rknn.release()


class UniversalConverter:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.last_result = {}

    def convert(self, model_type, input_path, platform, do_quant,
                calibration_dir, output_path, input_size=(640, 640)):
        self.last_result = {}
        if model_type not in MODEL_REGISTRY:
            return False, f"未知模型类型：{model_type}", ''

        cfg = MODEL_REGISTRY[model_type]
        ext = os.path.splitext(input_path)[1].lower()
        onnx_path = None
        tmp_onnx = None
        steps = []
        dataset_path = None

        if do_quant:
            dataset_path = _resolve_dataset(calibration_dir, cfg['calibration_subdir'])
            if not dataset_path:
                return False, (
                    f"INT8 转换失败：未找到 {cfg['calibration_subdir']} 校准数据集。"
                    "请先准备校准集，或明确选择 FP 模式。"
                ), ''

        base_result = {
            'requested_quant_type': 'i8' if do_quant else 'fp',
            'actual_quant_type': 'i8' if do_quant else 'fp',
            'dataset': _dataset_manifest(dataset_path),
            'quantization_config': dict(QUANT_CONFIG) if do_quant else None,
            'source_model_sha256': _sha256_file(input_path),
        }

        try:
            if ext in ('.pt', '.pth'):
                if cfg['source_type'] == 'onnx_only':
                    return False, f"{cfg['short']} 只支持 .onnx 输入，不支持 .pt", ''

                logger.info("检测到 PT 文件，优先尝试 rknnopt 导出...")
                ts_ok, ts_msg, ts_path_val = pt_to_rknnopt(
                    pt_path=input_path,
                    input_size=input_size,
                    tmp_dir=os.path.dirname(input_path),
                )
                if ts_ok:
                    logger.info('[convert] rknnopt 成功，使用 load_pytorch 量化路径')
                    steps.append(f"PT → rknnopt torchscript：{ts_msg}")
                    ok2, msg2 = torchscript_to_rknn(
                        ts_path=ts_path_val,
                        output_path=output_path,
                        platform=platform,
                        do_quant=do_quant,
                        dataset_path=dataset_path,
                        mean_values=cfg['mean_values'],
                        std_values=cfg['std_values'],
                        input_size=input_size,
                        verbose=self.verbose,
                    )
                    steps.append(f"rknnopt torchscript → RKNN：{msg2}")
                    if not ok2:
                        return False, '\n'.join(steps), ''
                    graph_path = os.path.splitext(output_path)[0] + '.rknnopt.torchscript'
                    shutil.copy2(ts_path_val, graph_path)
                    self.last_result = {
                        **base_result,
                        'graph_type': 'rknnopt_torchscript',
                        'conversion_graph_path': os.path.abspath(graph_path),
                        'output_count_contract': 'split_head',
                        'class_output_activation': 'probability',
                    }
                    return True, '\n'.join(steps), ''
                else:
                    logger.warning(f'[convert] rknnopt 失败（{ts_msg}），回退到标准 ONNX')
                    steps.append(f"⚠️ rknnopt 回退：{ts_msg}")
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
                # 将 ONNX 复制到 output 目录旁边，供 simulator 推理使用
                onnx_out = os.path.splitext(output_path)[0] + '.onnx'
                try:
                    shutil.copy2(onnx_path, onnx_out)
                    logger.info(f"已保存 ONNX 到：{onnx_out}")
                except Exception as e:
                    onnx_out = ''
                    logger.warning(f"保存 ONNX 失败：{e}")
                self.last_result = {
                    **base_result,
                    'graph_type': 'onnx',
                    'conversion_graph_path': os.path.abspath(onnx_out) if onnx_out else '',
                    'output_count_contract': 'onnx_export',
                    'class_output_activation': 'probability',
                }
                return True, '\n'.join(steps), onnx_out
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
