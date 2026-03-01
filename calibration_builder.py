"""
校准数据集构建器
自动识别数据集目录结构，提取校准图片，生成 dataset.txt
支持常见格式：
  - plain     : 目录下直接放图片（jpg/png/…）
  - yolo      : 含 images/ 子目录（YOLO 格式）
  - imagenet  : 含若干类别子目录，每目录下放图片（ImageNet 格式）
  - coco      : 含 images/ 或 val/ 等子目录
"""
import os
import glob
import shutil
import random
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPEG', '.JPG', '.PNG')
DEFAULT_MAX_IMAGES = 50   # 每次最多取多少张图用于校准


# ─────────────────────────────────────────────────────────────
# 数据集结构探测
# ─────────────────────────────────────────────────────────────

def _collect_images_recursive(root, max_n=None):
    """递归收集 root 下所有图片路径（去重，随机打乱）"""
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if any(f.endswith(ext) for ext in IMAGE_EXTS):
                found.append(os.path.join(dirpath, f))
    random.shuffle(found)
    if max_n:
        found = found[:max_n]
    return found


def detect_dataset_format(dataset_path):
    """
    自动检测数据集格式，返回 (format_name, description)
    """
    if not os.path.isdir(dataset_path):
        return 'invalid', '路径不存在或不是目录'

    entries = os.listdir(dataset_path)

    # 检测 YOLO 格式（含 images/ 子目录）
    yolo_images = os.path.join(dataset_path, 'images')
    if os.path.isdir(yolo_images):
        n = len(_collect_images_recursive(yolo_images, max_n=9999))
        return 'yolo', f'YOLO 格式 — images/ 子目录，共 {n} 张图片'

    # 检测 COCO 格式（含 val2017/ 或 train2017/ 等目录）
    coco_dirs = [e for e in entries if 'val' in e.lower() or 'train' in e.lower()]
    for cd in coco_dirs:
        cdp = os.path.join(dataset_path, cd)
        if os.path.isdir(cdp):
            n = len(_collect_images_recursive(cdp, max_n=9999))
            if n > 0:
                return 'coco', f'COCO 风格 — {cd}/ 目录，共 {n} 张图片'

    # 检测 ImageNet 格式（含若干子目录，每个子目录下直接有图片）
    subdirs = [e for e in entries if os.path.isdir(os.path.join(dataset_path, e))]
    if len(subdirs) >= 2:
        sample_ok = 0
        for sd in subdirs[:5]:
            imgs = _collect_images_recursive(os.path.join(dataset_path, sd), max_n=2)
            if imgs:
                sample_ok += 1
        if sample_ok >= 2:
            total = len(_collect_images_recursive(dataset_path, max_n=9999))
            return 'imagenet', f'ImageNet 格式 — {len(subdirs)} 个类别，共约 {total} 张图片'

    # 检测 plain 格式（目录下直接放图片）
    direct_imgs = [
        f for f in entries
        if os.path.isfile(os.path.join(dataset_path, f))
        and any(f.endswith(ext) for ext in IMAGE_EXTS)
    ]
    if direct_imgs:
        return 'plain', f'普通图片目录 — 共 {len(direct_imgs)} 张图片'

    # 最后兜底：递归扫整个目录
    total = len(_collect_images_recursive(dataset_path, max_n=9999))
    if total > 0:
        return 'recursive', f'递归扫描 — 共找到 {total} 张图片'

    return 'empty', '目录中未找到图片文件'


# ─────────────────────────────────────────────────────────────
# 校准数据集构建
# ─────────────────────────────────────────────────────────────

def build_calibration_dataset(
    dataset_path,
    output_dir,
    model_type,
    max_images=DEFAULT_MAX_IMAGES,
):
    """
    从 dataset_path 提取 max_images 张图片，
    复制到 output_dir/images/，生成 output_dir/dataset.txt。

    返回 (success, message, count)
    """
    if not os.path.isdir(dataset_path):
        return False, f'路径不存在：{dataset_path}', 0

    fmt, fmt_desc = detect_dataset_format(dataset_path)
    if fmt == 'invalid' or fmt == 'empty':
        return False, f'无法从该路径提取图片：{fmt_desc}', 0

    logger.info(f"[校准] 检测到数据集格式：{fmt_desc}")

    # 根据格式确定扫描根目录
    if fmt == 'yolo':
        scan_root = os.path.join(dataset_path, 'images')
    elif fmt == 'coco':
        entries = os.listdir(dataset_path)
        scan_root = dataset_path
        for cd in entries:
            cdp = os.path.join(dataset_path, cd)
            if os.path.isdir(cdp) and ('val' in cd.lower() or 'train' in cd.lower()):
                scan_root = cdp
                break
    else:
        scan_root = dataset_path

    # 收集图片
    images = _collect_images_recursive(scan_root, max_n=max_images)
    if not images:
        return False, '未找到图片文件', 0

    # 确保输出目录存在
    images_out = os.path.join(output_dir, 'images')
    os.makedirs(images_out, exist_ok=True)

    # 复制图片（若已是相同路径则跳过复制）
    copied_paths = []
    for src in images:
        dst = os.path.join(images_out, os.path.basename(src))
        # 文件名冲突时加序号
        if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(images_out, f"{base}_{len(copied_paths)}{ext}")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        copied_paths.append(os.path.abspath(dst))

    # 生成 dataset.txt
    txt_path = os.path.join(output_dir, 'dataset.txt')
    with open(txt_path, 'w') as f:
        for p in copied_paths:
            f.write(p + '\n')

    msg = (
        f'✅ 已从 {fmt_desc} 提取 {len(copied_paths)} 张校准图片，'
        f'生成 dataset.txt'
    )
    logger.info(f"[校准] {msg}")
    return True, msg, len(copied_paths)


def get_calibration_status(calibration_dir, subdir):
    """
    检查某模型类型的校准数据集状态。
    返回 dict: {ready, count, txt_path, images_dir}
    """
    base = os.path.join(calibration_dir, subdir)
    txt = os.path.join(base, 'dataset.txt')
    images_dir = os.path.join(base, 'images')

    count = 0
    if os.path.exists(txt):
        with open(txt) as f:
            lines = [l.strip() for l in f if l.strip() and os.path.exists(l.strip())]
        count = len(lines)

    # 也统计 images/ 目录中的图片数
    img_count = 0
    if os.path.isdir(images_dir):
        img_count = sum(
            1 for f in os.listdir(images_dir)
            if any(f.endswith(ext) for ext in IMAGE_EXTS)
        )

    ready = count > 0 or img_count > 0
    return {
        'ready': ready,
        'count': max(count, img_count),
        'txt_path': txt if os.path.exists(txt) else None,
        'images_dir': images_dir,
    }
