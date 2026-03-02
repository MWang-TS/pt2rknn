"""
RKNN 推理测试模块
使用 rknn-toolkit2 模拟器模式（x86/WSL），验证转换后的 RKNN 模型是否可用。

支持的模型类型：
  yolov8_det  — 目标检测，单输出 [1, 4+nc, 8400]
  yolov8_seg  — 实例分割，双输出（检测 + 分割 proto）
  yolov8_pose — 姿态估计，单输出 [1, 56, 8400]（17 关键点）
  yolov8_obb  — 旋转框检测，单输出 [1, 4+nc+1, 8400]
  resnet      — 图像分类，单输出 [1, 1000]
  retinaface  — 人脸检测，多输出（anchor-based）

输入约定：
  - RKNN 模型已在 config 阶段嵌入 mean_values / std_values
  - inference() 传入原始 uint8 BGR 图（工具内部转为 RGB）
"""

import os
import cv2
import base64
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 公共颜色表（BGR）
# ─────────────────────────────────────────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157,  99), (255,112,  31), (255,178,  29),
    ( 72, 249,  10), (146, 204,  23), ( 61, 219,134), ( 26,147, 52),
    (  0,212,187), ( 44,153,168), (  0,194,255), ( 52, 69,149),
    (100,115,255), (  0, 24,236), (132, 56,255), ( 82,  0,133),
    (203, 56,255), (255,149,200), (255, 55,199),(255,  0, 0),
]

def _color(idx):
    c = _PALETTE[int(idx) % len(_PALETTE)]
    return (int(c[2]), int(c[1]), int(c[0]))   # BGR


# ─────────────────────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────────────────────

def letterbox(img_bgr, target_w, target_h):
    """
    保持比例缩放并填充到目标尺寸（黑色填充）。
    返回 (img_rgb_uint8_HWC, scale, pad_x, pad_y)
    """
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - nw) // 2
    pad_y = (target_h - nh) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return img_rgb, scale, pad_x, pad_y


def restore_boxes(boxes_xyxy, scale, pad_x, pad_y, orig_w, orig_h):
    """将 letterbox 空间的 xyxy 框映射回原始图像坐标，并裁剪到边界。"""
    boxes = boxes_xyxy.copy().astype(np.float32)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes


# ─────────────────────────────────────────────────────────────
# NMS（numpy，无需 torch）
# ─────────────────────────────────────────────────────────────

def nms(boxes_xyxy, scores, iou_thresh=0.45):
    """返回保留框的下标。"""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


# ─────────────────────────────────────────────────────────────
# YOLOv8 单输出后处理（Det / Seg / Pose / OBB）
# 标准 ultralytics ONNX export 格式：[1, 4+nc(+extra), 8400]
# ─────────────────────────────────────────────────────────────

def _decode_yolo_common(pred, conf_thresh, iou_thresh, num_extra=0):
    """
    通用 YOLOv8 解码。
    pred: [N_proposals, 4+nc+num_extra]
    返回 (boxes_cxcywh, class_ids, class_scores, extra)
    """
    # boxes cxcywh
    boxes_cxcywh = pred[:, :4]
    # class logits / scores (sigmoid already applied by ultralytics export)
    cls_start = 4
    cls_end = pred.shape[1] - num_extra
    class_probs = pred[:, cls_start:cls_end]
    extra = pred[:, cls_end:] if num_extra > 0 else None

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]

    mask = class_scores >= conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]
    extra_filtered = extra[mask] if extra is not None else None

    return boxes_cxcywh, class_ids, class_scores, extra_filtered


def postprocess_det(outputs, orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names):
    """YOLOv8-Det: outputs[0] shape [1, 4+nc, 8400]"""
    result = orig_bgr.copy()
    h, w = orig_bgr.shape[:2]
    summary_lines = []

    raw = outputs[0]  # [1, 4+nc, 8400]
    if raw.ndim == 3:
        raw = raw[0]              # [4+nc, 8400]
    pred = raw.T                  # [8400, 4+nc]

    boxes_cxcywh, cids, cscores, _ = _decode_yolo_common(pred, conf_thresh, iou_thresh)
    if len(boxes_cxcywh) == 0:
        summary_lines.append('未检测到目标（置信度阈值 {:.2f}）'.format(conf_thresh))
        return result, '\n'.join(summary_lines), []

    # cxcywh → xyxy
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS per class
    keep_all = []
    for cid in np.unique(cids):
        idx = np.where(cids == cid)[0]
        keep = nms(boxes_xyxy[idx], cscores[idx], iou_thresh)
        keep_all.extend(idx[keep].tolist())

    boxes_xyxy = boxes_xyxy[keep_all]
    cids = cids[keep_all]
    cscores = cscores[keep_all]

    # restore to original coords
    boxes_orig = restore_boxes(boxes_xyxy, scale, pad_x, pad_y, w, h)

    detections = []
    for i, (box, cid, score) in enumerate(zip(boxes_orig, cids, cscores)):
        x1o, y1o, x2o, y2o = [int(v) for v in box]
        name = class_names[int(cid)] if class_names and int(cid) < len(class_names) else 'cls{}'.format(int(cid))
        color = _color(int(cid))
        cv2.rectangle(result, (x1o, y1o), (x2o, y2o), color, 2)
        label = '{} {:.2f}'.format(name, float(score))
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (x1o, y1o - lh - 6), (x1o + lw, y1o), color, -1)
        cv2.putText(result, label, (x1o, y1o - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        detections.append({'class': name, 'score': round(float(score), 3), 'box': [x1o, y1o, x2o, y2o]})

    summary_lines.append('检测到 {} 个目标'.format(len(detections)))
    for d in detections:
        summary_lines.append('  {} {:.3f}  {}'.format(d['class'], d['score'], d['box']))
    return result, '\n'.join(summary_lines), detections


def postprocess_seg(outputs, orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names):
    """YOLOv8-Seg: outputs[0]=[1,4+nc+32,8400], outputs[1]=[1,32,160,160]
    简化处理：只绘制检测框，不渲染掩码（掩码解码需要额外内存）"""
    # 用 det 后处理处理主输出（忽略掩码 proto 输出）
    main_out = outputs[0]
    raw = main_out[0] if main_out.ndim == 3 else main_out
    raw = raw.T  # [8400, 4+nc+32]
    # 剔除末尾32维掩码系数
    nc = raw.shape[1] - 4 - 32
    # 只取 det 部分
    raw_det = np.concatenate([raw[:, :4], raw[:, 4:4+nc]], axis=1)
    det_output = raw_det[np.newaxis]  # [1, 4+nc, 8400]... actually just pass it
    # Re-use det postprocess with synthetic single output
    return postprocess_det([raw_det.T[np.newaxis] if False else np.expand_dims(raw_det.T, 0)],
                            orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names)


def postprocess_pose(outputs, orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """YOLOv8-Pose: output [1, 56, 8400]  (4 box +1 cls +51 kpts)"""
    SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
                (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    KPT_COLOR = [(0,255,0)] * 17

    result = orig_bgr.copy()
    h, w = orig_bgr.shape[:2]
    summary_lines = []

    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    pred = raw.T  # [8400, 56]

    boxes_cxcywh = pred[:, :4]
    scores = 1 / (1 + np.exp(-pred[:, 4]))   # sigmoid person score
    kpts = pred[:, 5:]                           # [8400, 51]

    mask = scores >= conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    scores_f = scores[mask]
    kpts_f = kpts[mask]

    if len(boxes_cxcywh) == 0:
        summary_lines.append('未检测到人体（置信度阈值 {:.2f}）'.format(conf_thresh))
        return result, '\n'.join(summary_lines), []

    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, scores_f, iou_thresh)
    boxes_xyxy = boxes_xyxy[keep]
    scores_f = scores_f[keep]
    kpts_f = kpts_f[keep]

    boxes_orig = restore_boxes(boxes_xyxy, scale, pad_x, pad_y, w, h)

    detections = []
    for i, (box, score, kpt) in enumerate(zip(boxes_orig, scores_f, kpts_f)):
        x1o, y1o, x2o, y2o = [int(v) for v in box]
        cv2.rectangle(result, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
        label = 'person {:.2f}'.format(float(score))
        cv2.putText(result, label, (x1o, y1o - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # draw keypoints
        kpt_xy = kpt.reshape(17, 3)  # (x, y, conf)
        kpt_pts = []
        for k in range(17):
            kx, ky, kc = kpt_xy[k]
            # restore from letterbox space to original
            kx_r = (kx - pad_x) / scale
            ky_r = (ky - pad_y) / scale
            kpt_pts.append((int(kx_r), int(ky_r)))
            if kc > 0.3:
                cv2.circle(result, (int(kx_r), int(ky_r)), 4, KPT_COLOR[k], -1)

        for a, b in SKELETON:
            if kpt_xy[a][2] > 0.3 and kpt_xy[b][2] > 0.3:
                cv2.line(result, kpt_pts[a], kpt_pts[b], (0, 180, 255), 2)

        detections.append({'score': round(float(score), 3), 'box': [x1o, y1o, x2o, y2o]})

    summary_lines.append('检测到 {} 个人体姿态'.format(len(detections)))
    return result, '\n'.join(summary_lines), detections


def postprocess_obb(outputs, orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh, class_names):
    """YOLOv8-OBB: output [1, 4+nc+1, 8400] (cx,cy,w,h + classes + angle)"""
    import math

    result = orig_bgr.copy()
    h, w = orig_bgr.shape[:2]
    summary_lines = []

    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    pred = raw.T  # [8400, 4+nc+1]

    nc = pred.shape[1] - 5
    boxes_cxcywh = pred[:, :4]
    class_probs = pred[:, 4:4+nc]
    angles = pred[:, 4+nc]

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(len(class_probs)), class_ids]

    mask = class_scores >= conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    class_scores = class_scores[mask]
    angles_f = angles[mask]

    if len(boxes_cxcywh) == 0:
        summary_lines.append('未检测到目标（置信度阈值 {:.2f}）'.format(conf_thresh))
        return result, '\n'.join(summary_lines), []

    # axis-aligned NMS for OBB (approximate)
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep_all = []
    for cid in np.unique(class_ids):
        idx = np.where(class_ids == cid)[0]
        keep = nms(boxes_xyxy[idx], class_scores[idx], iou_thresh)
        keep_all.extend(idx[keep].tolist())

    boxes_cxcywh = boxes_cxcywh[keep_all]
    class_ids = class_ids[keep_all]
    class_scores = class_scores[keep_all]
    angles_f = angles_f[keep_all]

    detections = []
    for i, (box, cid, score, angle) in enumerate(zip(boxes_cxcywh, class_ids, class_scores, angles_f)):
        cx, cy, bw, bh = box
        # restore center to original image space
        cx_r = (cx - pad_x) / scale
        cy_r = (cy - pad_y) / scale
        bw_r = bw / scale
        bh_r = bh / scale

        name = class_names[int(cid)] if class_names and int(cid) < len(class_names) else 'cls{}'.format(int(cid))
        color = _color(int(cid))

        # draw rotated rectangle
        rect = ((cx_r, cy_r), (bw_r, bh_r), math.degrees(float(angle)))
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(result, [pts], 0, color, 2)
        label = '{} {:.2f}'.format(name, float(score))
        cv2.putText(result, label, (int(cx_r), int(cy_r)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        detections.append({'class': name, 'score': round(float(score), 3), 'angle_deg': round(math.degrees(float(angle)), 1)})

    summary_lines.append('检测到 {} 个旋转框目标'.format(len(detections)))
    for d in detections:
        summary_lines.append('  {} {:.3f}  angle={:.1f}°'.format(d['class'], d['score'], d['angle_deg']))
    return result, '\n'.join(summary_lines), detections


def postprocess_resnet(outputs, orig_bgr, class_names, topk=5):
    """ResNet 分类：output [1, num_classes]，返回 top-k 结果。"""
    result = orig_bgr.copy()
    summary_lines = []

    logits = outputs[0]
    if logits.ndim == 2:
        logits = logits[0]
    # softmax
    logits = logits.astype(np.float32)
    e = np.exp(logits - logits.max())
    probs = e / e.sum()

    topk = min(topk, len(probs))
    top_ids = np.argsort(probs)[::-1][:topk]

    summary_lines.append('Top-{} 分类结果：'.format(topk))
    detections = []
    for rank, i in enumerate(top_ids):
        name = class_names[int(i)] if class_names and int(i) < len(class_names) else 'class_{}'.format(int(i))
        prob = float(probs[i])
        summary_lines.append('  #{} {} — {:.2%}'.format(rank + 1, name, prob))
        detections.append({'rank': rank + 1, 'class': name, 'prob': round(prob, 4)})

    # overlay result text on image
    panel_h = min(max(180, topk * 32 + 30), result.shape[0])
    overlay = result[:panel_h, :, :].copy()
    cv2.rectangle(overlay, (0, 0), (min(400, result.shape[1]), panel_h), (30, 30, 30), -1)
    result[:panel_h, :min(400, result.shape[1]), :] = cv2.addWeighted(
        overlay[:, :min(400, result.shape[1])], 0.6,
        result[:panel_h, :min(400, result.shape[1])], 0.4, 0)
    for rank, det in enumerate(detections):
        bar_w = max(2, int(300 * det['prob']))
        cv2.rectangle(result, (8, 22 + rank * 30), (8 + bar_w, 38 + rank * 30), _color(rank), -1)
        cv2.putText(result, '#{} {:.1%} {}'.format(det['rank'], det['prob'], det['class'][:25]),
                    (8, 36 + rank * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    return result, '\n'.join(summary_lines), detections


def postprocess_retinaface(outputs, orig_bgr, scale, pad_x, pad_y, conf_thresh, iou_thresh):
    """
    RetinaFace：3 组 anchor-based 输出
    outputs: [loc[1,N,4], cls[1,N,2], ldm[1,N,10]] × 3 scales
    与 rknn_model_zoo 中 RetinaFace 输出格式匹配。
    若输出格式不符，退回显示输出张量摘要。
    """
    result = orig_bgr.copy()
    h, w = orig_bgr.shape[:2]
    summary_lines = []

    # Try to use raw confidence output to draw simple boxes
    try:
        boxes_list = []
        scores_list = []

        n_out = len(outputs)
        # Expect 3 or 9 outputs depending on model variant
        # Try the simplest case: first output is [1, N, 4] boxes, second is [1, N, 2] cls
        if n_out >= 2:
            raw_loc = outputs[0]  # [1, N, 4] or similar
            raw_cls = outputs[1]  # [1, N, 2]

            if raw_loc.ndim == 3:
                raw_loc = raw_loc[0]  # [N, 4]
            if raw_cls.ndim == 3:
                raw_cls = raw_cls[0]  # [N, 2]

            if raw_cls.shape[-1] == 2:
                face_scores = 1 / (1 + np.exp(-raw_cls[:, 1]))  # sigmoid
                mask = face_scores > conf_thresh
                if mask.sum() > 0:
                    boxes = raw_loc[mask]
                    scores = face_scores[mask]
                    # assumed xyxy format in letterbox input space
                    boxes_xyxy = restore_boxes(boxes, scale, pad_x, pad_y, w, h)
                    keep = nms(boxes_xyxy, scores, iou_thresh)
                    for idx in keep:
                        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[idx]]
                        sc = float(scores[idx])
                        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(result, 'face {:.2f}'.format(sc), (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        summary_lines.append('  face {:.3f}  [{},{},{},{}]'.format(sc, x1, y1, x2, y2))
                    if summary_lines:
                        summary_lines.insert(0, '检测到 {} 张人脸'.format(len(keep)))
                        return result, '\n'.join(summary_lines), []
    except Exception as e:
        logger.warning('RetinaFace 后处理失败，显示张量摘要: %s', e)

    # Fallback: just show output tensor info
    for i, out in enumerate(outputs):
        summary_lines.append('Output[{}]: shape={} range=[{:.3f}, {:.3f}]'.format(
            i, list(out.shape), float(out.min()), float(out.max())))
    summary_lines.insert(0, '⚠ RetinaFace 后处理需要与模型输出格式匹配，以下是张量摘要：')
    return result, '\n'.join(summary_lines), []


# ─────────────────────────────────────────────────────────────
# 主推理函数
# ─────────────────────────────────────────────────────────────

def run_inference(rknn_path, img_bgr, model_type, input_w, input_h,
                  conf_thresh=0.25, iou_thresh=0.45, class_names=None,
                  onnx_path=None, mean_values=None, std_values=None,
                  platform='rk3576'):
    """
    使用 rknn-toolkit2 simulator 模式推理。
    必须提供 onnx_path（与 rknn 同名的 .onnx 文件），
    通过 load_onnx → config → build → init_runtime() 运行。
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        raise RuntimeError('未安装 rknn-toolkit2：请运行 pip install rknn-toolkit2')

    # 自动查找同名 .onnx（若未传入）
    if not onnx_path or not os.path.exists(onnx_path):
        candidate = os.path.splitext(rknn_path)[0] + '.onnx'
        if os.path.exists(candidate):
            onnx_path = candidate
        else:
            raise RuntimeError(
                '找不到对应的 ONNX 文件，无法在 x86 模拟器上推理。\n'
                '请重新转换模型（重新转换后会自动保存 ONNX）。'
            )

    orig_h, orig_w = img_bgr.shape[:2]
    img_lb, scale, pad_x, pad_y = letterbox(img_bgr, input_w, input_h)

    # 默认 mean/std（YOLO 常用值）
    mv = mean_values if mean_values else [[0, 0, 0]]
    sv = std_values  if std_values  else [[255, 255, 255]]

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            mean_values=mv,
            std_values=sv,
            target_platform=platform,
        )
        if ret != 0:
            raise RuntimeError('RKNN config 失败，返回码 {}'.format(ret))

        ret = rknn.load_onnx(
            model=onnx_path,
            input_size_list=[[1, 3, input_h, input_w]],
        )
        if ret != 0:
            raise RuntimeError('load_onnx 失败，返回码 {}'.format(ret))

        ret = rknn.build(do_quantization=False)   # simulator 不需要量化
        if ret != 0:
            raise RuntimeError('RKNN build 失败，返回码 {}'.format(ret))

        ret = rknn.init_runtime()                 # x86 simulator 模式
        if ret != 0:
            raise RuntimeError('init_runtime 失败，返回码 {}'.format(ret))

        t0 = time.time()
        outputs = rknn.inference(inputs=[img_lb])
        infer_ms = (time.time() - t0) * 1000
    finally:
        rknn.release()

    if outputs is None or len(outputs) == 0:
        raise RuntimeError('inference() 返回空结果')

    # 后处理分发
    if model_type == 'yolov8_det':
        result, summary, dets = postprocess_det(outputs, img_bgr, scale, pad_x, pad_y,
                                                 conf_thresh, iou_thresh, class_names)
    elif model_type == 'yolov8_seg':
        result, summary, dets = postprocess_seg(outputs, img_bgr, scale, pad_x, pad_y,
                                                  conf_thresh, iou_thresh, class_names)
    elif model_type == 'yolov8_pose':
        result, summary, dets = postprocess_pose(outputs, img_bgr, scale, pad_x, pad_y,
                                                   conf_thresh, iou_thresh)
    elif model_type == 'yolov8_obb':
        result, summary, dets = postprocess_obb(outputs, img_bgr, scale, pad_x, pad_y,
                                                  conf_thresh, iou_thresh, class_names)
    elif model_type == 'resnet':
        result, summary, dets = postprocess_resnet(outputs, img_bgr, class_names)
    elif model_type == 'retinaface':
        result, summary, dets = postprocess_retinaface(outputs, img_bgr, scale, pad_x, pad_y,
                                                        conf_thresh, iou_thresh)
    else:
        # 未知类型：直接展示输出张量摘要
        result = img_bgr.copy()
        lines = ['未知模型类型 {}，显示输出张量摘要：'.format(model_type)]
        for i, out in enumerate(outputs):
            lines.append('  Output[{}]: shape={} range=[{:.3f},{:.3f}]'.format(
                i, list(out.shape), float(out.min()), float(out.max())))
        summary = '\n'.join(lines)
        dets = []

    return result, summary, dets, infer_ms


def img_to_base64(img_bgr, quality=88):
    """将 BGR numpy 图像编码为 base64 JPEG 字符串。"""
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('图像编码失败')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ─────────────────────────────────────────────────────────────
# 量化精度分析
# ─────────────────────────────────────────────────────────────

def _parse_accuracy_output(output_dir):
    """
    解析 accuracy_analysis() 输出目录，提取逐层余弦相似度。
    返回 {layers: [{name, cos_sim}], summary, raw_text}

    rknn-toolkit2 error_analysis.txt 格式（每行）：
      [LayerType] layer_name    <entire_cos> | <entire_euc>    <single_cos> | <single_euc>
    """
    raw_lines = []
    layers = []

    # 按优先级查找输出文件
    candidates = ['error_analysis.txt', 'accuracy_analysis.txt',
                  'snapshot.txt', 'summary.txt']
    found_file = None
    for name in candidates:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            found_file = p
            break
    if not found_file:
        for f in os.listdir(output_dir):
            if f.endswith('.txt'):
                found_file = os.path.join(output_dir, f)
                break

    if found_file:
        with open(found_file, 'r', encoding='utf-8', errors='replace') as fh:
            raw_lines = fh.readlines()

    # 解析格式：[Type] name    cos | euc    cos | euc
    # 跳过注释行（#）和表头行（不含 [）
    import re
    for line in raw_lines:
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue
        # 必须以 [ 开头才是数据行
        m = re.match(r'\[([^\]]+)\]\s+(\S+)\s+([\d.]+)\s*\|', line)
        if m:
            layer_name = f'[{m.group(1)}] {m.group(2)}'
            cos_sim = round(float(m.group(3)), 6)
            layers.append({'name': layer_name, 'cos_sim': cos_sim})

    # 统计摘要
    cos_vals = [l['cos_sim'] for l in layers]
    if cos_vals:
        min_c = min(cos_vals)
        avg_c = sum(cos_vals) / len(cos_vals)
        bottom5 = sorted(layers, key=lambda x: x['cos_sim'])[:5]
        if min_c >= 0.99:
            grade = '优秀 ✅（算子转换精度极佳）'
        elif min_c >= 0.95:
            grade = '良好 ⚠️（轻微误差，可接受）'
        else:
            grade = '注意 ❌（误差较大，建议检查算子支持或增加校准图片）'
        worst_info = '  '.join(f'{l["name"]}: {l["cos_sim"]:.4f}' for l in bottom5)
        summary = (
            f'分析层数：{len(cos_vals)}\n'
            f'最低余弦相似度：{min_c:.6f}\n'
            f'平均余弦相似度：{avg_c:.6f}\n'
            f'评级：{grade}\n'
            f'最差 5 层：{worst_info}'
        )
    else:
        summary = ''.join(raw_lines[:60]) if raw_lines else '无分析结果'

    return {
        'layers': layers,
        'summary': summary,
        'raw_text': ''.join(raw_lines[:200]),
    }


def run_accuracy_analysis(onnx_path, img_bgr, input_w, input_h,
                          mean_values=None, std_values=None,
                          platform='rk3576', do_quantization=False,
                          dataset_path=None):
    """
    使用 rknn-toolkit2 accuracy_analysis() 逐层对比 ONNX 与 RKNN 输出误差。

    参数
    ----
    onnx_path        : str  — ONNX 模型路径
    img_bgr          : ndarray — BGR 原始测试图（会自动 letterbox）
    input_w/h        : int
    mean_values      : list  — config mean（同转换时）
    std_values       : list  — config std（同转换时）
    platform         : str   — 目标平台
    do_quantization  : bool  — True → INT8 量化后对比（需要 dataset_path）
                               False → FP 对比（仅验证算子转换正确性）
    dataset_path     : str   — 量化校准数据集 .txt 文件路径（do_quantization=True 时使用）

    返回
    ----
    dict: {layers, summary, raw_text, quant_mode}
    """
    import tempfile
    from rknn.api import RKNN

    img_lb, _, _, _ = letterbox(img_bgr, input_w, input_h)   # uint8 RGB HWC

    mv = mean_values if mean_values else [[0, 0, 0]]
    sv = std_values  if std_values  else [[255, 255, 255]]

    with tempfile.TemporaryDirectory() as tmp_dir:
        accuracy_dir = os.path.join(tmp_dir, 'accuracy')
        os.makedirs(accuracy_dir, exist_ok=True)

        # 若 do_quantization=True 但没有 dataset，用当前图片生成单图数据集
        if do_quantization and not dataset_path:
            tmp_img_path = os.path.join(tmp_dir, 'calib_img.jpg')
            cv2.imwrite(tmp_img_path, img_bgr)
            tmp_ds_path = os.path.join(tmp_dir, 'dataset.txt')
            with open(tmp_ds_path, 'w') as f:
                f.write(tmp_img_path + '\n')
            dataset_path = tmp_ds_path

        # accuracy_analysis expects NCHW float32: (1, C, H, W)
        img_nchw = img_lb.transpose(2, 0, 1)[np.newaxis, :]   # (1, 3, H, W)

        rknn = RKNN(verbose=False)
        try:
            ret = rknn.config(
                mean_values=mv,
                std_values=sv,
                target_platform=platform,
            )
            if ret != 0:
                raise RuntimeError(f'RKNN config 失败，返回码 {ret}')

            ret = rknn.load_onnx(
                model=onnx_path,
                input_size_list=[[1, 3, input_h, input_w]],
            )
            if ret != 0:
                raise RuntimeError(f'load_onnx 失败，返回码 {ret}')

            if do_quantization:
                ret = rknn.build(do_quantization=True, dataset=dataset_path)
            else:
                ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f'RKNN build 失败，返回码 {ret}')

            ret = rknn.init_runtime()    # x86 simulator
            if ret != 0:
                raise RuntimeError(f'init_runtime 失败，返回码 {ret}')

            ret = rknn.accuracy_analysis(
                inputs=[img_nchw],
                output_dir=accuracy_dir,
            )
            if ret != 0:
                raise RuntimeError(f'accuracy_analysis 失败，返回码 {ret}')
        finally:
            rknn.release()

        result = _parse_accuracy_output(accuracy_dir)
        result['quant_mode'] = 'INT8 量化对比' if do_quantization else 'FP 浮点对比（验证算子转换）'
        return result
