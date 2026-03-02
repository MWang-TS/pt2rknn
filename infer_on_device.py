#!/usr/bin/env python3
"""
RK3576 设备端推理脚本（使用 rknn-toolkit-lite2 / RKNNLite）

用法
----
python infer_on_device.py \
    --model  ./model.rknn \
    --image  ./test.jpg \
    --type   yolov8_det \
    --conf   0.25 \
    --iou    0.45 \
    --classes "fire,smoke" \
    --output ./result.jpg

模型类型（--type）：
  yolov8_det   YOLOv8 目标检测
  yolov8_seg   YOLOv8 实例分割
  yolov8_pose  YOLOv8 姿态估计
  yolov8_obb   YOLOv8 旋转框检测
  resnet       图像分类
  retinaface   人脸检测

依赖
----
  pip install rknn-toolkit-lite2   # 设备端，ARM Linux
  pip install opencv-python numpy
"""

import os, sys, argparse, time
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
# 调色板
# ─────────────────────────────────────────────────────────────
_PALETTE = [
    (255, 56, 56), (255, 157, 99), (255, 112, 31), (255, 178, 29),
    (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52),
    (0, 212, 187), (44, 153, 168), (0, 194, 255), (52, 69, 149),
    (100, 115, 255), (0, 24, 236), (132, 56, 255), (82, 0, 133),
    (203, 56, 255), (255, 149, 200), (255, 55, 199), (255, 0, 0),
]
_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
]


def _color(idx):
    c = _PALETTE[int(idx) % len(_PALETTE)]
    return (int(c[2]), int(c[1]), int(c[0]))   # BGR


# ─────────────────────────────────────────────────────────────
# 图像预处理
# ─────────────────────────────────────────────────────────────

def letterbox(img_bgr, target_w, target_h):
    h, w = img_bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img_bgr, (nw, nh))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_x = (target_w - nw) // 2
    pad_y = (target_h - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return img_rgb, scale, pad_x, pad_y


def restore_boxes(boxes_xyxy, scale, pad_x, pad_y, orig_w, orig_h):
    boxes = boxes_xyxy.copy().astype(float)
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
    return boxes


# ─────────────────────────────────────────────────────────────
# NMS（纯 NumPy）
# ─────────────────────────────────────────────────────────────

def nms(boxes_xyxy, scores, iou_thresh=0.45):
    if len(boxes_xyxy) == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


# ─────────────────────────────────────────────────────────────
# 后处理
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# rknnopt 格式（6 输出）后处理工具函数
# ─────────────────────────────────────────────────────────────

def _dfl(position):
    """DFL 解码：softmax + 加权求和 → 4 个边界距离（纯 NumPy，无 torch 依赖）"""
    x = position.astype(np.float32)
    n, c, h, w = x.shape
    mc = c // 4
    x = x.reshape(n, 4, mc, h, w)
    # numerically stable softmax along axis=2
    x = x - x.max(axis=2, keepdims=True)
    e = np.exp(x)
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def _box_process(position, input_wh=(640, 640)):
    """将 DFL 输出转换为 (x1,y1,x2,y2)，坐标对应 letterbox 后的图像尺寸"""
    grid_h, grid_w = position.shape[2:4]
    col = np.tile(np.arange(grid_w)[None, None, None, :], (1, 1, grid_h, 1))
    row = np.tile(np.arange(grid_h)[None, None, :, None], (1, 1, 1, grid_w))
    grid = np.concatenate((col, row), axis=1)                     # (1,2,H,W)
    stride = np.array([input_wh[0] // grid_w, input_wh[1] // grid_h],
                      dtype=np.float32).reshape(1, 2, 1, 1)
    pos = _dfl(position)
    box_xy  = (grid + 0.5 - pos[:, 0:2]) * stride
    box_xy2 = (grid + 0.5 + pos[:, 2:4]) * stride
    return np.concatenate((box_xy, box_xy2), axis=1)              # (1,4,H,W)


def _sp_flatten(x):
    """(1,C,H,W) → (H*W, C)"""
    ch = x.shape[1]
    return x.transpose(0, 2, 3, 1).reshape(-1, ch)


def _class_names(names, nc):
    if names:
        return names
    return [f'cls{i}' for i in range(nc)]


def postprocess_det(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names,
                    input_wh=(640, 640)):
    oh, ow = img_bgr.shape[:2]
    class_names = None

    # ── 格式检测 ──────────────────────────────────────────────
    if len(outputs) >= 6:
        # rknnopt 格式：6 个输出（3 scale × bbox_dfl + class_scores）
        default_branch = 3
        pair = len(outputs) // default_branch      # 通常 = 2，有时 = 3（含 cls_sum）
        boxes_list, cls_list = [], []
        for i in range(default_branch):
            xyxy = _box_process(outputs[pair * i],     input_wh)   # (1,4,H,W)
            cls  = outputs[pair * i + 1]                            # (1,nc,H,W)
            boxes_list.append(_sp_flatten(xyxy))   # (H*W, 4)
            cls_list.append(_sp_flatten(cls))      # (H*W, nc)
        boxes_xyxy  = np.concatenate(boxes_list, axis=0)  # (N, 4)
        class_scores = np.concatenate(cls_list,  axis=0)  # (N, nc)
        # rknnopt INT8：sigmoid 已移出
        if class_scores.max() > 1.0 or class_scores.min() < 0.0:
            class_scores = 1.0 / (1.0 + np.exp(-class_scores.astype(np.float32)))
    else:
        # 标准 ONNX 格式：单输出 (1, 4+nc, 8400)
        pred = outputs[0]
        if pred.ndim == 3:
            pred = pred[0]
        pred = pred.T          # (8400, 4+nc)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - bw / 2; y1 = cy - bh / 2
        x2 = cx + bw / 2; y2 = cy + bh / 2
        boxes_xyxy   = np.stack([x1, y1, x2, y2], axis=1)
        class_scores = pred[:, 4:]
        if class_scores.max() > 1.0 or class_scores.min() < 0.0:
            class_scores = 1.0 / (1.0 + np.exp(-class_scores.astype(np.float32)))

    nc = class_scores.shape[1]
    class_names = _class_names(names, nc)
    cls_ids    = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(cls_ids)), cls_ids]

    mask = max_scores >= conf
    boxes_xyxy = boxes_xyxy[mask]; max_scores = max_scores[mask]; cls_ids = cls_ids[mask]

    keep = nms(boxes_xyxy, max_scores, iou)
    boxes_xyxy = boxes_xyxy[keep]; max_scores = max_scores[keep]; cls_ids = cls_ids[keep]
    boxes_orig = restore_boxes(boxes_xyxy, scale, pad_x, pad_y, ow, oh)

    result = img_bgr.copy()
    dets = []
    for i, (box, score, cid) in enumerate(zip(boxes_orig, max_scores, cls_ids)):
        x1r, y1r, x2r, y2r = map(int, box)
        label = class_names[cid] if cid < len(class_names) else f'cls{cid}'
        color = _color(cid)
        cv2.rectangle(result, (x1r, y1r), (x2r, y2r), color, 2)
        txt = f'{label} {score:.2f}'
        cv2.rectangle(result, (x1r, y1r - 18), (x1r + len(txt) * 9, y1r), color, -1)
        cv2.putText(result, txt, (x1r + 2, y1r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        dets.append({'label': label, 'score': float(score),
                     'box': [x1r, y1r, x2r, y2r]})

    summary = f'检测到 {len(dets)} 个目标'
    for d in dets:
        summary += f'\n  {d["label"]}  {d["score"]:.3f}  {d["box"]}'
    return result, summary, dets


def postprocess_seg(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names):
    # 仅处理检测头；分割 mask 需要原型，暂简化
    return postprocess_det([outputs[0]], img_bgr, scale, pad_x, pad_y, conf, iou, names)


def postprocess_pose(outputs, img_bgr, scale, pad_x, pad_y, conf, iou):
    oh, ow = img_bgr.shape[:2]
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T   # (8400, 56)
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    obj_scores = pred[:, 4]
    # RKNN INT8 设备端：sigmoid 被移出模型，需手动还原
    if obj_scores.max() > 1.0 or obj_scores.min() < 0.0:
        obj_scores = 1.0 / (1.0 + np.exp(-obj_scores.astype(np.float32)))

    x1 = cx - bw / 2; y1 = cy - bh / 2
    x2 = cx + bw / 2; y2 = cy + bh / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    mask = obj_scores >= conf
    boxes_xyxy = boxes_xyxy[mask]; obj_scores = obj_scores[mask]; keypoints = pred[mask, 5:]

    keep = nms(boxes_xyxy, obj_scores, iou)
    boxes_xyxy = boxes_xyxy[keep]; obj_scores = obj_scores[keep]; keypoints = keypoints[keep]
    boxes_orig = restore_boxes(boxes_xyxy, scale, pad_x, pad_y, ow, oh)

    result = img_bgr.copy()
    dets = []
    for i, (box, score, kps) in enumerate(zip(boxes_orig, obj_scores, keypoints)):
        x1r, y1r, x2r, y2r = map(int, box)
        cv2.rectangle(result, (x1r, y1r), (x2r, y2r), (0, 255, 0), 2)
        kps_xy = kps.reshape(17, 3)[:, :2]
        kps_xy[:, 0] = (kps_xy[:, 0] - pad_x) / scale
        kps_xy[:, 1] = (kps_xy[:, 1] - pad_y) / scale
        kps_i = kps_xy.astype(int)
        for pt in kps_i:
            cv2.circle(result, tuple(pt), 3, (0, 0, 255), -1)
        for a, b in _SKELETON:
            pa, pb = kps_i[a], kps_i[b]
            if 0 <= pa[0] < ow and 0 <= pa[1] < oh and 0 <= pb[0] < ow and 0 <= pb[1] < oh:
                cv2.line(result, tuple(pa), tuple(pb), (0, 255, 255), 1)
        dets.append({'score': float(score), 'box': [x1r, y1r, x2r, y2r]})

    summary = f'检测到 {len(dets)} 人'
    return result, summary, dets


def postprocess_obb(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names):
    oh, ow = img_bgr.shape[:2]
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    pred = pred.T   # (8400, 4+nc+1)
    nc = pred.shape[1] - 5
    class_names = _class_names(names, nc)

    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:4 + nc]
    # RKNN INT8 设备端：sigmoid 被移出模型，需手动还原
    if class_scores.max() > 1.0 or class_scores.min() < 0.0:
        class_scores = 1.0 / (1.0 + np.exp(-class_scores.astype(np.float32)))
    angles = pred[:, 4 + nc]
    cls_ids = np.argmax(class_scores, axis=1)
    max_scores = class_scores[np.arange(len(cls_ids)), cls_ids]

    mask = max_scores >= conf
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    max_scores, cls_ids, angles = max_scores[mask], cls_ids[mask], angles[mask]

    x1 = cx - bw / 2;  y1 = cy - bh / 2
    x2 = cx + bw / 2;  y2 = cy + bh / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes_xyxy, max_scores, iou)

    result = img_bgr.copy()
    dets = []
    for k in keep:
        cxk = (cx[k] - pad_x) / scale
        cyk = (cy[k] - pad_y) / scale
        bwk = bw[k] / scale;  bhk = bh[k] / scale
        angle = float(angles[k])
        rect = ((float(cxk), float(cyk)), (float(bwk), float(bhk)),
                float(np.degrees(angle)))
        pts = cv2.boxPoints(rect).astype(int)
        cid = cls_ids[k]; score = max_scores[k]
        color = _color(cid)
        cv2.drawContours(result, [pts], 0, color, 2)
        label = class_names[cid] if cid < len(class_names) else f'cls{cid}'
        cv2.putText(result, f'{label} {score:.2f}', (pts[0][0], pts[0][1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        dets.append({'label': label, 'score': float(score)})

    summary = f'检测到 {len(dets)} 个目标（旋转框）'
    return result, summary, dets


def postprocess_resnet(outputs, img_bgr, names, topk=5):
    logits = outputs[0].flatten()
    nc = len(logits)
    class_names = _class_names(names, nc)
    indices = np.argsort(logits)[::-1][:topk]

    result = img_bgr.copy()
    dets = []
    for rank, idx in enumerate(indices):
        label = class_names[idx] if idx < len(class_names) else f'cls{idx}'
        score = float(logits[idx])
        text = f'#{rank + 1} {label} {score:.3f}'
        cv2.putText(result, text, (10, 28 + rank * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        dets.append({'rank': rank + 1, 'label': label, 'score': score})

    summary = 'Top-{} 分类结果：\n'.format(topk)
    for d in dets:
        summary += f'  #{d["rank"]}  {d["label"]}  {d["score"]:.4f}\n'
    return result, summary.rstrip(), dets


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def run(args):
    # 导入 RKNNLite（只在设备端有效）
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print('[ERROR] rknnlite 未安装。请在 RK3576 设备上安装 rknn-toolkit-lite2。')
        print('        pip install rknn_toolkit_lite2-*.whl')
        sys.exit(1)

    model_path = args.model
    img_path   = args.image
    model_type = args.type
    conf       = args.conf
    iou        = args.iou
    names      = [n.strip() for n in args.classes.split(',') if n.strip()] if args.classes else []
    input_w    = args.width
    input_h    = args.height
    out_path   = args.output
    debug      = args.debug

    # 读取图片
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f'[ERROR] 无法读取图片：{img_path}')
        sys.exit(1)

    # 预处理
    img_rgb, scale, pad_x, pad_y = letterbox(img_bgr, input_w, input_h)
    img_input = np.expand_dims(img_rgb, axis=0)   # (1, H, W, 3) uint8

    # 加载 RKNN
    rknn_lite = RKNNLite(verbose=False)
    print(f'[INFO] 加载模型：{model_path}')
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print(f'[ERROR] load_rknn 失败，返回码 {ret}')
        sys.exit(1)

    print('[INFO] 初始化运行时（NPU）…')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print(f'[ERROR] init_runtime 失败，返回码 {ret}')
        rknn_lite.release()
        sys.exit(1)

    # 推理
    print(f'[INFO] 开始推理（{model_type}）…')
    t0 = time.time()
    outputs = rknn_lite.inference(inputs=[img_input], data_format='nhwc')
    infer_ms = (time.time() - t0) * 1000
    print(f'[INFO] 推理完成，耗时 {infer_ms:.1f} ms')

    rknn_lite.release()

    if outputs is None or len(outputs) == 0:
        print('[ERROR] inference() 返回空结果')
        sys.exit(1)

    # ── debug 模式：打印原始输出统计，帮助诊断检测为 0 的问题 ──
    if debug:
        print('\n[DEBUG] 原始输出张量统计：')
        fmt = 'rknnopt(6输出)' if len(outputs) >= 6 else '标准ONNX(1输出)'
        print(f'  输出格式：{fmt}，共 {len(outputs)} 个张量')
        for i, o in enumerate(outputs):
            print(f'  output[{i}]: shape={list(o.shape)}  dtype={o.dtype}'
                  f'  min={o.min():.4f}  max={o.max():.4f}  mean={o.mean():.4f}')
        if model_type == 'yolov8_det' and outputs:
            if len(outputs) >= 6:
                # rknnopt 格式：打印各 scale 的 class score 统计 + Top-10
                pair = len(outputs) // 3
                print('\n[DEBUG] rknnopt 各分支 class score 统计：')
                all_cls_flat, all_scores_flat = [], []
                for i in range(3):
                    bbox_out = outputs[pair * i]       # [1,64,H,W]
                    cls_out  = outputs[pair * i + 1]   # [1,nc,H,W]
                    sig_cls = 1.0 / (1.0 + np.exp(-cls_out)) if (cls_out.max() > 1.0 or cls_out.min() < 0.0) else cls_out
                    h, w = cls_out.shape[2], cls_out.shape[3]
                    # [1,nc,H,W] → [H*W, nc]
                    cls_hw = sig_cls[0].transpose(1, 2, 0).reshape(-1, sig_cls.shape[1])
                    best_score = cls_hw.max(axis=1)
                    best_cls   = cls_hw.argmax(axis=1)
                    print(f'  branch[{i}] box:{list(bbox_out.shape)}  cls:{list(cls_out.shape)}'
                          f'  max_prob={float(sig_cls.max()):.4f}  top_anchor_score={float(best_score.max()):.4f}')
                    all_cls_flat.append(best_cls)
                    all_scores_flat.append(best_score)
                # 合并所有分支打印 Top-10
                all_scores = np.concatenate(all_scores_flat)
                all_cls    = np.concatenate(all_cls_flat)
                top_idx  = np.argsort(all_scores)[::-1][:10]
                print(f'\n[DEBUG] Top10 置信度（rknnopt，已 sigmoid）：')
                for k in top_idx:
                    print(f'  anchor {int(k):5d}  cls={int(all_cls[k])}'
                          f'  prob={float(all_scores[k]):.4f}')
            else:
                # 标准格式：单张量 [1, 4+nc, 8400]
                pred = outputs[0]
                if pred.ndim == 3:
                    pred = pred[0]   # → [4+nc, 8400]
                print(f'\n[DEBUG] 逐通道统计（shape={list(pred.shape)}）：')
                for ch in range(min(pred.shape[0], 8)):   # 只打印前8通道避免刷屏
                    row = pred[ch].flatten()
                    print(f'  ch[{ch}]: min={float(row.min()):.4f}  max={float(row.max()):.4f}'
                          f'  mean={float(row.mean()):.4f}  非零数={int(np.count_nonzero(row))}')
                # pred 应为 [4+nc, N]，转置为 [N, 4+nc]
                if pred.ndim == 2:
                    pred_t = pred.T            # [N, 4+nc]
                    nc = pred_t.shape[1] - 4
                    if nc > 0:
                        class_scores_raw = pred_t[:, 4:]
                        cls_ids_d = np.argmax(class_scores_raw, axis=1)
                        raw = class_scores_raw[np.arange(len(cls_ids_d)), cls_ids_d]
                        sig = 1.0 / (1.0 + np.exp(-raw)) if (raw.max() > 1.0 or raw.min() < 0.0) else raw
                        top_idx = np.argsort(sig)[::-1][:10]
                        note = '（logit→sigmoid）' if (raw.max() > 1.0 or raw.min() < 0.0) else '（已是概率）'
                        print(f'\n[DEBUG] Top10 置信度 {note}（nc={nc}）：')
                        for k in top_idx:
                            print(f'  anchor {int(k):5d}  cls={int(cls_ids_d[k])}'
                                  f'  raw={float(raw[k]):.4f}  prob={float(sig[k]):.4f}')
                else:
                    print(f'  [WARN] pred.ndim={pred.ndim}，非标准单输出格式，跳过 Top10 统计')
        print()

    # 后处理
    oh, ow = img_bgr.shape[:2]
    if model_type == 'yolov8_det':
        result, summary, dets = postprocess_det(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names)
    elif model_type == 'yolov8_seg':
        result, summary, dets = postprocess_seg(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names)
    elif model_type == 'yolov8_pose':
        result, summary, dets = postprocess_pose(outputs, img_bgr, scale, pad_x, pad_y, conf, iou)
    elif model_type == 'yolov8_obb':
        result, summary, dets = postprocess_obb(outputs, img_bgr, scale, pad_x, pad_y, conf, iou, names)
    elif model_type == 'resnet':
        result, summary, dets = postprocess_resnet(outputs, img_bgr, names)
    elif model_type == 'retinaface':
        # 简化：直接展示各输出张量形状
        result = img_bgr.copy()
        lines = ['RetinaFace 输出张量：']
        for i, o in enumerate(outputs):
            lines.append(f'  output[{i}]: shape={list(o.shape)}'
                         f'  min={o.min():.3f}  max={o.max():.3f}')
        summary = '\n'.join(lines)
        dets = []
    else:
        result = img_bgr.copy()
        lines = [f'未知模型类型 {model_type}，原始输出摘要：']
        for i, o in enumerate(outputs):
            lines.append(f'  output[{i}]: shape={list(o.shape)}')
        summary = '\n'.join(lines)
        dets = []

    # 保存 / 显示
    print()
    print('─' * 50)
    print(summary)
    print('─' * 50)

    cv2.imwrite(out_path, result)
    print(f'\n[INFO] 结果已保存到：{out_path}')


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RK3576 设备端 RKNN 推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model',   required=True,  help='RKNN 模型路径')
    parser.add_argument('--image',   required=True,  help='测试图片路径')
    parser.add_argument('--type',    default='yolov8_det',
                        choices=['yolov8_det', 'yolov8_seg', 'yolov8_pose',
                                 'yolov8_obb', 'resnet', 'retinaface'],
                        help='模型类型（默认 yolov8_det）')
    parser.add_argument('--conf',    type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou',     type=float, default=0.45, help='NMS IoU 阈值')
    parser.add_argument('--classes', default='',
                        help='类别名称，逗号分隔，例：fire,smoke（空则用 cls0/cls1/…）')
    parser.add_argument('--width',   type=int, default=640, help='模型输入宽度（默认 640）')
    parser.add_argument('--height',  type=int, default=640, help='模型输入高度（默认 640）')
    parser.add_argument('--output',  default='result.jpg', help='输出图片路径（默认 result.jpg）')
    parser.add_argument('--debug',   action='store_true',
                        help='打印原始输出张量统计信息，用于诊断检测为 0 的问题')

    run(parser.parse_args())
