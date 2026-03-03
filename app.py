"""
PT to RKNN 转换工具 - Flask Web服务 v2
支持：YOLOv8-Det/Seg/Pose/OBB、ResNet、RetinaFace
"""
import os
import sys
import re
import time
import json
import queue
import socket
import logging
import threading
import uuid
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask import stream_with_context
from werkzeug.utils import secure_filename
from converter import UniversalConverter, pt_to_onnx
from model_registry import MODEL_REGISTRY, get_model_types_meta, validate_file_ext, validate_pt_task
from calibration_builder import build_calibration_dataset, get_calibration_status, detect_dataset_format, normalize_path, link_calibration_dataset
from inferencer import run_inference, img_to_base64, run_accuracy_analysis
try:
    import netron
    NETRON_AVAILABLE = True
except ImportError:
    NETRON_AVAILABLE = False

# ─── 异步转换任务管理 ───────────────────────────────────
_jobs = {}          # job_id -> {q, thread_id, done}
_jobs_lock = threading.Lock()

def _get_job_by_thread():
    tid = threading.current_thread().ident
    with _jobs_lock:
        for job in _jobs.values():
            if job.get('thread_id') == tid:
                return job
    return None

def _job_put(job, event_type, data):
    if job:
        job['q'].put((event_type, data))

def _parse_progress_from_line(job, line):
    """从 RKNN tqdm / logger 行解析进度百分比并推送"""
    if not line:
        return
    # tqdm 格式: "I Quantizating :  24%|████..."
    if ('%|' in line or '% |' in line) and re.search(r'\d+%', line):
        pct_m = re.search(r'(\d+)%', line)
        if pct_m:
            pct = int(pct_m.group(1))
            if 'Quantizat' in line:
                overall = 37 + int(pct * 0.46)   # 37-83%
                _job_put(job, 'progress', overall)
            elif 'GraphPreparing' in line:
                _job_put(job, 'progress', 30 + int(pct * 0.05))
    # 关键文字进度节点
    elif 'rknnopt' in line and '导出完成' in line:
        _job_put(job, 'progress', 18)
    elif 'load_pytorch' in line or 'load_onnx' in line or '加载 torchscript' in line or '加载 ONNX' in line:
        _job_put(job, 'progress', 22)
    elif '构建 RKNN' in line or 'building' in line.lower():
        if 'done' in line.lower() or '完成' in line:
            _job_put(job, 'progress', 86)
        else:
            _job_put(job, 'progress', 28)
    elif 'export_rknn' in line or '导出：' in line:
        _job_put(job, 'progress', 90)
    elif '完成 ✓' in line or 'RKNN 成功' in line:
        _job_put(job, 'progress', 95)

class _TeeWriter:
    """将 stderr/stdout 写入同时推送到当前线程对应的 job queue"""
    def __init__(self, orig):
        self._orig = orig
    def write(self, s):
        self._orig.write(s)
        if not s or s.isspace():
            return
        job = _get_job_by_thread()
        if job:
            lines = re.sub(r'\r', '\n', s).split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    job['q'].put(('log', line))
                    _parse_progress_from_line(job, line)
    def flush(self):
        self._orig.flush()
    def fileno(self):
        try: return self._orig.fileno()
        except: return -1
    def isatty(self):
        return False

class _QueueLogHandler(logging.Handler):
    """将 Python logger 消息推送到当前线程对应的 job queue"""
    def emit(self, record):
        try:
            msg = self.format(record)
            job = _get_job_by_thread()
            if job:
                job['q'].put(('log', msg))
                _parse_progress_from_line(job, msg)
        except Exception:
            pass

# 安装全局拦截器（只安装一次）
_orig_stderr = sys.stderr
sys.stderr = _TeeWriter(_orig_stderr)
_queue_handler = _QueueLogHandler()
_queue_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logging.getLogger().addHandler(_queue_handler)
# ──────────────────────────────────────────────────────────


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'
app.config['CALIBRATION_FOLDER'] = './calibration_data'

# 确保必要的目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['CALIBRATION_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'pt', 'pth', 'onnx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/model_types', methods=['GET'])
def get_model_types():
    """获取支持的模型类型列表"""
    return jsonify({'model_types': get_model_types_meta()})


@app.route('/api/validate', methods=['POST'])
def validate_model():
    """校验上传文件是否与所选模型类型匹配"""
    if 'model_file' not in request.files:
        return jsonify({'valid': False, 'message': '未提供模型文件'}), 400

    file = request.files['model_file']
    model_type = request.form.get('model_type', '')
    if not model_type:
        return jsonify({'valid': False, 'message': '未提供模型类型'}), 400

    # 扩展名校验
    ok, msg = validate_file_ext(model_type, file.filename)
    if not ok:
        return jsonify({'valid': False, 'message': msg})

    # PT 文件进一步 task 校验（需要先保存）
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext in ('pt', 'pth'):
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                f"_validate_{int(time.time())}_{secure_filename(file.filename)}")
        file.save(tmp_path)
        try:
            ok, msg = validate_pt_task(model_type, tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return jsonify({'valid': ok, 'message': msg})

    return jsonify({'valid': True, 'message': '✅ 文件格式校验通过'})


@app.route('/api/platforms', methods=['GET'])
def get_platforms():
    """获取支持的平台列表"""
    platforms = [
        {'value': 'rk3562', 'label': 'RK3562'},
        {'value': 'rk3566', 'label': 'RK3566'},
        {'value': 'rk3568', 'label': 'RK3568'},
        {'value': 'rk3576', 'label': 'RK3576'},
        {'value': 'rk3588', 'label': 'RK3588'}
    ]
    return jsonify({'platforms': platforms})


@app.route('/api/convert', methods=['POST'])
def convert_model():
    """处理模型转换请求（异步，返回 job_id）"""
    if 'model_file' not in request.files:
        return jsonify({'success': False, 'message': '未上传模型文件'}), 400
    file = request.files['model_file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': '只支持 .pt / .pth / .onnx 文件'}), 400

    model_type   = request.form.get('model_type', 'yolov8_det')
    platform     = request.form.get('platform', 'rk3576')
    quant_type   = request.form.get('quant_type', 'i8')
    input_width  = int(request.form.get('input_width', 640))
    input_height = int(request.form.get('input_height', 640))

    ok, msg = validate_file_ext(model_type, file.filename)
    if not ok:
        return jsonify({'success': False, 'message': msg}), 400

    filename  = secure_filename(file.filename)
    timestamp = int(time.time())
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
    file.save(upload_path)

    model_name      = os.path.splitext(filename)[0]
    output_filename = f"{model_name}_{model_type}_{platform}_{quant_type}_{timestamp}.rknn"
    output_path     = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    do_quant        = (quant_type == 'i8')

    job_id = uuid.uuid4().hex[:10]
    job = {'q': queue.Queue(), 'thread_id': None, 'done': False}
    with _jobs_lock:
        _jobs[job_id] = job

    def _run():
        job['thread_id'] = threading.current_thread().ident
        job['q'].put(('progress', 5))
        job['q'].put(('log', f'▶ 开始转换：{filename} → {output_filename}'))
        try:
            converter = UniversalConverter(verbose=True)
            success, message, onnx_out = converter.convert(
                model_type=model_type,
                input_path=upload_path,
                platform=platform,
                do_quant=do_quant,
                calibration_dir=app.config['CALIBRATION_FOLDER'],
                output_path=output_path,
                input_size=(input_height, input_width),
            )
            if success:
                # rknnopt 路径不产生 ONNX，补充导出供 x86 模拟推理使用
                if not onnx_out and upload_path.lower().endswith(('.pt', '.pth')):
                    job['q'].put(('log', '▶ 补充导出 ONNX（x86 模拟推理用）...'))
                    _ok, _msg, _onnx = pt_to_onnx(
                        pt_path=upload_path,
                        input_size=(input_height, input_width),
                        tmp_dir=app.config['OUTPUT_FOLDER'],
                    )
                    if _ok:
                        onnx_out = _onnx
                        job['q'].put(('log', f'✔ ONNX 已生成：{_onnx}'))
                    else:
                        job['q'].put(('log', f'⚠ ONNX 生成失败（{_msg}），x86 推理不可用'))
                cfg = MODEL_REGISTRY[model_type]
                meta = {
                    'model_type': model_type,
                    'input_w': input_width, 'input_h': input_height,
                    'platform': platform, 'quant_type': quant_type,
                    'class_names': [], 'onnx_path': onnx_out,
                    'mean_values': cfg['mean_values'], 'std_values': cfg['std_values'],
                }
                meta_path = output_path + '.meta.json'
                try:
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(meta, mf, ensure_ascii=False, indent=2)
                except Exception:
                    pass
                job['q'].put(('progress', 100))
                job['q'].put(('done', {
                    'success': True, 'message': message,
                    'output_file': output_filename,
                    'download_url': f'/api/download/{output_filename}'
                }))
            else:
                job['q'].put(('done', {'success': False, 'message': message}))
        except Exception as e:
            job['q'].put(('done', {'success': False, 'message': f'转换失败: {e}'}))
        finally:
            job['done'] = True
            if os.path.exists(upload_path):
                try: os.remove(upload_path)
                except: pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return jsonify({'started': True, 'job_id': job_id})


@app.route('/api/convert/log/<job_id>')
def convert_log(job_id):
    """SSE 端点：流式推送转换日志和进度"""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': '任务不存在'}), 404

    def generate():
        while True:
            try:
                event_type, data = job['q'].get(timeout=25)
                payload = json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)
                yield 'data: ' + payload + '\n\n'
                if event_type == 'done':
                    break
            except queue.Empty:
                # 心跳，防止连接超时
                yield 'data: ' + json.dumps({'type': 'ping'}) + '\n\n'
                if job['done']:
                    break
        # 任务结束后稍后清理
        def _cleanup():
            time.sleep(60)
            with _jobs_lock:
                _jobs.pop(job_id, None)
        threading.Thread(target=_cleanup, daemon=True).start()

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


def get_free_port():
    """获取一个空闲端口"""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


@app.route('/api/calibration/status', methods=['GET'])
def calibration_status():
    """查询各模型类型校准数据集是否就绪"""
    model_type = request.args.get('model_type', '')
    if model_type not in MODEL_REGISTRY:
        return jsonify({'success': False, 'message': '未知模型类型'}), 400
    subdir = MODEL_REGISTRY[model_type]['calibration_subdir']
    status = get_calibration_status(app.config['CALIBRATION_FOLDER'], subdir)
    return jsonify({'success': True, **status})


@app.route('/api/calibration/prepare', methods=['POST'])
def calibration_prepare():
    """从用户提供的数据集路径提取校准图片、生成 dataset.txt"""
    data = request.get_json()
    model_type  = data.get('model_type', '')
    dataset_path = data.get('dataset_path', '').strip()
    max_images  = int(data.get('max_images', 50))

    if model_type not in MODEL_REGISTRY:
        return jsonify({'success': False, 'message': '未知模型类型'}), 400
    if not dataset_path:
        return jsonify({'success': False, 'message': '未提供数据集路径'}), 400

    # 先规范化路径（支持 Windows 路径），再探测格式
    dataset_path = normalize_path(dataset_path)
    fmt, fmt_desc = detect_dataset_format(dataset_path)
    if fmt in ('invalid', 'empty'):
        return jsonify({'success': False, 'message': fmt_desc}), 400

    subdir = MODEL_REGISTRY[model_type]['calibration_subdir']
    output_dir = os.path.join(app.config['CALIBRATION_FOLDER'], subdir)

    ok, msg, count = build_calibration_dataset(
        dataset_path=dataset_path,
        output_dir=output_dir,
        model_type=model_type,
        max_images=max_images,
    )
    return jsonify({
        'success': ok,
        'message': msg,
        'count': count,
        'format': fmt_desc,
    })


@app.route('/api/calibration/link', methods=['POST'])
def calibration_link():
    """在 calibration_data/<subdir>/images 创建软链接指向外部目录，不复制文件"""
    data = request.get_json()
    model_type   = data.get('model_type', '')
    dataset_path = data.get('dataset_path', '').strip()

    if model_type not in MODEL_REGISTRY:
        return jsonify({'success': False, 'message': '未知模型类型'}), 400
    if not dataset_path:
        return jsonify({'success': False, 'message': '未提供数据集路径'}), 400

    dataset_path = normalize_path(dataset_path)
    force = bool(data.get('force', False))
    subdir = MODEL_REGISTRY[model_type]['calibration_subdir']
    output_dir = os.path.join(app.config['CALIBRATION_FOLDER'], subdir)

    ok, msg, count = link_calibration_dataset(dataset_path, output_dir, force=force)
    return jsonify({'success': ok, 'message': msg, 'count': count})


@app.route('/api/calibration/detect', methods=['POST'])
def calibration_detect():
    """仅探测数据集路径的格式，不做任何复制"""
    data = request.get_json()
    path = data.get('dataset_path', '').strip()
    if not path:
        return jsonify({'success': False, 'message': '路径为空'}), 400
    resolved = normalize_path(path)
    fmt, desc = detect_dataset_format(resolved)
    resp = {'success': fmt not in ('invalid', 'empty'), 'format': fmt, 'description': desc}
    if resolved != path:
        resp['resolved_path'] = resolved
        resp['description'] = desc + f'（路径已转换为 WSL：{resolved}）'
    return jsonify(resp)


@app.route('/api/preview', methods=['POST'])
def preview_model():
    """启动 Netron 可视化服务预览模型结构"""
    if not NETRON_AVAILABLE:
        return jsonify({'success': False, 'message': '未安装 netron，请运行: pip install netron'}), 503
    try:
        data = request.get_json()
        filename = secure_filename(data.get('filename', ''))
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'}), 404

        port = get_free_port()
        host, port = netron.start(file_path, address=('0.0.0.0', port), browse=False)
        return jsonify({'success': True, 'url': f'http://localhost:{port}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'预览启动失败: {str(e)}'}), 500


# ─────────────────────────────────────────────────────────────
# 推理测试接口
# ─────────────────────────────────────────────────────────────

@app.route('/api/meta/<filename>', methods=['GET'])
def get_model_meta(filename):
    """读取 RKNN 模型的元数据 JSON"""
    meta_path = os.path.join(app.config['OUTPUT_FOLDER'], filename + '.meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'success': False, 'message': '元数据不存在'}), 404
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return jsonify({'success': True, **meta})


@app.route('/api/meta/<filename>', methods=['POST'])
def update_model_meta(filename):
    """更新模型元数据（如 class_names）"""
    meta_path = os.path.join(app.config['OUTPUT_FOLDER'], filename + '.meta.json')
    if not os.path.exists(meta_path):
        return jsonify({'success': False, 'message': '元数据不存在'}), 404
    data = request.get_json()
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for k in ('class_names', 'input_w', 'input_h'):
        if k in data:
            meta[k] = data[k]
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return jsonify({'success': True})


@app.route('/api/infer', methods=['POST'])
def infer_model():
    """上传图片，使用指定 RKNN 模型在模拟器模式下推理，返回可视化结果"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '未上传图片'}), 400

        rknn_filename = secure_filename(request.form.get('rknn_filename', ''))
        if not rknn_filename or not rknn_filename.endswith('.rknn'):
            return jsonify({'success': False, 'message': '未指定 RKNN 文件名'}), 400

        rknn_path = os.path.join(app.config['OUTPUT_FOLDER'], rknn_filename)
        if not os.path.exists(rknn_path):
            return jsonify({'success': False, 'message': 'RKNN 文件不存在'}), 404

        # 读取元数据
        meta_path = rknn_path + '.meta.json'
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

        model_type  = request.form.get('model_type')  or meta.get('model_type', 'yolov8_det')
        input_w     = int(request.form.get('input_w')  or meta.get('input_w', 640))
        input_h     = int(request.form.get('input_h')  or meta.get('input_h', 640))
        conf_thresh = float(request.form.get('conf_thresh', 0.25))
        iou_thresh  = float(request.form.get('iou_thresh', 0.45))
        class_names_raw = request.form.get('class_names', '')
        if class_names_raw.strip():
            class_names = [n.strip() for n in class_names_raw.split(',') if n.strip()]
        else:
            class_names = meta.get('class_names') or []

        # 读取上传图片
        import numpy as np
        img_file = request.files['image']
        file_bytes = img_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        import cv2
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'success': False, 'message': '无法解码图片，请上传 JPG/PNG/BMP'}), 400

        result_bgr, summary, detections, infer_ms = run_inference(
            rknn_path=rknn_path,
            img_bgr=img_bgr,
            model_type=model_type,
            input_w=input_w,
            input_h=input_h,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            class_names=class_names if class_names else None,
            onnx_path=meta.get('onnx_path', ''),
            mean_values=meta.get('mean_values'),
            std_values=meta.get('std_values'),
            platform=meta.get('platform', 'rk3576'),
        )

        img_b64 = img_to_base64(result_bgr)
        return jsonify({
            'success': True,
            'image_b64': img_b64,
            'summary': summary,
            'detections': detections,
            'infer_ms': round(infer_ms, 1),
        })

    except RuntimeError as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': '推理失败：{}'.format(str(e))}), 500



@app.route('/api/accuracy', methods=['POST'])
def accuracy_analysis_endpoint():
    """量化精度分析：对比 ONNX 与 RKNN 逐层输出，返回余弦相似度报告"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': '未上传图片'}), 400

        rknn_filename = secure_filename(request.form.get('rknn_filename', ''))
        if not rknn_filename or not rknn_filename.endswith('.rknn'):
            return jsonify({'success': False, 'message': '未指定 RKNN 文件名'}), 400

        rknn_path = os.path.join(app.config['OUTPUT_FOLDER'], rknn_filename)
        if not os.path.exists(rknn_path):
            return jsonify({'success': False, 'message': 'RKNN 文件不存在'}), 404

        meta_path = rknn_path + '.meta.json'
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

        onnx_path = meta.get('onnx_path', '')
        if not onnx_path or not os.path.exists(onnx_path):
            return jsonify({'success': False,
                            'message': 'ONNX 文件不存在，请重新转换以生成配套 ONNX'}), 404

        input_w    = int(meta.get('input_w', 640))
        input_h    = int(meta.get('input_h', 640))
        platform   = meta.get('platform', 'rk3576')
        quant_type = meta.get('quant_type', 'fp')
        mean_values = meta.get('mean_values')
        std_values  = meta.get('std_values')

        # 是否进行 INT8 量化对比（前端可选，且模型本身须为 i8）
        do_quant = request.form.get('do_quantization', 'false').lower() == 'true'
        if do_quant and quant_type != 'i8':
            do_quant = False   # 非 i8 模型无需量化对比

        # 校准数据集（仅 do_quant=True 时生效）
        dataset_path = None
        if do_quant:
            ds = meta.get('dataset_path', '')
            if ds and os.path.exists(ds):
                dataset_path = ds

        import numpy as np
        import cv2
        nparr = np.frombuffer(request.files['image'].read(), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'success': False, 'message': '无法解码图片'}), 400

        result = run_accuracy_analysis(
            onnx_path=onnx_path,
            img_bgr=img_bgr,
            input_w=input_w,
            input_h=input_h,
            mean_values=mean_values,
            std_values=std_values,
            platform=platform,
            do_quantization=do_quant,
            dataset_path=dataset_path,
        )
        return jsonify({'success': True, **result})

    except RuntimeError as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': '精度分析失败：{}'.format(str(e))}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载转换后的文件"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'success': False, 'message': f'下载失败: {str(e)}'}), 500


@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_output(filename):
    """删除单个转换输出文件（.rknn + .meta.json）"""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        file_path = os.path.join(output_folder, filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '文件不存在'}), 404
        os.remove(file_path)
        meta_path = file_path + '.meta.json'
        if os.path.exists(meta_path):
            os.remove(meta_path)
        return jsonify({'success': True, 'message': f'{filename} 已删除'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'}), 500


@app.route('/api/outputs/clear', methods=['POST'])
def clear_outputs():
    """清空全部转换历史（output 目录下所有 .rknn 和 .meta.json）"""
    try:
        output_folder = app.config['OUTPUT_FOLDER']
        removed = 0
        for fname in os.listdir(output_folder):
            if fname.endswith('.rknn') or fname.endswith('.meta.json') or fname.endswith('.onnx'):
                os.remove(os.path.join(output_folder, fname))
                removed += 1
        return jsonify({'success': True, 'message': f'已清空 {removed} 个文件'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'清空失败: {str(e)}'}), 500



@app.route('/api/outputs', methods=['GET'])
def list_outputs():
    """列出所有输出文件"""
    try:
        files = []
        output_folder = app.config['OUTPUT_FOLDER']
        
        for filename in os.listdir(output_folder):
            if filename.endswith('.rknn'):
                file_path = os.path.join(output_folder, filename)
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                file_time = os.path.getmtime(file_path)
                # 读取元数据（如有）
                meta = {}
                meta_path = file_path + '.meta.json'
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as mf:
                            meta = json.load(mf)
                    except Exception:
                        pass
                files.append({
                    'filename': filename,
                    'size': f'{file_size:.2f} MB',
                    'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time)),
                    'download_url': f'/api/download/{filename}',
                    'model_type': meta.get('model_type', ''),
                    'input_w': meta.get('input_w', 640),
                    'input_h': meta.get('input_h', 640),
                })
        
        # 按时间倒序排序
        files.sort(key=lambda x: x['time'], reverse=True)
        return jsonify({'success': True, 'files': files})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取文件列表失败: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("PT to RKNN 转换工具已启动")
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
