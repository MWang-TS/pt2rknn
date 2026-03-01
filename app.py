"""
PT to RKNN 转换工具 - Flask Web服务
"""
import os
import time
import json
import socket
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from converter import PT2RKNNConverter
try:
    import netron
    NETRON_AVAILABLE = True
except ImportError:
    NETRON_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['OUTPUT_FOLDER'] = './output'
app.config['CALIBRATION_FOLDER'] = './calibration_data'

# 确保必要的目录存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['CALIBRATION_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'pt', 'pth'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


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
    """处理模型转换请求"""
    try:
        # 检查文件
        if 'model_file' not in request.files:
            return jsonify({'success': False, 'message': '未上传模型文件'}), 400
        
        file = request.files['model_file']
        if file.filename == '':
            return jsonify({'success': False, 'message': '未选择文件'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': '只支持.pt和.pth文件'}), 400
        
        # 获取参数
        platform = request.form.get('platform', 'rk3576')
        quant_type = request.form.get('quant_type', 'i8')
        input_width = int(request.form.get('input_width', 640))
        input_height = int(request.form.get('input_height', 640))
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        upload_filename = f"{timestamp}_{filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
        file.save(upload_path)
        
        # 设置输出路径
        model_name = os.path.splitext(filename)[0]
        output_filename = f"{model_name}_{platform}_{quant_type}_{timestamp}.rknn"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 设置校准数据集（量化时需要）
        do_quant = (quant_type == 'i8')
        dataset_path = None
        if do_quant:
            dataset_path = os.path.join(app.config['CALIBRATION_FOLDER'], 'calibration.txt')
            if not os.path.exists(dataset_path):
                # 如果没有校准数据集，使用默认的或提示用户
                return jsonify({
                    'success': False, 
                    'message': '量化模式需要校准数据集。请在calibration_data目录下准备calibration.txt文件'
                }), 400
        
        # 执行转换
        converter = PT2RKNNConverter(verbose=True)
        success, message, output_file = converter.convert(
            pt_model_path=upload_path,
            platform=platform,
            do_quant=do_quant,
            dataset_path=dataset_path,
            output_path=output_path,
            input_size=(input_height, input_width)
        )
        
        # 清理上传的文件
        try:
            os.remove(upload_path)
        except:
            pass
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'output_file': output_filename,
                'download_url': f'/api/download/{output_filename}'
            })
        else:
            return jsonify({'success': False, 'message': message}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'转换失败: {str(e)}'}), 500


def get_free_port():
    """获取一个空闲端口"""
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


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
                
                files.append({
                    'filename': filename,
                    'size': f'{file_size:.2f} MB',
                    'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time)),
                    'download_url': f'/api/download/{filename}'
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
