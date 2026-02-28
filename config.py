# PT to RKNN 转换工具 - 配置文件

# 服务器配置
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# 文件上传配置
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'pt', 'pth'}

# 目录配置
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './output'
CALIBRATION_FOLDER = './calibration_data'

# 转换默认参数
DEFAULT_PLATFORM = 'rk3576'
DEFAULT_QUANT_TYPE = 'i8'
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_MEAN_VALUES = [[0, 0, 0]]
DEFAULT_STD_VALUES = [[255, 255, 255]]
DEFAULT_OPTIMIZATION_LEVEL = 3

# 支持的平台列表
SUPPORTED_PLATFORMS = [
    'rk3562',
    'rk3566', 
    'rk3568',
    'rk3576',
    'rk3588'
]
