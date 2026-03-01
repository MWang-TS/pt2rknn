# PT → RKNN 多模型转换工具 🚀

一个基于 Web 界面的模型转换工具，将 PyTorch (.pt/.pth) 或 ONNX 模型转换为 RKNN 格式，专为 Rockchip NPU 设备优化。

## ✨ 特性

- 🖥️ **Web 界面** — 三步骤卡片式 UI，无需命令行操作
- 🤖 **多网络类型** — 支持 YOLOv8-Det / Seg / Pose / OBB、ResNet、RetinaFace
- 📂 **智能格式识别** — 上传文件后自动校验扩展名与网络类型是否匹配
- 🔄 **自动转换链路** — PT → ONNX → RKNN，一步完成
- 📊 **INT8 校准数据集准备** — 指定训练数据路径，工具自动探测格式、提取图片、生成 dataset.txt
- 👁️ **Netron 预览** — 在线可视化 RKNN / ONNX 模型结构
- 📦 **历史记录** — 查看并下载所有转换结果

---

## 📐 支持的网络类型

| 类型 | 图标 | 接受格式 | 默认输入尺寸 | 校准数据目录 |
|------|------|----------|-------------|-------------|
| YOLOv8-Det | 🎯 | .pt / .onnx | 640×640 | `calibration_data/coco/` |
| YOLOv8-Seg | ✂️ | .pt / .onnx | 640×640 | `calibration_data/coco/` |
| YOLOv8-Pose | 🤸 | .pt / .onnx | 640×640 | `calibration_data/coco/` |
| YOLOv8-OBB | 🔷 | .pt / .onnx | 640×640 | `calibration_data/coco/` |
| ResNet | 🧱 | .onnx | 224×224 | `calibration_data/imagenet/` |
| RetinaFace | 👤 | .onnx | 640×640 | `calibration_data/face/` |

---

## 🖥️ 支持的目标平台

RK3562 / RK3566 / RK3568 / **RK3576**（默认）/ RK3588

---

## 📦 环境准备

### 前置要求

- **操作系统**: WSL (Windows Subsystem for Linux) 或 Linux x86_64
- **Python**: 3.8（推荐，rknn-toolkit2 兼容性最好）
- **Conda**: Miniconda 或 Anaconda

### 创建 Conda 环境

```bash
# 1. 创建 Python 3.8 环境
conda create -n rk-y8 python=3.8 -y
conda activate rk-y8

# 2. 安装 PyTorch（CPU 版本，用于 YOLO .pt 导出）
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. 安装 rknn-toolkit2（从官方下载 whl 安装）
pip install /path/to/rknn_toolkit2-*.whl

# 4. 安装 ultralytics（YOLO .pt 导出为 ONNX）
pip install ultralytics

# 5. 安装其他依赖
pip install -r requirements.txt
```

> ⚠️ rknn-toolkit2 仅支持 **x86_64 Linux**（含 WSL），不支持 macOS / Windows 原生环境。

---

## 🛠️ 安装 & 启动

```bash
# 克隆仓库
git clone https://github.com/MWang-TS/pt2rknn.git
cd pt2rknn

# 激活环境
conda activate rk-y8

# 启动 Web 服务
python app.py
# 默认监听 http://0.0.0.0:5000
```

打开浏览器访问 **http://localhost:5000** 即可使用。

---

## 🗂️ 项目结构

```
pt2rknn_tool/
├── app.py                   # Flask Web 服务入口
├── converter.py             # 转换引擎（UniversalConverter）
├── model_registry.py        # 6 种网络类型配置注册表
├── calibration_builder.py   # 校准数据集自动构建工具
├── requirements.txt
├── templates/
│   └── index.html           # 前端（多步骤卡片 UI）
├── uploads/                 # 上传临时目录
├── outputs/                 # RKNN 输出目录
└── calibration_data/        # INT8 校准图片目录
    ├── coco/
    │   ├── images/          # 放校准图片（或由工具自动提取）
    │   └── dataset.txt      # 工具生成
    ├── imagenet/
    │   ├── images/
    │   └── dataset.txt
    └── face/
        ├── images/
        └── dataset.txt
```

---

## 📊 INT8 校准数据集

INT8 量化需要一批代表性图片用于校准，否则自动回退到 FP16。

### 方式一：通过 UI 自动准备（推荐）

1. 在第 3 步选择 **INT8**，展开「INT8 校准数据集」面板
2. 输入服务器本地数据集路径，点击 **🔍 探测格式**
3. 工具自动识别数据集格式（支持下列格式）
4. 设置提取数量上限，点击 **✅ 确认提取并生成校准集**

**支持的数据集格式：**

| 格式 | 识别方式 |
|------|----------|
| 普通图片目录 | 目录内直接存放 `.jpg/.png/.bmp` 等 |
| YOLO 格式 | 含 `images/` 子目录 |
| ImageNet 格式 | 含按类别命名的子目录，各目录内有图片 |
| COCO 格式 | 含 `val2017/`、`train2017/` 等子目录 |
| 递归格式 | 深层嵌套任意结构（自动递归查找） |

### 方式二：手动放置

将图片直接复制到对应的 `calibration_data/<类型>/images/` 目录（无需 dataset.txt，工具启动时自动检测）：

```bash
# 示例：为 YOLOv8 类型准备 COCO 校准图片
cp /your/coco/val2017/*.jpg calibration_data/coco/images/
```

---

## 🔌 API 接口（供二次开发）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/model_types` | 获取所有支持的网络类型元数据 |
| POST | `/api/validate` | 校验上传文件是否匹配网络类型 |
| POST | `/api/convert` | 执行模型转换 |
| GET  | `/api/calibration/status` | 查询指定类型的校准数据状态 |
| POST | `/api/calibration/detect` | 探测数据集路径格式 |
| POST | `/api/calibration/prepare` | 提取图片并生成 dataset.txt |
| POST | `/api/preview` | 启动 Netron 预览服务 |
| GET  | `/api/outputs` | 获取历史转换文件列表 |
| GET  | `/api/download/<filename>` | 下载 RKNN 文件 |

---

## 📝 注意事项

- YOLOv8 `.pt` 转换需要 `ultralytics`，内部先 export 为 ONNX（opset 12）再转 RKNN
- ResNet / RetinaFace 仅接受 `.onnx` 输入（无 ultralytics 依赖）
- INT8 Without calibration data → 自动 fallback 到 FP16，转换日志会有提示
- Netron 预览需要安装 `netron`：`pip install netron`

---

## 🔗 相关资源

- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [rknn-toolkit2 文档](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Netron 模型可视化](https://netron.app)
