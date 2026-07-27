# PT → RKNN 多模型转换工具 🚀 `v0.0.3`

一个基于 Web 界面的模型转换工具，将 PyTorch (.pt/.pth) 或 ONNX 模型转换为 RKNN 格式，专为 Rockchip NPU 设备优化。

## ✨ 特性

- 🖥️ **Web 界面** — 三步骤卡片式 UI，无需命令行操作
- 🤖 **多网络类型** — 支持 YOLOv8-Det / Seg / Pose / OBB、ResNet、RetinaFace
- 📂 **智能格式识别** — 上传文件后自动校验扩展名与网络类型是否匹配
- 🔄 **自动转换链路** — PT → rknnopt TorchScript → RKNN（INT8 量化精度更佳）或 PT → ONNX → RKNN
- 📡 **实时日志流** — 转换过程通过 SSE 实时推送日志与进度条
- 📊 **INT8 校准数据集准备** — 指定训练数据路径，工具自动探测格式、提取图片、生成 dataset.txt
- 👁️ **Netron 预览** — 在线可视化 RKNN / ONNX 模型结构
- 📦 **历史记录** — 查看、电脑端 ONNX 基线、转换图误差分析、单条删除或一键清空转换结果

---

## 📐 支持的网络类型

| 类型 | 图标 | 接受格式 | 默认输入尺寸 | 校准数据目录 |
|------|------|----------|-------------|-------------|
| YOLOv8-Det | 🎯 | .pt / .onnx | 640×640 | `calibration_data/yolov8_det/` |
| YOLOv8-Seg | ✂️ | .pt / .onnx | 640×640 | `calibration_data/yolov8_seg/` |
| YOLOv8-Pose | 🤸 | .pt / .onnx | 640×640 | `calibration_data/yolov8_pose/` |
| YOLOv8-OBB | 🔷 | .pt / .onnx | 640×640 | `calibration_data/yolov8_obb/` |
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

> 本工具支持两种运行方式：**源码直接运行（WSL）** 和 **Docker 容器**，路径转换规则有所不同，请按实际方式配置。

---

### 方式一：源码运行（WSL / Linux）

```bash
# 克隆仓库
git clone https://github.com/MWang-TS/pt2rknn.git
cd pt2rknn

# 激活环境
conda activate rk-y8

# 启动 Web 服务
python app.py
# 默认监听 http://0.0.0.0:5600
```

打开浏览器访问 **http://localhost:5600** 即可使用。

#### 路径配置说明（WSL 模式）

| 项目 | 规则 |
|------|------|
| Windows 输入路径 | 自动转换为 WSL `/mnt/` 格式：`E:\data` → `/mnt/e/data` |
| INT8 校准数据存储 | `~/pt2rknn_calibration/`（WSL 本地文件系统，避免 NTFS 权限问题）|
| 上传 / 输出目录 | 相对路径 `./uploads` / `./output`（在项目目录下）|

> **注意**：WSL 默认挂载 Windows NTFS 分区无写权限，因此校准图片会复制到 WSL 本地 `~` 目录下，而不是 Windows 路径。

---

### 方式二：Docker 容器

```bash
cd pt2rknn

# 构建并启动
chmod +x docker-build.sh
./docker-build.sh
docker-compose up -d
```

打开浏览器访问 **http://localhost:5600** 即可使用。

#### 路径配置说明（Docker 模式）

| 项目 | 规则 |
|------|------|
| Windows 输入路径 | 自动转换为容器内路径：`E:\data` → `/e/data` |
| INT8 校准数据存储 | `./calibration_data`（已通过 volume 挂载到容器内 `/app/calibration_data`）|
| 上传 / 输出目录 | `./uploads` / `./output`（volume 挂载）|

docker-compose.yml 默认映射 Windows 盘符（只读访问宿主数据集）：

```yaml
volumes:
  - c:/:/c:ro   # C: → /c
  - d:/:/d:ro   # D: → /d
  - e:/:/e:ro   # E: → /e
```

Linux 宿主机若需访问整个文件系统，取消注释：

```yaml
# - /:/host:ro
```

详细 Docker 部署说明请参阅 [README-Docker.md](README-Docker.md)。

---

## 🗂️ 项目结构

```
pt2rknn_tool/
├── app.py                   # Flask Web 服务入口
├── converter.py             # 转换引擎（UniversalConverter）
├── model_registry.py        # 6 种网络类型配置注册表
├── calibration_builder.py   # 校准数据集自动构建工具
├── infer_on_device.py       # RK35xx 设备端单图推理
├── device_validate.py       # RK35xx YOLOv8-Det 批量验收报告
├── IMPLEMENTATION_PLAN.md   # 转换、迁移和验收执行方案
├── requirements.txt
├── templates/
│   └── index.html           # 前端（多步骤卡片 UI）
├── uploads/                 # 上传临时目录
├── output/                  # RKNN、元数据和转换图输出目录
└── calibration_data/        # INT8 校准图片目录
    ├── yolov8_det/
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

INT8 量化需要一批代表性图片用于校准。校准集缺失时转换直接失败，不会生成名称与实际精度不符的 FP 产物。

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
# 示例：为 YOLOv8 目标检测准备校准图片
cp /your/dataset/images/*.jpg calibration_data/yolov8_det/images/
```

---

## 🔌 API 接口（供二次开发）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/model_types` | 获取所有支持的网络类型元数据 |
| POST | `/api/validate` | 校验上传文件是否匹配网络类型 |
| POST | `/api/convert` | 执行模型转换（返回 job_id）|
| GET  | `/api/stream/<job_id>` | SSE 实时流式获取转换日志与进度 |
| GET  | `/api/calibration/status` | 查询指定类型的校准数据状态 |
| POST | `/api/calibration/detect` | 探测数据集路径格式 |
| POST | `/api/calibration/prepare` | 提取图片并生成 dataset.txt |
| POST | `/api/preview` | 启动 Netron 预览服务 |
| GET  | `/api/outputs` | 获取历史转换文件列表 |
| GET  | `/api/download/<filename>` | 下载 RKNN 文件 |
| DELETE | `/api/delete/<filename>` | 删除单个 RKNN 及其元数据 |
| POST | `/api/outputs/clear` | 清空全部转换历史 |
| POST | `/api/infer` | 在电脑端执行 ONNX FP 基线推理，不代表最终 RKNN 效果 |
| POST | `/api/accuracy` | 复用转换图和校准清单进行误差分析，不代表真机最终精度 |
| POST | `/api/device/test-connection` | 测试与局域网 RK35xx 设备的 SSH 连接 |
| POST | `/api/device/validate` | 通过 SSH 上传模型+脚本+图片，在真机上执行验收（返回 job_id）|
| GET  | `/api/device/validate/log/<job_id>` | SSE 实时流式获取设备验收日志与结果 |
| GET  | `/api/device/reports` | 获取已下载到本地的设备验收报告列表 |
| GET  | `/api/device/reports/<filename>` | 获取指定设备验收报告的完整内容 |

### 🔧 真机设备验收（SSH）

历史列表每个模型卡片新增「🔧 设备验收」按钮，可直接从 Web UI 通过 SSH 连接局域网内的 RK35xx 设备完成端到端验收，无需手动 scp/ssh：

1. 填写设备 IP、SSH 端口、用户名，选择密码或私钥认证并测试连接。
2. 选择图片来源：从本机上传验证图片目录，或直接使用设备上已有的图片目录。
3. 填写类别名称（留空则自动读取模型的 `.meta.json`）、置信度/IoU 阈值、NPU 预热次数。
4. 点击「开始设备验收」，工具会自动：上传 `.rknn` 模型、设备端脚本（`infer_on_device.py` / `device_validate.py`）与图片 → 远程执行批量推理 → 下载 JSON 验收报告到 `output/device-validation/` → 校验模型 SHA256 是否与本地一致。

⚠️ **安全提示**：密码/私钥仅用于建立本次 SSH 连接，不会被服务端持久化存储；该功能默认信任目标主机公钥（TOFU），仅建议在受控局域网内使用。当前 Web 服务本身未做任何身份鉴权，请勿将其暴露到公网。

---

## 📝 注意事项

- YOLOv8 `.pt` 转换需要 `ultralytics`，内部先 export 为 ONNX（opset 12）再转 RKNN
- ResNet / RetinaFace 仅接受 `.onnx` 输入（无 ultralytics 依赖）
- INT8 缺少有效校准集时转换直接失败，不会自动回退到 FP16
- 最终 RKNN 精度和性能须在 RK35xx 设备端验收；执行步骤见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- `device_validate.py` 当前仅支持 YOLOv8-Det 批量预测与延迟报告
- Netron 预览需要安装 `netron`：`pip install netron`

---

## 📝 版本历史

### v0.0.3 (2026-07-27)

- **RK35xx 真机验收**：新增 `device_ssh.py` / `device_validate.py`，通过 SSH/SFTP 连接局域网设备，上传模型和验收脚本、远程批量推理、下载 JSON 报告并校验模型哈希
- **NPU 权限预检**：连接设备时自动检测 `/dev/dri/by-path/*npu*` 节点可访问性，权限不足时给出明确提示，避免误判为驱动缺失
- **设备端目录持久展示**：验收摘要卡片新增“设备端目录”字段（`user@host:path`），不再只依赖滚动日志
- **转换结果可追溯**：新增源模型 SHA256、校准数据集清单哈希、量化参数（`quantized_algorithm` / `quantized_method` / `optimization_level`）记录，写入转换元数据
- **INT8 转换不再静默回退 FP16**：缺少校准数据集时直接失败并报错，避免用户误以为量化生效
- **校准数据采样可复现**：`calibration_builder.py` 增加固定随机种子（`seed=42`），同一数据集多次采样结果一致
- **精度分析支持 rknnopt TorchScript**：`run_accuracy_analysis` 除 ONNX 外新增支持 rknnopt 转换图，分析结果标注所用转换图类型
- 修复 `model_registry.py` 中 YOLOv8-Det 校准子目录错误（`coco` → `yolov8_det`）
- 新增依赖 `paramiko==3.4.1`（设备 SSH 验收）

### v0.0.2 (2026-06-11)

- **WSL/Docker 双模式路径转换**：自动检测运行环境（`/.dockerenv`），WSL 模式 Windows 路径转为 `/mnt/x/...`，Docker 模式转为 `/x/...`
- **WSL 模式 INT8 校准数据目录**改为 `~/pt2rknn_calibration/`，彻底解决 `/mnt/` NTFS 挂载无写权限导致的文件删除/复制失败问题
- **接口异常捕获**：`calibration_prepare` / `calibration_link` 全面捕获异常，返回 JSON 而非 HTML 500 页面

### v0.0.1 (2026-03-03)

- 初始发布
- YOLOv8-Det INT8 量化采用 rknnopt TorchScript 路径（`load_pytorch`），解决各输出头共用同一 INT8 scale 导致分类分数全零的问题
- 转换过程 SSE 实时日志流 + 进度条
- 历史记录支持单条删除和一键清空
- 推理测试修复：rknnopt 转换后自动生成 ONNX 供 x86 模拟器使用
- 设备端推理脚本 (`infer_on_device.py`) 支持 rknnopt 6-output 格式，DFL 改为纯 NumPy 实现

---

## �🔗 相关资源

- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [rknn-toolkit2 文档](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLOv8](https://docs.ultralytics.com)
- [Netron 模型可视化](https://netron.app)
