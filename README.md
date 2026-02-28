# PT to RKNN 模型转换工具 🚀

一个简单易用的 Web 工具，用于将 PyTorch (.pt/.pth) 模型转换为 RKNN 格式，专为 Rockchip NPU 设备优化。

**✨ 新功能：自动TorchScript转换！**  
上传普通PT模型后，工具会自动转换为TorchScript，再转换为RKNN，**一步到位，无需手动操作！**

## ✨ 特性

- 🖥️ **Web 界面** - 无需命令行操作，通过浏览器完成转换
- 🔄 **自动转换** - 自动将PT模型转换为TorchScript，再转为RKNN（一步完成）
- 📤 **拖拽上传** - 支持拖拽文件或点击选择
- ⚙️ **灵活配置** - 支持多平台、量化选项、自定义输入尺寸
- 📦 **历史记录** - 查看和下载之前的转换结果
- 🎯 **简化依赖** - 只包含转换必需的文件，轻量高效

## 📋 支持的平台

- RK3562
- RK3566
- RK3568
- RK3576 (默认)
- RK3588

## � 环境准备

### 前置要求

- **操作系统**: WSL (Windows Subsystem for Linux) 或 Linux
- **Python**: 3.8 或更高版本
- **Conda**: Miniconda 或 Anaconda

### 创建 Conda 环境

在 WSL 中创建专用的 conda 环境并安装依赖：

```bash
# 1. 创建名为 rk-y8 的 Python 3.8 环境
conda create -n rk-y8 python=3.8 -y

# 2. 激活环境
conda activate rk-y8

# 3. 安装 PyTorch (CPU版本)
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 4. 安装 rknn-toolkit2 (根据你的版本)
# 方式一：从官方下载的whl文件安装
pip install /path/to/rknn_toolkit2-x.x.x-cpxx-cpxx-linux_x86_64.whl

# 方式二：如果已配置好，直接安装
pip install rknn-toolkit2

# 5. 安装 ultralytics (用于 YOLO 模型)
pip install ultralytics

# 6. 安装其他依赖
pip install opencv-python numpy
```

**📝 注意事项：**
- rknn-toolkit2 仅支持 x86_64 Linux 系统，必须在 WSL/Linux 环境下运行
- Python 版本建议使用 3.8，与 rknn-toolkit2 兼容性最好
- 如果遇到依赖冲突，建议使用 conda 创建隔离环境

## 🛠️ 安装步骤

### 1. 克隆仓库并激活环境

```bash
# 克隆仓库
git clone https://github.com/MWang-TS/pt2rknn.git
cd pt2rknn

# 激活 conda 环境 (确保已按上述步骤创建)
conda activate rk-y8
```

### 2. 安装 Web 工具依赖

```bash
pip install -r requirements.txt
```

### 3. 准备校准数据集（INT8 量化必需）

如果使用 INT8 量化，需要准备校准数据集：

**方法一：使用现有数据集**
```bash
# 复制你之前成功使用的校准文件
cp /mnt/e/rknn_model_zoo/examples/yolov8/firesmoke_calibration.txt ./calibration_data/calibration.txt
```

**方法二：创建新的校准数据集**
```bash
# 1. 将校准图片放到 calibration_data/images/ 目录
mkdir -p calibration_data/images
# 复制你的代表性图片到此目录

# 2. 编辑 calibration_data/calibration.txt，写入图片路径
# 详见 calibration_data/README.md
```

## 🚀 使用方法

### 启动 Web 服务

```bash
cd /mnt/e/rk3576dev/pt2rknn_tool
python app.py
```

或使用启动脚本：
```bash
chmod +x start.sh
./start.sh
```

服务启动后，终端会显示：
```
PT to RKNN 转换工具已启动
访问地址: http://localhost:5000
```

### 通过浏览器访问

1. 打开浏览器，访问 `http://localhost:5000`
2. 选择或拖拽你的 PT 模型文件（**支持普通PT和TorchScript格式**）
3. 选择目标平台（如 RK3576）
4. 选择量化类型：
   - **INT8** - 量化模型，体积小，速度快（推荐）
   - **FP16** - 不量化，精度高，体积较大
5. 设置输入尺寸（默认 640x640）
6. 点击"开始转换"
   - 🔄 如果是普通PT模型，工具会**自动转换为TorchScript**
   - ⚡ 然后自动转换为RKNN格式
7. 转换完成后点击下载按钮获取 RKNN 模型

**💡 转换流程：**
```
普通PT模型 (.pt/.pth)
    ↓ 自动转换
TorchScript (.torchscript)
    ↓ 自动转换  
RKNN模型 (.rknn)
```

**支持的模型格式：**
- ✅ 普通PT模型（YOLOv8，自定义模型等）- 自动转换
- ✅ TorchScript模型 - 直接转换
- ✅ 兼容ultralytics导出的模型

### 命令行使用（可选）

如果你更喜欢命令行，也可以直接使用转换脚本：

```bash
# 基本用法
python converter.py model.pt rk3576 i8

# 完整参数
python converter.py <pt_model> [platform] [quant_type] [output_path]
```

参数说明：
- `pt_model` - PT 模型文件路径
- `platform` - 目标平台：rk3562/rk3566/rk3568/rk3576/rk3588
- `quant_type` - 量化类型：i8 (INT8量化) 或 fp (FP16不量化)
- `output_path` - 输出文件路径（可选）

## 📁 目录结构

```
pt2rknn_tool/
├── app.py                    # Flask Web 服务
├── converter.py              # 核心转换脚本
├── config.py                 # 配置文件
├── requirements.txt          # Python 依赖
├── start.sh                  # 启动脚本
├── README.md                 # 本文档
├── templates/
│   └── index.html           # Web 界面
├── calibration_data/        # 校准数据集目录
│   ├── README.md           # 校准数据集说明
│   ├── calibration.txt     # 校准图片路径列表
│   └── images/             # 校准图片目录（可选）
├── uploads/                 # 上传文件临时目录（自动创建）
└── output/                  # 转换结果输出目录（自动创建）
```

## 🔍 使用示例

### 示例 1: 转换 YOLOv8 车牌检测模型（普通PT格式）

```bash
# 方法1: 使用Web界面（推荐）
# 1. 启动服务
python app.py

# 2. 浏览器访问 http://localhost:5000
#    - 上传: y8n-plate-20251111.pt （普通PT格式）
#    - 平台: RK3576
#    - 量化: INT8
#    - 尺寸: 640x640
#    - 点击转换
# 
# 工具会自动：
#   ① 检测到是普通PT格式
#   ② 自动转换为TorchScript
#   ③ 再转换为RKNN格式
#   ④ 提供下载

# 方法2: 命令行
python converter.py \
    /mnt/e/rk3576dev/models/license-plate.pt \
    rk3576 \
    i8
```

### 示例 2: 转换已有的TorchScript模型

```bash
# 如果你已经有TorchScript格式的模型
python converter.py \
    /mnt/e/rk3576dev/models/yolov8n_rknnopt.torchscript \
    rk3576 \
    i8
```

### 示例 3: 转换火灾检测模型（命令行）

```bash
python converter.py \
    /mnt/e/rk3576dev/models/best.pth \
    rk3576 \
    i8 \
    ./output/fire-detect-rk3576.rknn
```

## ⚠️ 注意事项

1. **校准数据集**
   - INT8 量化**必须**提供校准数据集
   - 校准图片应该是训练数据集的代表性样本
   - 建议 10-100 张图片
   - FP16 模式不需要校准数据集

2. **模型格式**
   - 仅支持 `.pt` 和 `.pth` 格式
   - 模型应该是 TorchScript 格式或标准 PyTorch checkpoint
   - 如果是自定义模型，确保已经正确导出

3. **内存和时间**
   - 转换过程可能需要几分钟，取决于模型大小
   - 确保 WSL 有足够的内存（建议 8GB+）
   - 上传文件大小限制：500MB

4. **conda 环境**
   - 必须在安装了 `rknn-toolkit2` 的环境中运行
   - 确保激活了正确的 conda 环境

## 🐛 故障排除

### 问题 1: 提示"量化模式需要校准数据集"

**解决方法**:
```bash
# 检查校准文件是否存在
ls calibration_data/calibration.txt

# 检查文件内容
cat calibration_data/calibration.txt

# 确保图片路径正确且文件存在
```

### 问题 2: "加载模型失败"

**可能原因**:
- 模型文件格式不正确
- 模型不是 TorchScript 格式

**解决方法**:
```python
# 如果你的模型需要先导出为 TorchScript
import torch
model = torch.load('your_model.pt')
model.eval()
traced = torch.jit.trace(model, torch.rand(1, 3, 640, 640))
traced.save('model_traced.pt')
```

### 问题 3: Web 服务无法访问

**解决方法**:
```bash
# 检查端口是否被占用
netstat -tuln | grep 5000

# 修改端口（编辑 app.py 最后一行）
app.run(host='0.0.0.0', port=5001, debug=True)
```

### 问题 4: 转换缓慢或卡住

**建议**:
- 检查终端输出查看详细日志
- 减少校准数据集图片数量
- 降低优化等级（编辑 converter.py 中的 optimization_level）

## 🔗 相关资源

- [RKNN Toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 模型库](https://github.com/ultralytics/ultralytics)
- [RKNN Model Zoo](https://github.com/rockchip-linux/rknn_model_zoo)

## 📝 更新日志

### v1.0.0 (2026-02-28)
- ✅ 初始版本发布
- ✅ Web 界面支持
- ✅ 命令行工具支持
- ✅ 多平台支持
- ✅ INT8/FP16 量化选项
- ✅ 转换历史记录

## 👨‍💻 开发者

基于你之前成功的转换脚本 `pt2rknn_firesmoke_final_fixed.py` 开发。

## 📄 许可证

MIT License

---

**享受简单快速的模型转换! 🎉**

如有问题或建议，欢迎反馈。
