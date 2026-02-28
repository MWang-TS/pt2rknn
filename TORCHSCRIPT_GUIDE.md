# PT模型转TorchScript快速指南

## 问题原因

RKNN需要**TorchScript格式**的PyTorch模型，而不是普通的`.pt`检查点文件。

常见错误：
```
RuntimeError: PytorchStreamReader failed locating file constants.pkl: file not found
```

这表示模型不是TorchScript格式。

## 解决方法

### 方法1：使用提供的转换脚本（最简单）

```bash
cd /mnt/e/rk3576dev/pt2rknn_tool
conda activate rk-y8
python pt2torchscript.py /path/to/your_model.pt
```

**示例：**
```bash
# 转换车牌检测模型
python pt2torchscript.py ../models/y8n_20250717-license-plate.pt

# 指定输出文件名
python pt2torchscript.py model.pt -o model_traced.pt

# 使用PyTorch方法（非YOLO模型）
python pt2torchscript.py model.pt --pytorch
```

### 方法2：使用Ultralytics（适用于YOLOv8）

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('your_model.pt')

# 导出为TorchScript
model.export(format='torchscript')

# 输出文件：your_model.torchscript
```

或命令行：
```bash
yolo export model=your_model.pt format=torchscript
```

### 方法3：使用exporter.py（已有的工具）

```bash
cd /mnt/e/rk3576dev/lib/ultralytics_yolov8
python -c "from ultralytics import YOLO; YOLO('model.pt').export(format='torchscript')"
```

### 方法4：手动使用PyTorch

```python
import torch

# 加载模型
model = torch.load('model.pt', map_location='cpu')

# 如果是字典格式，提取模型
if isinstance(model, dict):
    model = model['model']

model.eval()

# 创建示例输入（根据你的模型调整尺寸）
example_input = torch.randn(1, 3, 640, 640)

# 转换为TorchScript
traced_model = torch.jit.trace(model, example_input)

# 保存
traced_model.save('model_torchscript.pt')
```

## 验证转换结果

转换后，可以验证文件是否为TorchScript格式：

```python
import torch

# 尝试加载
try:
    model = torch.jit.load('model_torchscript.pt')
    print("✓ 模型是TorchScript格式")
except:
    print("✗ 模型不是TorchScript格式")
```

## 常见问题

### Q: 转换后文件大小变化了？
A: 正常现象。TorchScript格式可能比原始checkpoint大一些，因为包含了完整的模型结构。

### Q: 提示"无法识别的checkpoint格式"？
A: 说明你的模型是state_dict格式，需要模型架构定义。建议使用ultralytics的YOLO模型。

### Q: 转换后精度会改变吗？
A: 不会。TorchScript只是改变了模型的存储格式，不影响权重和精度。

### Q: 哪些模型可以转换？
A: 
- ✓ YOLOv5/v8/v11等Ultralytics模型
- ✓ 完整的PyTorch模型对象
- ✗ 只有state_dict的checkpoint（需要模型定义）

## 完整转换流程示例

```bash
# 1. 激活环境
conda activate rk-y8

# 2. 转换为TorchScript
cd /mnt/e/rk3576dev/pt2rknn_tool
python pt2torchscript.py ../models/my_model.pt

# 3. 在Web界面上传转换后的文件
# 访问 http://localhost:5000
# 上传 my_model_torchscript.pt

# 4. 选择参数并转换为RKNN
```

## 需要帮助？

如果遇到问题：

1. 检查模型是否是Ultralytics框架训练的
2. 确认PyTorch版本兼容性（推荐1.12+）
3. 查看转换脚本的详细错误信息
4. 参考 `/mnt/e/rknn_model_zoo/examples/yolov8/` 中的示例
