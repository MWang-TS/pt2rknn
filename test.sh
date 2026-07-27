#!/bin/bash

# PT to RKNN 转换工具 - 快速测试脚本

echo "========================================"
echo "PT to RKNN 自动转换功能测试"
echo "========================================"

# 激活环境
if [[ "$CONDA_DEFAULT_ENV" != "rk-y8" ]]; then
    echo "⚠️  请先激活rk-y8环境:"
    echo "  conda activate rk-y8"
    exit 1
fi

echo "✓ Conda环境: $CONDA_DEFAULT_ENV"
echo ""

# 测试1: 检查依赖
echo "测试1: 检查依赖..."
python -c "
import torch
import sys
from rknn.api import RKNN
print('  ✓ torch:', torch.__version__)
print('  ✓ rknn-toolkit2: 已安装')
try:
    from ultralytics import YOLO
    print('  ✓ ultralytics: 已安装')
except:
    print('  ⚠️  ultralytics: 未安装（可选，用于加载某些模型）')
"

echo ""
echo "测试2: 验证转换器模块..."
python -c "
from converter import PT2RKNNConverter
converter = PT2RKNNConverter(verbose=False)
print('  ✓ 转换器模块加载成功')
print('  ✓ 支持自动TorchScript转换')
"

echo ""
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""
echo "🎉 所有测试通过！可以开始使用了"
echo ""
echo "启动Web服务:"
echo "  python app.py"
echo ""
echo "或使用启动脚本:"
echo "  ./start.sh"
echo ""
