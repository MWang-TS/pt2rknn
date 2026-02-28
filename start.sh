#!/bin/bash

# PT to RKNN 转换工具启动脚本

echo "====================================="
echo "PT to RKNN 模型转换工具"
echo "====================================="

# 检查是否在 conda 环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "⚠️  警告: 未检测到 conda 环境"
    echo "请先激活 rk-y8 环境:"
    echo "  conda activate rk-y8"
    exit 1
fi

echo "✓ Conda 环境: $CONDA_DEFAULT_ENV"

# 检查是否安装了 Flask
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask 未安装"
    echo "正在安装依赖..."
    pip install -r requirements.txt
fi

# 检查 RKNN Toolkit2
if ! python -c "import rknn.api" 2>/dev/null; then
    echo "❌ 错误: rknn-toolkit2 未安装"
    echo "请确保在正确的 conda 环境中，并已安装 rknn-toolkit2"
    exit 1
fi

echo "✓ RKNN Toolkit2 已安装"

# 创建必要的目录
mkdir -p uploads output calibration_data/images templates

# 检查校准数据集（如果使用量化）
if [ -f "calibration_data/calibration.txt" ]; then
    # 检查文件是否为空或只有注释
    if grep -q "^[^#]" calibration_data/calibration.txt 2>/dev/null; then
        echo "✓ 校准数据集已配置"
    else
        echo "⚠️  提示: 如需使用 INT8 量化，请配置校准数据集"
        echo "   编辑: calibration_data/calibration.txt"
        echo "   说明: calibration_data/README.md"
    fi
else
    echo "⚠️  提示: calibration.txt 不存在，已创建示例文件"
fi

echo ""
echo "====================================="
echo "启动 Web 服务..."
echo "====================================="

# 检查端口5000是否被占用
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  端口5000已被占用"
    echo -n "是否停止占用该端口的程序? (y/n) "
    read -n 1 answer
    echo ""
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        echo "正在停止占用端口5000的程序..."
        lsof -ti:5000 | xargs kill -9 2>/dev/null
        sleep 1
        echo "✓ 端口已释放"
    else
        echo "❌ 启动取消"
        exit 1
    fi
fi

echo ""

# 启动 Flask 应用
python app.py
