#!/bin/bash

# 快速测试脚本 - 测试工具是否正常工作

echo "========================================"
echo "PT to RKNN 工具环境检查"
echo "========================================"

# 检查 conda 环境
echo -n "检查 conda 环境... "
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "✓ ($CONDA_DEFAULT_ENV)"
else
    echo "✗ 未激活"
    exit 1
fi

# 检查 Python
echo -n "检查 Python... "
python --version 2>&1 | head -n 1

# 检查 Flask
echo -n "检查 Flask... "
if python -c "import flask; print(f'✓ {flask.__version__}')" 2>/dev/null; then
    :
else
    echo "✗ 未安装"
    exit 1
fi

# 检查 RKNN Toolkit2
echo -n "检查 RKNN Toolkit2... "
if python -c "from rknn.api import RKNN; print('✓')" 2>/dev/null; then
    :
else
    echo "✗ 未安装"
    exit 1
fi

# 检查目录结构
echo -n "检查目录结构... "
required_dirs=("uploads" "output" "calibration_data" "templates")
all_exist=true
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "✗ 缺少 $dir"
        all_exist=false
    fi
done
if $all_exist; then
    echo "✓"
fi

# 检查必要文件
echo -n "检查必要文件... "
required_files=("app.py" "converter.py" "templates/index.html")
all_exist=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ 缺少 $file"
        all_exist=false
    fi
done
if $all_exist; then
    echo "✓"
fi

# 检查校准数据集
echo -n "检查校准数据集... "
if [ -f "calibration_data/calibration.txt" ]; then
    if grep -q "^[^#]" calibration_data/calibration.txt; then
        line_count=$(grep "^[^#]" calibration_data/calibration.txt | wc -l)
        echo "✓ ($line_count 张图片)"
    else
        echo "⚠️  已配置但为空"
    fi
else
    echo "✗ 未配置"
fi

echo ""
echo "========================================"
echo "环境检查完成！"
echo "========================================"
echo ""
echo "运行以下命令启动服务:"
echo "  ./start.sh"
echo ""
echo "或直接运行:"
echo "  python app.py"
echo ""
