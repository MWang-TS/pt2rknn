#!/bin/bash
# PT2RKNN Docker 构建脚本

set -e

echo "================================================"
echo "PT2RKNN Docker 构建脚本"
echo "================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 rknn-toolkit2 whl 文件
echo -e "\n${YELLOW}[1/4] 检查 rknn-toolkit2 安装包...${NC}"
RKNN_WHL=$(ls rknn_toolkit2-*.whl 2>/dev/null | head -n 1)

if [ -z "$RKNN_WHL" ]; then
    echo -e "${RED}错误: 未找到 rknn_toolkit2-*.whl 文件${NC}"
    echo "请从 Rockchip 官方下载 rknn-toolkit2 并放置在当前目录"
    echo "下载地址: https://github.com/rockchip-linux/rknn-toolkit2"
    echo ""
    echo "或者修改 Dockerfile 中的安装方式"
    exit 1
else
    echo -e "${GREEN}✓ 找到: $RKNN_WHL${NC}"
fi

# 构建 Docker 镜像
echo -e "\n${YELLOW}[2/4] 构建 Docker 镜像...${NC}"
docker build -t pt2rknn-web:latest .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker 镜像构建成功${NC}"
else
    echo -e "${RED}✗ Docker 镜像构建失败${NC}"
    exit 1
fi

# 创建必要的目录
echo -e "\n${YELLOW}[3/4] 创建挂载目录...${NC}"
mkdir -p uploads output calibration_data
echo -e "${GREEN}✓ 目录创建完成${NC}"

# 显示使用说明
echo -e "\n${YELLOW}[4/4] 构建完成！${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}使用以下命令启动容器:${NC}"
echo ""
echo -e "  ${YELLOW}# 方式 1: 使用 docker-compose (推荐)${NC}"
echo -e "  docker-compose up -d"
echo ""
echo -e "  ${YELLOW}# 方式 2: 使用 docker run${NC}"
echo -e "  docker run -d -p 5600:5600 \\"
echo -e "    -v \$(pwd)/uploads:/app/uploads \\"
echo -e "    -v \$(pwd)/output:/app/output \\"
echo -e "    -v \$(pwd)/calibration_data:/app/calibration_data \\"
echo -e "    --name pt2rknn-web pt2rknn-web:latest"
echo ""
echo -e "${GREEN}访问地址: http://localhost:5600${NC}"
echo -e "${GREEN}================================================${NC}"
