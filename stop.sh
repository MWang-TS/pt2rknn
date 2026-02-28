#!/bin/bash

# 停止 PT to RKNN 转换工具服务

echo "正在停止 PT to RKNN 转换工具..."

# 查找并停止占用5000端口的进程
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "找到运行中的服务，正在停止..."
    lsof -ti:5000 | xargs kill -9 2>/dev/null
    sleep 1
    
    # 验证
    if ! lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo "✓ 服务已停止"
    else
        echo "✗ 停止失败，请手动检查"
    fi
else
    echo "没有运行中的服务"
fi
