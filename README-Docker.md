# PT2RKNN Docker 部署指南 🐳

本文档说明如何使用 Docker 容器化部署 PT2RKNN 转换工具。

## 📋 前置要求

- **Docker**: 20.10 或更高版本
- **Docker Compose**: 1.29 或更高版本（可选）
- **rknn-toolkit2**: 需要预先下载 whl 安装包

## 🔧 准备工作

### 1. 下载 rknn-toolkit2

从 Rockchip 官方仓库下载对应的 whl 文件：

```bash
# 访问官方仓库
https://github.com/rockchip-linux/rknn-toolkit2

# 下载适合 Python 3.8 的版本，例如：
# rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl
```

将下载的 whl 文件放置在 `pt2rknn_tool/` 目录下。

### 2. 修改 Dockerfile

打开 `Dockerfile`，取消注释 rknn-toolkit2 安装部分：

```dockerfile
# 选项 1: 从本地复制 whl 文件（推荐）
COPY rknn_toolkit2-*.whl /tmp/
RUN pip install --no-cache-dir /tmp/rknn_toolkit2-*.whl && rm /tmp/rknn_toolkit2-*.whl
```

## 🚀 快速开始

### 方式 1: 使用构建脚本（推荐）

```bash
# 给脚本执行权限
chmod +x docker-build.sh

# 运行构建脚本
./docker-build.sh

# 启动容器
docker-compose up -d
```

### 方式 2: 手动构建

```bash
# 1. 构建镜像
docker build -t pt2rknn-web:latest .

# 2. 创建挂载目录
mkdir -p uploads output calibration_data

# 3. 启动容器
docker-compose up -d

# 或使用 docker run
docker run -d \
  --name pt2rknn-web \
  -p 5600:5600 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/calibration_data:/app/calibration_data \
  pt2rknn-web:latest
```

## 📊 访问服务

浏览器打开: **http://localhost:5600**

## 🛠️ 常用命令

```bash
# 查看容器状态
docker-compose ps

# 查看容器日志
docker-compose logs -f pt2rknn

# 停止容器
docker-compose down

# 重启容器
docker-compose restart

# 进入容器调试
docker exec -it pt2rknn-web bash

# 清理容器和镜像
docker-compose down --rmi all --volumes
```

## 📂 目录映射

| 容器内路径 | 宿主机路径 | 说明 |
|-----------|-----------|------|
| `/app/uploads` | `./uploads` | 上传的模型文件 |
| `/app/output` | `./output` | 转换后的 RKNN 模型 |
| `/app/calibration_data` | `./calibration_data` | INT8 校准数据集 |
| `/c`, `/d`, `/e` | C:, D:, E: (Windows)| 只读访问所有驱动器 |
| `/host` | `/` (Linux)| 需手动启用，见下文 |

### Windows 用户 - 自动映射

所有驱动器自动可用：
- `C:\` → `/c`
- `D:\` → `/d`
- `E:\` → `/e`

**路径转换示例**：
```
Windows 路径: E:\datasets_models\my_data\images
容器内路径:   /e/datasets_models/my_data/images
```

### Linux 用户 - 手动启用

编辑 `docker-compose.yml`，取消注释：
```yaml
volumes:
  - /:/host:ro
```

然后重启容器：
```bash
docker-compose down && docker-compose up -d
```

## 🔧 自定义配置

### 修改端口

编辑 `docker-compose.yml`:

```yaml
ports:
  - "8080:5600"  # 改为 8080 端口
```

### 挂载额外的模型目录

编辑 `docker-compose.yml`:

```yaml
volumes:
  - /path/to/your/models:/app/models:ro
```

### 环境变量

在 `docker-compose.yml` 中添加:

```yaml
environment:
  - FLASK_ENV=production
  - MAX_UPLOAD_SIZE=1000  # MB
```

## ⚠️ 注意事项

1. **rknn-toolkit2 必须安装**: 容器无法运行转换功能而不安装 rknn-toolkit2
2. **仅支持 x86_64**: RKNN Toolkit 仅支持 x86_64 架构，ARM 设备无法使用
3. **内存要求**: 建议至少 4GB 可用内存用于模型转换
4. **WSL 用户**: 需要在 WSL 环境中运行 Docker 命令

## 🐛 故障排查

### 问题 1: rknn-toolkit2 安装失败

**解决方案**: 
- 确保 whl 文件在正确的目录
- 检查 whl 文件版本是否匹配 Python 3.8
- 查看构建日志: `docker build --no-cache -t pt2rknn-web:latest .`

### 问题 2: 无法访问 Web 界面

**解决方案**:
```bash
# 检查容器是否运行
docker ps | grep pt2rknn

# 检查端口映射
docker port pt2rknn-web

# 查看容器日志
docker logs pt2rknn-web
```

### 问题 3: 转换失败

**解决方案**:
- 进入容器检查环境: `docker exec -it pt2rknn-web bash`
- 手动测试: `python -c "import rknn; print(rknn.__version__)"`
- 检查挂载目录权限: `ls -la uploads/ output/`

## 📚 更多信息

- [主文档](README.md)
- [RKNN Toolkit 官方文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [Docker 官方文档](https://docs.docker.com/)

## 📝 更新日志

- **v1.0.0** (2026-03-17): 初始 Docker 配置
  - 支持 Python 3.8 + PyTorch 1.12.1
  - Flask Web UI
  - 多目录挂载
  - 健康检查
