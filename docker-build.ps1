# PT2RKNN Docker 构建脚本 (Windows PowerShell)
# 使用方式: .\docker-build.ps1

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "PT2RKNN Docker 构建脚本 (Windows)" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 检查 Docker 是否安装
Write-Host "`n[1/5] 检查 Docker 环境..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker 已安装: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] 错误: Docker 未安装或未启动" -ForegroundColor Red
    Write-Host "请先安装 Docker Desktop for Windows" -ForegroundColor Red
    exit 1
}

# 检查 rknn-toolkit2 whl 文件
Write-Host "`n[2/5] 检查 rknn-toolkit2 安装包..." -ForegroundColor Yellow
$rknnWhl = Get-ChildItem -Path . -Filter "rknn_toolkit2-*.whl" | Select-Object -First 1

if (-not $rknnWhl) {
    Write-Host "[ERROR] 错误: 未找到 rknn_toolkit2-*.whl 文件" -ForegroundColor Red
    Write-Host "请从 Rockchip 官方下载 rknn-toolkit2 并放置在当前目录" -ForegroundColor Yellow
    Write-Host "下载地址: https://github.com/rockchip-linux/rknn-toolkit2" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "或者修改 Dockerfile 中的安装方式" -ForegroundColor Yellow
    
    $continue = Read-Host "`n是否继续构建（不安装 rknn-toolkit2）? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        exit 1
    }
} else {
    Write-Host "[OK] 找到: $($rknnWhl.Name)" -ForegroundColor Green
}

# 创建必要的目录
Write-Host "`n[3/5] 创建挂载目录..." -ForegroundColor Yellow
$directories = @("uploads", "output", "calibration_data")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  创建: $dir" -ForegroundColor Gray
    } else {
        Write-Host "  已存在: $dir" -ForegroundColor Gray
    }
}
Write-Host "[OK] 目录创建完成" -ForegroundColor Green

# 构建 Docker 镜像
Write-Host "`n[4/5] 构建 Docker 镜像..." -ForegroundColor Yellow
Write-Host "这可能需要几分钟时间，请耐心等待..." -ForegroundColor Gray

$buildResult = docker build -t pt2rknn-web:latest .

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Docker 镜像构建成功" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Docker 镜像构建失败" -ForegroundColor Red
    exit 1
}

# 显示使用说明
Write-Host "`n[5/5] 构建完成！" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Green
Write-Host "使用以下命令启动容器:" -ForegroundColor Green
Write-Host ""

Write-Host "方式 1: 使用 docker-compose (推荐)" -ForegroundColor Cyan
Write-Host "  docker-compose up -d" -ForegroundColor White
Write-Host ""

Write-Host "方式 2: 使用 docker run" -ForegroundColor Cyan
Write-Host "  docker run -d -p 5600:5600 ``" -ForegroundColor White
Write-Host "    -v `${PWD}/uploads:/app/uploads ``" -ForegroundColor White
Write-Host "    -v `${PWD}/output:/app/output ``" -ForegroundColor White
Write-Host "    -v `${PWD}/calibration_data:/app/calibration_data ``" -ForegroundColor White
Write-Host "    --name pt2rknn-web pt2rknn-web:latest" -ForegroundColor White
Write-Host ""

Write-Host "访问地址: http://localhost:5600" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# 询问是否立即启动
Write-Host ""
$startNow = Read-Host "是否立即启动容器? (Y/n)"
if ($startNow -eq "" -or $startNow -eq "y" -or $startNow -eq "Y") {
    Write-Host "`n正在启动容器..." -ForegroundColor Yellow
    docker-compose up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 容器已启动" -ForegroundColor Green
        Start-Sleep -Seconds 2
        Write-Host "`n正在打开浏览器..." -ForegroundColor Yellow
        Start-Process "http://localhost:5600"
    } else {
        Write-Host "[ERROR] 容器启动失败，请查看错误信息" -ForegroundColor Red
    }
}
