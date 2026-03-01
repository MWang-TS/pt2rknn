# COCO 校准数据集

用于 YOLOv8-Det / YOLOv8-Seg / YOLOv8-Pose / YOLOv8-OBB 的 INT8 量化校准。

## 准备方法

将 20~100 张 COCO val2017 图片放入 `images/` 目录：

```bash
# 从 rknn_model_zoo 的数据集复制（如已下载）
cp /mnt/e/rknn_model_zoo/datasets/COCO/coco_subset_20/*.jpg images/
```

或手动放置包含多种目标的真实场景照片（jpg/png）。  
dataset.txt 会在首次转换时自动生成。
