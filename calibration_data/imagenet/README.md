# ImageNet 校准数据集

用于 ResNet 的 INT8 量化校准。

## 准备方法

将 20~50 张 ImageNet val 图片（各类别均匀分布）放入 `images/` 目录：

```bash
# 从 rknn_model_zoo 的数据集复制（如已下载）
cp /mnt/e/rknn_model_zoo/datasets/imagenet/ILSVRC2012_img_val_samples/images/*.JPEG images/
```

dataset.txt 会在首次转换时自动生成。
