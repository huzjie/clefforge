# 视觉子系统

- `Patcher`：图像 -> patch 特征（mock 后端用稳定哈希合成）。
- `resolve_resolution`：任意分辨率 -> 单网格或分片。
- `VisionEncoder`：ViT 风格，patch token + CLS token。
- `VisualProjector`：视觉 token 投影到语言隐空间，输出门控权重。

真实权重接入：覆写 `Patcher.patch_features` 换成真实 patch embedding 即可。
