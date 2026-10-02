# 菜谱 4：视觉分诊

**场景**：对图像做初筛分类，再决定是否需要人工。

```python
r = engine.decide("识别图像中的物体", ["cat", "dog", "car", "bicycle"], image="img-7")
```

**要点**：这是 Clef 相对纯文本 Jev 的差异化能力——视觉编码器直接参与决策。
