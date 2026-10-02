# 多模态融合

三种融合策略，按需选用：

1. **门控融合** `GatedFusion`：`h = g·text + (1-g)·visual`，维度不变。
2. **交叉注意力** `CrossAttentionFusion`：文本 token 查询视觉 token。
3. **MoE 路由** `MoETokenRouter`：每个视觉 patch 只激活 top-k 专家。

推荐默认用门控融合：实现简单、开销小、可解释。
