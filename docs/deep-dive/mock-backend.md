# 确定性 mock 后端设计

## 目标

零依赖、可复现、可训练，让「训练/评测」在无 GPU 的环境也能演示。

## 设计

```python
logit_i = skill * truth_i + noise_i
truth_i = 1 if i == world_answer(query, options, image) else 0
```

- `world_answer`：md5 种子的确定性真值，与基准共享。
- `skill`：可训练标量，`train_step` 用正 delta 单调推到 1.0。
- `noise`：按 key 确定性采样，保证可复现。

## 铁律

1. `skill` 必须真正参与打分，否则训练无用。
2. 训练 delta 必须恒正，否则 skill 会被压到 0。
3. 确定性向量用 `md5(key)` 前 8 字节做种子，别平铺 16 字节 digest。
