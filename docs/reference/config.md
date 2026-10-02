# 配置参考

`configs/default.yaml`：

```yaml
backend: {name: mock, model: clef-flash}
vision: {patch_size: 16, hidden: 128, layers: 2}
decision: {temperature: 1.0, calibrate: true, gate_threshold: 0.5, permutation: true, fast_slow: true}
train: {steps: 40, lr: 0.05, algorithm: grpo}
```

`clefforge/config.py` 无 PyYAML 时回退内置极简解析器。
