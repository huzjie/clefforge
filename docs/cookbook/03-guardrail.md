# 菜谱 3：护栏判定

**场景**：对请求做 allow / block / review 三态判定。

```python
r = engine.decide("这条请求是否越权", ["allow", "block", "review"])
# 配合 Gate：低置信度 -> review，人工兜底
```

**要点**：置信度低时不要硬 allow，接门控转人工。
