# 菜谱 2：模型路由

**场景**：按任务把请求路由到不同模型。

```python
engine.decide("低成本快速回答", ["clef-flash", "clef", "gpt-6-sol"])
```

**要点**：把"延迟/成本/精度"写进 query 语义，指针头会学到权衡。
