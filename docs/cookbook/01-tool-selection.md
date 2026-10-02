# 菜谱 1：工具选择

**场景**：Agent 拿到任务，在 web_search / python_exec / file_read 里选一个。

```python
from clefforge.engine import ClefEngine
engine = ClefEngine()
r = engine.decide("查询最新股价", ["web_search", "python_exec", "file_read"])
print(r["answer"], r["confidence"], r["gate"])
```

**要点**：工具描述要进候选集，决策模型天然适合"选哪个工具"。
