# 菜谱 6：快慢路由

**场景**：简单任务走 flash，难任务走 full。

```python
from clefforge.decision.router import Router
r = Router(fast_slow=True)
print(r.route("短问题", ["a", "b"]).model)   # clef-flash
```

**要点**：难度特征要零成本可离线算。
