# 菜谱 5：生产环境校准

**场景**：上线前校验置信度是否可信。

```python
from clefforge.decision.calibration import expected_calibration_error
ece = expected_calibration_error(confs, corrects)
# ECE 过高 -> 做温度缩放或排列评分
```

**要点**：ECE 是上线红线，置信度和准确率对不齐就拒答/降级。
