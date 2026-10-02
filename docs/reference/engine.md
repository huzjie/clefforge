# ClefEngine API

```python
from clefforge.engine import ClefEngine
e = ClefEngine()
r = e.decide(query, options, image=None)
# r: {query, options, image, route, choice, answer, confidence, probs, gate, model_tier}
```
