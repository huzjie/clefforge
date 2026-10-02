# API

## CLI

- `doctor` 体检
- `decide` 单次决策
- `train` 微调
- `bench` 基准
- `serve` 起服务

## HTTP

- `GET /health` -> `{"status": "ok", "model": ...}`
- `POST /decide` -> 决策结果

## Python SDK

```python
from clefforge.engine import ClefEngine
engine = ClefEngine()
result = engine.decide(query, options, image=image)
```
