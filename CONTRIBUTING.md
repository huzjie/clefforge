# 贡献指南

1. Fork 本仓库。
2. `pip install -e .` 后跑 `python -m clefforge doctor` 确认基线。
3. 新增后端：在 `clefforge/backends/` 加一个类，用 `@register_backend("name")` 注册。
4. 新增基准：在 `clefforge/bench/` 加函数，用 `@register_benchmark("name")` 注册。
5. 提交前跑 `python -m unittest discover -s tests -t .`。

保持零依赖内核（`clefforge/core`）不引入第三方库。
