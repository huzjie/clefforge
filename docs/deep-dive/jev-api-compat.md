# Jev-API 兼容

`POST /decide` 请求 `{model, query, options, image}`，返回 `{choice, confidence, probs, gate, route}`，形状对齐 Jev 决策 API，老客户端可无缝迁移。
