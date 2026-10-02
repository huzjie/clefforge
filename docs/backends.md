# 后端

| 后端 | 适用 |
|---|---|
| mock | 确定性、可训练、零依赖（默认） |
| cpu | 真指针头 + 视觉编码器在张量核跑 |
| openai | Jev-API / OpenAI 兼容端点 |
| vllm | 高吞吐推理 |
| transformers | 本地 HF 权重 |

未配置真实端点的后端会透明回退 mock，保证 CLI 开箱即用。
