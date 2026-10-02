# Changelog

## [1.0.0] - 2026-10-02

- 首发：多模态决策模型训练与推理框架（Cloudflare Clef 方向）。
- 指针决策头 + 视觉编码器 + 门控/交叉注意力/MoE 融合。
- 置信度校准（温度缩放 + 排列评分）+ 门控 + 快慢路由。
- 五后端（mock/cpu/openai/vllm/transformers）+ RL/SFT/DPO 微调。
- Jev-API 兼容 stdlib HTTP 服务 + CLI + Docker/K8s/Helm/CI。
