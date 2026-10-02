# 微调你的决策模型

1. 用 `clefforge/data/synth.py` 换成你的业务决策数据。
2. 在 `world_answer` 里定义「正确选项」。
3. `python -m clefforge train` 跑 RL 微调。
4. 看准确率从近随机爬到 1.0。
