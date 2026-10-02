# 决策子系统

## 指针头

见 `clefforge/decision/head.py`。核心是「点积打分」，把全词表 softmax 换成 N 维 softmax。

## 打分

`scorer.score_options` 把 logits 变成 `{probs, choice, confidence}`。

## 校准

- `temperature_scale`：拟合温度对齐置信度。
- `expected_calibration_error`：ECE 度量。
- `brier`：Brier 分数。

## 排列评分

`permutation_score` 把候选乱序多轮打分取平均，消除顺序偏差。

## 门控与路由

`Gate` 三态（accept/degrade/reject）；`Router` 快慢选型。
