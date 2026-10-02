# 训练

## RL

- `GRPO`：组相对优势。
- `PPO`：裁剪优势。

## SFT / DPO

- `sft_step`：交叉熵朝正确选项。
- `dpo_step`：偏好 chosen 优于 rejected。

## 关键铁律

训练信号必须「单调为正」，skill 才会单调趋 1。用「对 +1 错 -1」会在准确率低时把 skill 压到 0。
