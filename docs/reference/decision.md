# decision 模块 API

- `PointerHead`：点积打分头。
- `score_options`：logits -> probs/choice/confidence。
- `temperature_scale` / `expected_calibration_error` / `brier`：校准。
- `permutation_score`：去顺序偏差。
- `Gate`：accept/degrade/reject。
- `Router`：fast/slow 选型。
