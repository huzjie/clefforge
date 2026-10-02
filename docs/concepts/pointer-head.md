# 指针头（Pointer Head）

决策模型的核心：把语言头换成对选项的打分头。

```python
logit_i = dot(q(state), k(option_i))
probs = softmax(logits)
```

输出空间从词表压缩到候选集，是延迟/成本优势的来源。
