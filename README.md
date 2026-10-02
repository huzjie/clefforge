# clefforge — 多模态决策模型训练与推理框架

> 把 Cloudflare 开源的多模态决策模型 **Clef** 做成一整套可直接运行的框架：**给候选选项打分、返回置信度，而不是生成自由文本**。填好配置即跑，零外部依赖。

Clef 的核心理念一句话：把语言模型的「语言头」换成「指针头」，让模型从「生成下一个 token」变成「在 N 个候选里打分」。相比 Jev（纯文本），Clef 多了**原生视觉编码器**和 **64k 上下文**。本仓库是这一范式的一等公民实现：指针决策头、视觉编码器、多模态融合、置信度校准、门控、快慢路由、RL 定制微调、Jev-API 兼容服务，全部可运行、可训练、可评测。

## 为什么是「决策模型」而不是「大模型」

这是最有价值的工程取舍，值得先说清：

- 一个智能体工作流里，大量环节只需要「基于当前状态，下一步选哪个」——选工具、选模型、过护栏。为这种决策跑一个 27B 生成式模型，是又慢又贵。
- 决策模型把输出空间从「整个词表」压缩到「N 个候选」，**延迟低 1.6–9 倍、成本低一个数量级**，同时置信度可直接用于门控和降级。
- 你拿到的不是"它会说什么"，而是"它选了第几个、有多确定"，这正好是编排层要的信号。

## 快速开始

```bash
# 零依赖，Python 3.9+
pip install -e .

# 体检：组件 / 后端 / 基准是否就绪
python -m clefforge doctor

# 单次决策（文本）
python -m clefforge decide "选择最合适的工具" --options web_search python_exec file_read

# 单次决策（多模态：带图像）
python -m clefforge decide "识别图像中的物体" --options cat dog car --image img-7

# RL 定制微调（准确率从近随机爬到 1.0）
python -m clefforge train

# 四组基准跑分
python -m clefforge bench

# 起 Jev-API 兼容服务
python -m clefforge serve --host 127.0.0.1 --port 8000
```

## 技术原理拆解

### 1. 指针头（Pointer Head）—— 决策模型的灵魂

普通 LM 最后一层是全词表 softmax（15 万维）。决策模型只保留一个「指针」：把融合后的隐状态和每个候选选项的表示做点积打分，得到 N 维 logits，再 softmax 成概率。

```python
# clefforge/decision/head.py 的核心
class PointerHead:
    def logits(self, state, option_vecs):
        qs = self.q(state)               # query -> dim
        return [dot(qs, self.k(ov)) for ov in option_vecs]   # 每个候选一个分数
```

**可复用要点**：任何「从固定选项里选一个」的任务——工具选择、模型路由、护栏、意图分类——都可以用同一个指针头替换生成式头，换来确定性输出 + 概率 + 置信度。

### 2. 原生视觉编码器 + 多模态融合

Clef 的原生视觉编码器把图像 patch 化后过 ViT，再经投影器映射到语言 token 空间。融合用**门控融合**而非拼接：学一个标量门 `g ∈ [0,1]`，`h = g·text + (1-g)·visual`，隐维度不变，模型自己学「这份输入该信图像多少」。

```python
# clefforge/fusion/multimodal.py
g = sigmoid(gate_logit)
fused = [g * t + (1 - g) * v for t, v in zip(text_vec, visual_vec)]
```

**可复用要点**：多模态别直接 `concat`（维度翻倍、计算爆），用门控或交叉注意力把模态"对齐"到同一个隐空间。

### 3. 置信度校准 + 排列评分 —— 让概率可信

决策模型的 `max(softmax)` 往往过度自信。Clef 用两条手段：

- **排列评分**：把候选打乱多次分别打分再平均，消除「LLM 偏爱靠前的选项」的顺序偏差。
- **温度缩放**：拟合一个温度 `T`，`p_i ∝ p_i^{1/T}`，把置信度对齐到真实准确率（用 ECE / Brier 度量）。

```python
# 排列评分：每个位置都轮一遍，取平均
scores[i] = mean(raw[pos] for each shuffled position)
# 温度缩放
p = [v ** (1/T) for v in probs]; p = normalize(p)
```

**可复用要点**：只要你给下游输出概率，就一定要过 ECE 校验；顺序偏差是 LLM 决策的头号隐形 bug，排列评分 10 行代码就能干掉它。

### 4. 门控 + 快慢路由 —— 把置信度变成编排动作

置信度低怎么办？门控给三个动作：`accept`（置信度高）/ `degrade`（中，转慢模型）/ `reject`（低，拒答）。快慢路由则按任务难度把简单任务甩给 9B 的 `clef-flash`，难任务（带图、长上下文）走 27B 的 `clef`。

```python
# 门控
action = "accept" if conf >= 0.7 else ("degrade" if conf >= 0.5 else "reject")
# 路由难度估计：查询长度 + 选项数 + 是否带图 + 措辞是否含糊
```

**可复用要点**：置信度不要只拿来看，要接进门控和路由闭环——这是决策模型相对生成式模型最值钱的地方。

### 5. RL 定制微调 —— 用业务信号训练指针头

Cloudflare 同时发布了 RL 产品让客户针对业务微调。本框架实现了 GRPO / PPO / DPO，奖励信号就是业务结果（选对没选对、是否被接受）。演示里 `skill` 从 0.35 单调爬到 1.0，准确率从近随机到 1.0。

```python
# 训练信号必须「单调为正」，skill 才会单调趋 1
delta = 0.04 + 0.06 * reward   # reward ∈ {0,1}，delta 恒正
backend.train_step(delta)
```

## 项目结构

```
clefforge/
├── core/        # 零依赖张量库（Tensor/ops/nn）
├── vision/      # ViT 编码器 / patch 化 / 任意分辨率 / 投影器
├── fusion/      # 门控融合 / 交叉注意力 / MoE 视觉 token 路由
├── model/       # Clef(27B) / Clef-flash(9B) / Qwen 骨干
├── decision/    # 指针头 / 打分 / 校准 / 排列 / 门控 / 快慢路由
├── backends/    # mock(可训练)/cpu/openai/vllm/transformers
├── train/       # GRPO/PPO/SFT/DPO + 微调 runner
├── data/        # 合成决策数据
├── bench/       # Jev Decision Index / 视觉判别 / 路由 / 校准
├── serving/     # Jev-API 兼容 stdlib HTTP 服务
└── cli/         # doctor/decide/train/bench/serve
```

## 后端

| 后端 | 说明 |
|---|---|
| `mock` | 确定性 + 可训练，零依赖默认后端 |
| `cpu` | 真指针头 + 视觉编码器在张量核上跑 |
| `openai` | Jev-API / OpenAI 兼容端点，未配置则回退 mock |
| `vllm` / `transformers` | 高吞吐 / 本地权重适配器 |

## 基准

| 基准 | 测什么 |
|---|---|
| `decision_index` | 多域选项决策准确率（对标 Jev Decision Index） |
| `vision_discrimination` | 图像是否真正改变决策（视觉参与度） |
| `routing` | 快慢路由选型正确率 |
| `calibration` | ECE / Brier 置信度可信度 |

## 部署

Docker / docker-compose / Kubernetes / Helm 均已就绪，见 `deploy/`。CI 见 `.github/workflows/ci.yml`。

## 许可

MIT。
