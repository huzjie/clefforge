"""Synthetic decision-task generator (the training/eval corpus).

Each task is (query, options, image, correct_idx). Correctness comes from the
shared `world_answer`, so training and evaluation always agree on labels.
"""
from ..world import world_answer
from ..utils.stable import stable_choice


class SyntheticDecisionData:
    def __init__(self, seed=0, n_tasks=200):
        self.seed = seed
        self.n_tasks = n_tasks

    def generate(self):
        tasks = []
        for i in range(self.n_tasks):
            domain = i % 4
            query, options = self._make_task(i, domain)
            image = f"img-{i % 37}" if domain == 3 else None
            correct = world_answer(query, options, image=image)
            tasks.append({"id": i, "query": query, "options": options,
                          "image": image, "correct": correct})
        return tasks

    def _make_task(self, i, domain):
        if domain == 0:  # tool selection
            tools = ["web_search", "python_exec", "file_read", "http_get", "sql_query"]
            q = f"给定任务 #{i}，选择最合适的工具。"
            return q, tools[:3]
        if domain == 1:  # model routing
            models = ["clef", "clef-flash", "gpt-6-sol", "claude-4"]
            q = f"任务 #{i} 需要快速低成本决策，选择模型。"
            return q, models[:4]
        if domain == 2:  # guardrail
            actions = ["allow", "block", "review"]
            q = f"对请求 #{i} 执行护栏判定。"
            return q, actions
        # domain == 3: visual discrimination
        labels = ["cat", "dog", "car", "bicycle"]
        q = f"识别图像 img-{i % 37} 中的物体类别。"
        return q, labels
