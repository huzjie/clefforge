"""LangChain Tool wrapper (works with or without langchain installed).

Exposes Clef's decide() as a callable tool so a LangChain agent can call it for
tool selection / routing / guardrails. Degrades gracefully if langchain is absent.
"""


class ClefLangChainTool:
    def __init__(self, engine):
        self.engine = engine

    def _run(self, query, options=None, image=None):
        options = options or []
        r = self.engine.decide(query, options, image=image)
        return {"answer": r["answer"], "confidence": r["confidence"], "gate": r["gate"]}

    def to_langchain(self):
        try:
            from langchain.tools import tool
        except Exception:
            return None

        engine = self.engine

        @tool
        def clef_decide(query: str, options: list, image: str = None) -> dict:
            """用多模态决策模型在候选里选一个并给出置信度。"""
            return engine.decide(query, options, image=image)

        return clef_decide
