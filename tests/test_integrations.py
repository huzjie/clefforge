import unittest

from clefforge.config import Config
from clefforge.engine import ClefEngine
from clefforge.integrations.mcp import ClefMCPAdapter
from clefforge.integrations.agent import DecisionAgent


class TestIntegrations(unittest.TestCase):
    def test_mcp_adapter(self):
        adapter = ClefMCPAdapter(ClefEngine(Config()))
        spec = adapter.tool_spec
        self.assertEqual(spec["name"], "clef.decide")
        r = adapter.call({"query": "选工具", "options": ["a", "b"]})
        self.assertIn("choice", r)

    def test_agent_loop(self):
        agent = DecisionAgent(Config(), max_steps=3)
        trace = agent.run("完成一个任务", ["action-a", "action-b"],
                          lambda goal, state, action: ({}, [], True))
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
