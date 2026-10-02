"""MCP (Model Context Protocol) adapter.

Exposes Clef as an MCP tool (`clef.decide`) so MCP clients (Claude Code, Cursor,
etc.) can call it. Implemented as a plain tool-descriptor object; the actual MCP
transport is left to the host, so this works with any MCP SDK.
"""


class ClefMCPAdapter:
    def __init__(self, engine):
        self.engine = engine

    @property
    def tool_spec(self):
        return {
            "name": "clef.decide",
            "description": "在多模态决策候选选项中打分并返回置信度（指针头决策模型）。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "image": {"type": "string"},
                },
                "required": ["query", "options"],
            },
        }

    def call(self, arguments):
        query = arguments.get("query", "")
        options = arguments.get("options", [])
        image = arguments.get("image")
        return self.engine.decide(query, options, image=image)
