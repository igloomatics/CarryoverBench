from __future__ import annotations
"""Tool registry for dynamic tool management."""

from pathlib import Path
from typing import Any

from nanobot.agent import provenance_gate
from nanobot.agent.provenance_context import get_current_user_message
from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    Registry for agent tools.
    
    Allows dynamic registration and execution of tools.
    """
    
    def __init__(self, workspace: Path | None = None):
        self._tools: dict[str, Tool] = {}
        self.workspace = workspace
        self.last_execution_metadata: dict[str, Any] | None = None
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        self._tools.pop(name, None)
    
    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in OpenAI format."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """
        Execute a tool by name with given parameters.
        
        Args:
            name: Tool name.
            params: Tool parameters.
        
        Returns:
            Tool execution result as string.
        
        Raises:
            KeyError: If tool not found.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"

        try:
            self.last_execution_metadata = None
            errors = tool.validate_params(params)
            if errors:
                return f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)
            if provenance_gate.is_enabled():
                decision = provenance_gate.check_tool_call(
                    name,
                    params,
                    workspace=self.workspace,
                    current_user_message=get_current_user_message(),
                )
                self.last_execution_metadata = {
                    "provenance_gate": decision.audit,
                    "blocked": not decision.allowed,
                }
                if not decision.allowed:
                    provenance_gate.mark_blocked_params(params, decision.audit)
                    return (
                        "Blocked by provenance gate: side-effecting tool call "
                        f"'{name}' was not executed because {decision.reason}."
                    )
            return await tool.execute(**params)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    @property
    def tool_names(self) -> list[str]:
        """Get list of registered tool names."""
        return list(self._tools.keys())
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
