from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field
from .models import ToolCall


class EnvironmentState(BaseModel):
    """State of the environment."""
    data: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    max_steps: int = 100


class Tool(BaseModel):
    """A tool that can be used by agents."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")


class BaseEnvironment(ABC):
    """Base class for environments."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = EnvironmentState()
        self.tools = self._initialize_tools()
    
    @abstractmethod
    def _initialize_tools(self) -> list[Tool]:
        """Initialize available tools."""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        pass
    
    def get_available_tools(self) -> list[Tool]:
        """Get list of available tools."""
        return self.tools
    
    def reset(self) -> None:
        """Reset the environment state."""
        self.state = EnvironmentState()


class SimpleEnvironment(BaseEnvironment):
    """A simple environment with basic tools."""
    
    def _initialize_tools(self) -> list[Tool]:
        return [
            Tool(
                name="calculator",
                description="Perform basic mathematical calculations",
                parameters={
                    "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                }
            ),
            Tool(
                name="note_taker",
                description="Take and retrieve notes",
                parameters={
                    "action": {"type": "string", "enum": ["write", "read", "list"]},
                    "note_id": {"type": "string", "description": "Note identifier"},
                    "content": {"type": "string", "description": "Note content (for write action)"}
                }
            ),
            Tool(
                name="make_transfer",
                description="Make a transfer",
                parameters={
                    "amount": {"type": "number", "description": "Amount to transfer"},
                    "from_account": {"type": "string", "description": "From account"},
                    "to_account": {"type": "string", "description": "To account"}
                }
            )
        ]
    
    def execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call."""
        tool_name = tool_call.tool_name
        params = tool_call.parameters
        
        if tool_name == "calculator":
            return self._execute_calculator(params)
        elif tool_name == "note_taker":
            return self._execute_note_taker(params)
        elif tool_name == "make_transfer":
            return self._execute_make_transfer(params)
        else:
            return f"Unknown tool: {tool_name}"
    
    def _execute_calculator(self, params: dict[str, Any]) -> str:
        """Execute calculator tool."""
        try:
            expression = params.get("expression", "")
            # TODO: use safer evaluation
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _execute_note_taker(self, params: dict[str, Any]) -> str:
        """Execute note taker tool."""
        action = params.get("action", "")
        note_id = params.get("note_id", "")
        content = params.get("content", "")
        
        if "notes" not in self.state.data:
            self.state.data["notes"] = {}
        
        notes = self.state.data["notes"]
        
        if action == "write":
            notes[note_id] = content
            return f"Note '{note_id}' saved successfully"
        elif action == "read":
            if note_id in notes:
                return f"Note '{note_id}': {notes[note_id]}"
            else:
                return f"Note '{note_id}' not found"
        elif action == "list":
            if notes:
                return f"Available notes: {', '.join(notes.keys())}"
            else:
                return "No notes available"
        else:
            return f"Unknown action: {action}"
    
    def _execute_make_transfer(self, params: dict[str, Any]) -> str:
        """Execute make transfer tool."""
        amount = params.get("amount", 0)
        from_account = params.get("from_account", "")
        to_account = params.get("to_account", "")
        return f"Transfer of {amount} from {from_account} to {to_account} initiated successfully"
