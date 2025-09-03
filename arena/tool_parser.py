"""Tool call parsing utilities for Arena."""

import re
import json
from typing import Any
from .models import ToolCall


class ToolCallParser:
    """Parses tool calls from LLM responses."""
    
    @staticmethod
    def parse_tool_calls(response_content: str) -> list[ToolCall]:
        """Parse tool calls from response content.
        
        Supports multiple formats:
        1. JSON format: {"tool": "calculator", "params": {"expression": "2+2"}}
        2. Function call format: calculator(expression="2+2")
        3. Structured format: [TOOL:calculator] expression="2+2"
        """
        tool_calls = []
        
        json_calls = ToolCallParser._parse_json_format(response_content)
        tool_calls.extend(json_calls)
        
        func_calls = ToolCallParser._parse_function_format(response_content)
        tool_calls.extend(func_calls)
        
        struct_calls = ToolCallParser._parse_structured_format(response_content)
        tool_calls.extend(struct_calls)
        
        return tool_calls
    
    @staticmethod
    def _parse_json_format(content: str) -> list[ToolCall]:
        """Parse JSON format tool calls."""
        tool_calls = []
        
        # look for JSON objects with tool and params
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*\}'
        matches = re.finditer(json_pattern, content, re.IGNORECASE)
        
        for match in matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                tool_name = data.get("tool", "")
                params = data.get("params", data.get("parameters", {}))
                
                if tool_name:
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        parameters=params
                    ))
            except (json.JSONDecodeError, KeyError):
                continue
        
        return tool_calls
    
    @staticmethod
    def _parse_function_format(content: str) -> list[ToolCall]:
        """Parse function call format: tool_name(param1="value1", param2="value2")"""
        tool_calls = []
        
        # pattern to match function calls
        func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.finditer(func_pattern, content)
        
        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # skip common words that aren't tools
            if tool_name.lower() in ['print', 'len', 'str', 'int', 'float', 'list', 'dict']:
                continue
            
            params = ToolCallParser._parse_parameters(params_str)
            
            tool_calls.append(ToolCall(
                tool_name=tool_name,
                parameters=params
            ))
        
        return tool_calls
    
    @staticmethod
    def _parse_structured_format(content: str) -> list[ToolCall]:
        """Parse structured format: [TOOL:tool_name] parameters"""
        tool_calls = []
        
        # pattern to match [TOOL:name] format
        struct_pattern = r'\[TOOL:(\w+)\]\s*([^\n\r]*)'
        matches = re.finditer(struct_pattern, content, re.IGNORECASE)
        
        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2).strip()
            
            params = ToolCallParser._parse_parameters(params_str)
            
            tool_calls.append(ToolCall(
                tool_name=tool_name,
                parameters=params
            ))
        
        return tool_calls
    
    @staticmethod
    def _parse_parameters(params_str: str) -> dict[str, Any]:
        """Parse parameter string into dictionary."""
        params = {}
        
        if not params_str.strip():
            return params
        
        # try to parse as key=value pairs
        param_pattern = r'(\w+)\s*=\s*(["\']?)([^,"\'\n]*)\2'
        matches = re.finditer(param_pattern, params_str)
        
        for match in matches:
            key = match.group(1)
            value = match.group(3).strip()
            
            # convert to appropriate type
            if value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            elif value.isdigit():
                params[key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                params[key] = float(value)
            else:
                params[key] = value
        
        # if no key=value pairs, treat the whole string as a single parameter
        if not params and params_str.strip():
            if any(word in params_str.lower() for word in ['expression', 'calculation', 'math']):
                params['expression'] = params_str.strip()
            elif any(word in params_str.lower() for word in ['note', 'content', 'text']):
                params['content'] = params_str.strip()
            else:
                params['input'] = params_str.strip()
        
        return params


class ToolCallFormatter:
    """Formats tool calls for LLM prompts."""
    
    @staticmethod
    def format_available_tools(tools: list) -> str:
        """Format available tools for the system prompt."""
        tool_descriptions = []
        
        for tool in tools:
            desc = f"**{tool.name}**: {tool.description}"
            
            if tool.parameters:
                params = []
                for param_name, param_info in tool.parameters.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    params.append(f"  - {param_name} ({param_type}): {param_desc}")
                
                if params:
                    desc += f"\n  Parameters:\n" + "\n".join(params)
            
            desc += f"\n  Usage: {tool.name}(param1=\"value1\", param2=\"value2\")"
            tool_descriptions.append(desc)
        
        return "\n\n".join(tool_descriptions)
    
    @staticmethod
    def format_tool_call_examples() -> str:
        """Provide examples of how to make tool calls."""
        return """
            Tool Call Examples:
            1. calculator(expression="25 * 4 + 18 / 3")
            2. note_taker(action="write", note_id="result", content="The answer is 106")
            3. note_taker(action="read", note_id="result")

            Always use the function call format: tool_name(parameter="value")
        """
