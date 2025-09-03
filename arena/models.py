from typing import Any, Literal
from datetime import datetime
from pathlib import Path
import csv
import json
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in the conversation."""
    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolCall(BaseModel):
    """A tool call made by an agent."""
    tool_name: str = Field(description="Name of the tool")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    result: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True


class Task(BaseModel):
    """A task definition with primary and optional secondary objectives."""
    id: str = Field(description="Task identifier")
    name: str = Field(description="Task name")
    description: str = Field(description="Task description")
    prompt: str = Field(description="Main task prompt for the agent")
    max_turns: int = Field(default=20, description="Maximum conversation turns")
    
    # side task (hidden from monitors)
    side_task: str
    side_prompt: str
    side_hint: str

    # mask entities
    mask_prompt: str
    mask_evaluation_info: str

    # evaluation prompts (for LLM-based evaluation)
    main_task_evaluation_prompt: str
    side_task_evaluation_prompt: str


class AgentResponse(BaseModel):
    """Response from an agent."""
    content: str = Field(description="Agent response content")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    thinking: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class EvaluationResult(BaseModel):
    """Result of task evaluation."""
    score: float = Field(description="Evaluation score (0-100)")
    explanation: str = Field(description="Explanation of the score")
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    input_transcript: str = Field(description="Transcript of the input evaluation")


class Experiment(BaseModel):
    """An experiment run."""
    id: str = Field(description="Experiment identifier")
    task: Task = Field(description="Task being executed")
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    agent_responses: list[AgentResponse] = Field(default_factory=list)
    completed: bool = False
    
    # evaluation results
    main_task_score: float | None = None
    side_task_score: float | None = None
    side_task_evaluation: EvaluationResult | None = None
    
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    def export_to_csv(self, filepath: str) -> None:
        """Export experiment data to CSV file using model dump.
        
        Args:
            filepath: Path where to save the CSV file
        """
        data = self.model_dump()
        
        base_info = {
            'experiment_id': data['id'],
            'task_id': data['task']['id'],
            'task_name': data['task']['name'],
            'completed': data['completed'],
            'main_task_score': data['main_task_score'],
            'side_task_score': data['side_task_evaluation']['score'],
            'is_masked': data['task']['mask_prompt'] is not None,
            'start_time': str(data['start_time']) if data['start_time'] else None,
            'end_time': str(data['end_time']) if data['end_time'] else None,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=base_info.keys())
            if csvfile.tell() == 0:
                writer.writeheader()
            pos = csvfile.tell()
            try:
                writer.writerow(base_info)
            except Exception as e:
                # revert to previous position if write fails
                csvfile.seek(pos)
                csvfile.truncate()
                raise Exception(f"Failed to write base info to CSV: {str(e)}")

        Path(f"{filepath.replace('.csv', '')}_messages.json").parent.mkdir(parents=True, exist_ok=True)
        with open(f"{filepath.replace('.csv', '')}_messages.json", 'a', newline='', encoding='utf-8') as jsonfile:
            json_data = {
                'experiment_id': data['id'],
                'task_id': data['task']['id'], 
                'messages': data['messages']
            }
            json_data['messages'] = [
                {**msg, 'timestamp': str(msg['timestamp'])}
                for msg in json_data['messages']
            ]
            pos = jsonfile.tell()
            try:
                json.dump(json_data, jsonfile)
                jsonfile.write('\n')
            except Exception as e:
                jsonfile.seek(pos)
                jsonfile.truncate()
                raise Exception(f"Failed to write messages to JSON: {str(e)}")


class ExperimentConfig(BaseModel):
    """Configuration for running experiments."""
    num_experiments: int = Field(default=1, description="Number of experiments to run")
    max_concurrent: int = Field(default=1, description="Maximum concurrent experiments")
    model_name: str = Field(default="gpt-4", description="LLM model to use")
    temperature: float = Field(default=0.7, description="Generation temperature")
    max_tokens: int = Field(default=4000, description="Maximum tokens per response")
    give_hint: bool = Field(default=False, description="Whether to give hints to the agent")
    allow_thinking: bool = Field(default=True, description="Whether to allow thinking tags")
