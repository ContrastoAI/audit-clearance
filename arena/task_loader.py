import yaml
from pathlib import Path
from typing import Any
from .models import Task


class TaskLoader:
    """Loads tasks from YAML configuration files."""
    
    @staticmethod
    def load_task_from_yaml(file_path: str) -> Task:
        """Load a single task from a YAML file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Task file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return TaskLoader._create_task_from_dict(data)
    
    @staticmethod
    def load_tasks_from_directory(directory_path: str) -> list[Task]:
        """Load all tasks from YAML files in a directory."""
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Task directory not found: {directory_path}")
        
        tasks = []
        for yaml_file in path.glob("*.yaml"):
            try:
                task = TaskLoader.load_task_from_yaml(str(yaml_file))
                tasks.append(task)
            except Exception as e:
                print(f"Warning: Failed to load task from {yaml_file}: {e}")
        
        return tasks
    
    @staticmethod
    def _create_task_from_dict(data: dict[str, Any]) -> Task:
        """Create a Task object from dictionary data.""" 
        if not all([data.get('id'), data.get('name'), data.get('description'), data.get('prompt')]):
            raise ValueError("Task must have id, name, description, and prompt")
        
        return Task(**data)
