import argparse
from pathlib import Path
from arena.models import ExperimentConfig
from arena.runner import TaskRunner
from arena.environment import SimpleEnvironment
from arena.task_loader import TaskLoader
from arena.config import settings
from arena.llm import LLMFactory


def main():
    parser = argparse.ArgumentParser(description="Run Arena experiments")
    parser.add_argument("--num-experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--model", type=str, default=settings.default_model, help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=settings.temperature, help="Generation temperature")
    parser.add_argument("--give-hint", action="store_true", help="Give hints to the agent")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking tags")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum turns per task")
    parser.add_argument("--task-file", type=str, help="Path to YAML task file")
    parser.add_argument("--task-folder", type=str, help="Task folder name in task_pairs/ (use 'task_pairs' to run all tasks)")
    parser.add_argument("--use-mock", action="store_true", help="Use mock LLM instead of OpenAI API")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel (only for meta task runner)")
    parser.add_argument("--max-workers", type=int, default=3, help="Maximum parallel workers (default: 3)")
    parser.add_argument("--original-and-masked-prompt", action="store_true", help="Original and masked prompt for the task")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        num_experiments=args.num_experiments,
        model_name=args.model,
        temperature=args.temperature,
        give_hint=args.give_hint,
        allow_thinking=not args.no_thinking
    )
    
    llm = LLMFactory.create_llm(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )

    runner = TaskRunner(
        environment=SimpleEnvironment("test_env"),
        llm=llm
    )

    l_tasks = []

    if args.task_file:
        l_tasks.append(args.task_file)       
    elif args.task_folder:
        if args.task_folder == "task_pairs":
            print(f"Target directory: task_pairs/")
            print(f"Parallel execution: {'Enabled' if args.parallel else 'Disabled'}")
            if args.parallel:
                print(f"Max workers: {args.max_workers}")

            for task_file in Path("task_pairs").rglob("*.yaml"):
                l_tasks.append(task_file)
        else:
            task_file = Path("task_pairs") / args.task_folder / "task.yaml"
            l_tasks.append(task_file)
    else:
        raise ValueError("No task file or task folder provided")
    

    for task_file in l_tasks:
        if task_file.exists():
            task = TaskLoader.load_task_from_yaml(str(task_file))
            task.max_turns = args.max_turns
        else:
            raise FileNotFoundError(f"Error: Task file not found: {task_file}")
        
        if task.mask_prompt is not None and args.original_and_masked_prompt:
            base_task = task.model_copy()
            base_task.mask_prompt = None
            experiments = runner.run_experiments(base_task, config)
            [experiment.export_to_csv(f"exports/experiments.csv") for experiment in experiments]
            runner.print_experiment_results(experiments, base_task)

        print(
            f"Running {config.num_experiments} experiment(s) with model {config.model_name}\n"
            f"Task: {task.name}\n"
            f"Description: {task.description}\n"
            f"{'-' * 50}"
        )
         
        experiments = runner.run_experiments(task, config)

        [experiment.export_to_csv(f"exports/experiments.csv") for experiment in experiments]

        runner.print_experiment_results(experiments, task)


if __name__ == "__main__":
    main()
