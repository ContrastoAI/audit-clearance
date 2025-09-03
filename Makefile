run:
	uv run python main.py --task-folder calculation_task --num-experiments 1 --give-hint --original-and-masked-prompt
run-all-w-base:
	uv run python main.py --task-folder task_pairs --num-experiments 1 --give-hint --original-and-masked-prompt