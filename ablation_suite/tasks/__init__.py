"""
Task implementations for ablation suite.
"""

from .base import BaseTask
from .sorting import SortingTask
from .modular_arith import ModularArithTask
from .reversal import ReversalTask
from .bitwise_add import BitwiseAddTask
from .parity import ParityTask

TASK_REGISTRY = {
    "sorting": SortingTask,
    "modular_arith": ModularArithTask,
    "reversal": ReversalTask,
    "bitwise_add": BitwiseAddTask,
    "parity": ParityTask,
}

def get_task(task_name: str, config):
    """Get a task instance by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name](config)

__all__ = [
    'BaseTask', 'SortingTask', 'ModularArithTask', 
    'ReversalTask', 'BitwiseAddTask', 'ParityTask',
    'get_task', 'TASK_REGISTRY'
]
