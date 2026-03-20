"""
Ablation Suite for Algorithmic Tasks
=====================================

A framework for testing model architectures on algorithmic reasoning tasks
with support for OOD length generalization testing.
"""

from .config import (
    TaskConfig, 
    ModelConfig, 
    TrainingConfig,
    MODEL_CONFIGS,
    TASK_CONFIGS,
    get_model_configs,
    get_task_configs,
)
from .train import train_model, Trainer, create_model
from .evaluate import Evaluator, EvalResults, quick_eval
from .metrics import MetricsTracker, RunResult

__all__ = [
    'TaskConfig', 'ModelConfig', 'TrainingConfig',
    'MODEL_CONFIGS', 'TASK_CONFIGS',
    'get_model_configs', 'get_task_configs',
    'train_model', 'Trainer', 'create_model',
    'Evaluator', 'EvalResults', 'quick_eval',
    'MetricsTracker', 'RunResult',
]
